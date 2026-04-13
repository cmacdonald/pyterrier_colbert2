import torch
import torch.nn.functional as F
import pandas as pd
import pyterrier as pt
from colbert.modeling.tokenization import DocTokenizer

def make_initial_stage(factory, dataset, *, top_psg=5, output_exptok=False):
    """
    返回一个 PyTerrier transformer:
      输入:  含有 ['qid','query'] 的 DataFrame (所有 topics)
      输出:  每个 query 一行的 DataFrame，列包括:
             ['qid','query','Q','pids','base_scores','V','lens','codes','wpids']

    用法示例:
      initial_stage = make_initial_stage(factory, dataset, top_psg=3)
      df_stage1 = initial_stage(topics2019)
    """

    # 取 config 给 DocTokenizer
    cfg = (getattr(factory.searcher, "config", None)
           or getattr(factory.searcher.ranker, "config", None)
           or getattr(factory.searcher.ranker, "colbert_config", None))
    doc_tok = DocTokenizer(config=cfg)
    embS = factory.searcher.ranker.embeddings_strided

    @torch.no_grad()
    def _initial(dfq: pd.DataFrame) -> pd.DataFrame:
        """
        dfq: 单个 query 的 DataFrame (by_query 保证了这一点)
        返回: 单行 DataFrame，包含本 query 的 Q / pids / V / codes 等
        
        """
        qid, qtext = dfq.iloc[0]["qid"], dfq.iloc[0]["query"]

        # 1) encode 原始查询
        Q = factory.searcher.encode([qtext]).squeeze(0).to(torch.float32)
        Q = F.normalize(Q, p=2, dim=-1)

        # 2) dense PLAID 初始检索，得到 PRF docs
        pids, ranks, scores = factory.searcher.dense_search(Q.unsqueeze(0), k=top_psg)

        if not pids:
            # 没有 PRF 文档时，用 None 填充，后续 PRF_stage 可以直接 fall back
            return pd.DataFrame([{
                "qid": qid,
                "query": qtext,
                "Q": Q,
                "pids": None,
                "base_scores": None,
                "V": None,
                "lens": None,
                "codes": None,
                "wpids": None,
            }])

        base_scores = torch.tensor(scores, dtype=torch.float32, device="cpu")

        # 3) PRF docs 的 token 向量 + codes
        V, lens = embS.lookup_pids(pids)  # V: [sumL, d]
        if V is None or V.numel() == 0:
            return pd.DataFrame([{
                "qid": qid,
                "query": qtext,
                "Q": Q,
                "pids": None,
                "base_scores": None,
                "V": None,
                "lens": None,
                "codes": None,
                "wpids": None,
            }])

        V = F.normalize(V.to(torch.float32), p=2, dim=-1)
        codes, _ = embS.lookup_codes(pids)  # [sumL]
        assert int(codes.numel()) == int(V.size(0))

        wpids = None
        if output_exptok:
            # 从 corpus 还原 WP id，并用 keep_mask 对齐
            wpids_trim, keep_mask_cpu = build_wpids_and_keepmask_from_corpus(
                factory, dataset, pids, lens, doc_tok
            )

            if keep_mask_cpu.sum().item() == 0:
                return pd.DataFrame([{
                    "qid": qid,
                    "query": qtext,
                    "Q": Q,
                    "pids": None,
                    "base_scores": None,
                    "V": None,
                    "lens": None,
                    "codes": None,
                    "wpids": None,
                }])

            if V.is_cuda:
                V = V[keep_mask_cpu.to(V.device)]
            else:
                V = V[keep_mask_cpu]
            codes = codes[keep_mask_cpu]
            wpids = wpids_trim   # 与裁剪后的 V / codes 对齐

        return pd.DataFrame([{
            "qid": qid,
            "query": qtext,
            "Q": Q,
            "pids": pids,
            "base_scores": base_scores,
            "V": V,
            "lens": lens,
            "codes": codes,
            "wpids": wpids,
        }])

    # 关键：像 PRF_stage 一样，用 by_query 封装，自动遍历所有 topics
    return pt.apply.by_query(_initial, add_ranks=False)


import torch
import pandas as pd
import pyterrier as pt

# 需要从你的 plaid_prf_tools 中 import 这些函数：
# from plaid_prf_tools import (
#     compute_default_idf, tf_from_codes,
#     weights_tf_idf, weights_rm1_from_prf, weights_rm3,
#     weights_bo1, weights_dfr_rsj,
#     rel_from_code, mmr_select_unified,
# )

def make_prf_stage_from_stage1(
    *,
    idf_map, N_global, eps=1.0, add_one=True,
    top_exp=16, beta=0.4,
    lambda_div=0.3, lambda_q=0.3,
    dedup_same_wp=True,
    mmr_selection=True,
    weighting="tf-idf",       # 'tf-idf' | 'rm1' | 'rm3' | 'bo1' | 'dfr'
    cf_map=None, total_tokens=None,
    rm3_lambda=0.5, temperature=1.0,
    df_map=None, N_docs=None,
    output_exptok=False, doc_tok=None,   # 如需把 wpids 反映射成 token 文本
):
    """
    构造一个 PyTerrier transformer：
      输入：来自 initial_stage 的 DataFrame（每行一个 query，包含 Q, pids, base_scores, V, lens, codes, wpids）
      输出：每行一个 query，包含扩展后的 query_vec，以及若干 debug 列。

    用法：
      prf_stage = make_prf_stage_from_stage1(...)
      df_qe = prf_stage(df_stage1)
    """
    default_idf = compute_default_idf(N_global, eps, add_one)

    @torch.no_grad()
    def _prf(dfq: pd.DataFrame) -> pd.DataFrame:
        # by_query 保证 dfq 只对应一个 qid
        row = dfq.iloc[0]

        qid        = row["qid"]
        qtext      = row["query"]
        Q          = row["Q"]
        pids       = row["pids"]
        V          = row["V"]
        lens       = row["lens"]
        codes      = row["codes"]
        wpids      = row["wpids"]
        base_scores = row["base_scores"]

        # 1) 若 initial stage 没拿到 PRF 文档，直接回退到原始 Q
        if (pids is None) or (V is None) or (getattr(V, "numel", lambda:0)() == 0):
            return pd.DataFrame([{
                "qid": qid,
                "query": qtext,
                "query_vec": Q.unsqueeze(0),  # [1, Lq, d] 形式给 PLAID
                "n_exp": 0,
                "lambda_div": float(lambda_div),
                "exp_idx": [],
                "exp_wpids": None,
                "exp_wptoks": None,
                "exp_codes": None,
            }])

        # ----------------- 以下部分基本是你原来 QE 函数的 5)~8) -----------------

        # 2) 统计 PRF codes 的 tf
        tf_map = tf_from_codes(codes)

        # 3) 按不同 weighting 方式计算 weights_by_code
        if weighting == "tf-idf":
            assert idf_map is not None, "idf_map is required for tf-idf weighting"
            weights_by_code = weights_tf_idf(tf_map, idf_map=idf_map, default_idf=default_idf)

        elif weighting == "rm1":
            weights_by_code = weights_rm1_from_prf(
                codes, lens, base_scores,
                temperature=temperature,
                normalize_out=True
            )

        elif weighting == "rm3":
            assert idf_map is not None, "idf_map is required for RM3 (tf-idf 部分)"
            tfidf_w = weights_tf_idf(tf_map, idf_map=idf_map, default_idf=default_idf)
            rm1_w   = weights_rm1_from_prf(
                codes, lens, base_scores,
                temperature=temperature,
                normalize_out=True
            )
            weights_by_code = weights_rm3(tfidf_w, rm1_w, lam=rm3_lambda)

        elif weighting == "bo1":
            assert (cf_map is not None) and (total_tokens is not None), \
                "bo1 weighting 需要 cf_map 和 total_tokens"
            weights_by_code = weights_bo1(tf_map, cf_map=cf_map, total_tokens=total_tokens)

        elif weighting == "dfr":
            assert df_map is not None, "dfr weighting 需要 df_map"
            weights_by_code = weights_dfr_rsj(tf_map, df_map=df_map, N_docs=N_docs)

        else:
            raise ValueError(f"Unknown weighting: {weighting}")

        # 4) Dict -> Tensor & 归一化，得到 rel（每个 code 的重要性）
        rel = rel_from_code(
            codes,
            code_rel=weights_by_code,
            device=V.device,
            normalize=True
        )

        # 5) 按 MMR 或 top-k 选择 expansion 向量
        if mmr_selection:
            dedup_wp = wpids if (dedup_same_wp and (wpids is not None)) else None
            selected = mmr_select_unified(
                V, rel,
                top_k=top_exp,
                Q=Q,                     # Q 形状 [Lq, d]
                lambda_uni=lambda_div,
                dedup_wpids=dedup_wp
            )
        else:
            k = min(top_exp, len(rel))
            _, selected = torch.topk(rel, k=k)

        if len(selected) == 0:
            return pd.DataFrame([{
                "qid": qid,
                "query": qtext,
                "query_vec": Q.unsqueeze(0),
                "n_exp": 0,
                "lambda_div": float(lambda_div),
                "exp_idx": [],
                "exp_wpids": None,
                "exp_wptoks": None,
                "exp_codes": None,
            }])

        sel_idx = torch.as_tensor(selected, dtype=torch.long, device=V.device)
        E = beta * V[sel_idx]  # [K, d]

        # 6) 拼接扩展向量，构造新的 query_vec
        Q_new = torch.cat(
            [Q, E.to(Q.device, dtype=Q.dtype)],
            dim=0
        ).unsqueeze(0)          # [1, Lq+K, d]

        # 7) 质性输出：wpids / tokens / codes（可选）
        exp_wpids = None
        exp_wptoks = None

        if wpids is not None:
            exp_wpids = [int(wpids[i].item()) for i in selected]
            if output_exptok and (doc_tok is not None):
                exp_wptoks = doc_tok.tok.convert_ids_to_tokens(exp_wpids)

        exp_codes = [int(codes[i].item()) for i in selected]

        return pd.DataFrame([{
            "qid": qid,
            "query": qtext,
            "query_vec": Q_new,
            "n_exp": int(E.size(0)),
            "lambda_div": float(lambda_div),
            "exp_idx": selected,
            "exp_wpids": exp_wpids,
            "exp_wptoks": exp_wptoks,
            "exp_codes": exp_codes,
        }])

    # 和 initial_stage 一样，用 by_query 封装
    return pt.apply.by_query(_prf, add_ranks=False)





import time
import numpy as np
import pandas as pd
import torch
import pyterrier as pt

def plaid_end_to_end_qe_profile_report(factory, k=1000, materialize=True) -> pt.Transformer:
    """
    Profile PLAID rerank with optional materialization:
    - materialize=False: return 1 row per query (fast; excludes pid->docno + DF build)
    - materialize=True:  return k rows per query (closer to 'more detailed' end-to-end)
    """
    assert factory.plaid_mode is True, "profile should only be used in PLAID mode"

    def _sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def _as_tensor_query(row, df_query):
        query_text = row["query"] if "query" in row else ""
        if "query_vec" in df_query.columns:
            qv = row["query_vec"]
            if isinstance(qv, np.ndarray):
                Q = torch.from_numpy(qv)
            elif torch.is_tensor(qv):
                Q = qv
            else:
                raise TypeError(f"Unsupported type for query_vec: {type(qv)}")
        else:
            Q = factory.searcher.encode([query_text])

        if torch.cuda.is_available():
            Q = Q.to("cuda", non_blocking=True)
        return Q.to(dtype=torch.float16)

    def _search(df_query):
        row = df_query.iloc[0]
        qid = row["qid"]
        lat = {}
        t_total0 = time.perf_counter()

        # ---- 1) query encoding / prep ----
        _sync()
        t0 = time.perf_counter()
        Q = _as_tensor_query(row, df_query)
        _sync()
        lat["lat_query_encoding_ms"] = (time.perf_counter() - t0) * 1000.0
        assert Q.dim() == 3

        # ---- 2) candidate generation ----
        _sync()
        t0 = time.perf_counter()
        pids, centroid_scores = factory.searcher.ranker.generate_candidates(factory.searcher.config, Q)
        _sync()
        lat["lat_candidate_generation_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- 3) score_pids total ----
        _sync()
        t0 = time.perf_counter()
        scores, pids = factory.searcher.ranker.score_pids(factory.searcher.config, Q, pids, centroid_scores)
        _sync()
        lat["lat_score_pids_total_ms"] = (time.perf_counter() - t0) * 1000.0

        # ---- 4) topk + materialize ----
        _sync()
        t0 = time.perf_counter()
        topk = min(k, len(pids))

        # 用 torch.topk（避免全量排序）
        top_scores, top_idx = torch.topk(scores, k=topk, largest=True)

        if materialize:
            # 一次性搬到 CPU，避免循环里 .item() 反复同步
            top_pids = pids[top_idx].detach().cpu().numpy()
            top_scores_cpu = top_scores.detach().cpu().numpy()

            results = []
            for rank, (pid, score) in enumerate(zip(top_pids, top_scores_cpu), start=1):
                docno = factory.docnos.fwd[int(pid)]
                results.append([qid, docno, float(score), rank])

            out = pd.DataFrame(results, columns=["qid", "docno", "score", "rank"])
        else:
            # 仍然做一次 CPU transfer，让 topk 这段更接近真实耗时（可选）
            _ = pids[top_idx].detach().cpu()
            _ = top_scores.detach().cpu()
            out = pd.DataFrame([{"qid": qid, "docno": "__PROFILE__", "score": 0.0, "rank": 1}])

        _sync()
        lat["lat_materialize_ms"] = (time.perf_counter() - t0) * 1000.0

        lat["lat_total_ms"] = (time.perf_counter() - t_total0) * 1000.0

        # 把延迟逐列添加到每一行（与 more detailed 输出形态一致）
        for key, val in lat.items():
            out[key] = float(val)

        return out

    return pt.apply.by_query(_search)



def make_prf_stage_from_stage1_new(
    *,
    idf_map, N_global, eps=1.0, add_one=True,
    top_psg: int = 3,         
    top_exp=16, beta=0.4,
    lambda_div=0.3, lambda_q=0.3,
    dedup_same_wp=True,
    mmr_selection=True,
    weighting="tf-idf",
    cf_map=None, total_tokens=None,
    rm3_lambda=0.5, temperature=1.0,
    df_map=None, N_docs=None,
    output_exptok=False, doc_tok=None,
):
    default_idf = compute_default_idf(N_global, eps, add_one)

    @torch.no_grad()
    def _prf(dfq: pd.DataFrame) -> pd.DataFrame:
        row = dfq.iloc[0]

        qid   = row["qid"]
        qtext = row["query"]
        Q     = row["Q"]

        # ========== 0) sanity ==========
        if ("pids" not in row) or (row["pids"] is None):
            return pd.DataFrame([{"qid": qid, "query": qtext, "query_vec": Q.unsqueeze(0), "n_exp": 0}])

        # ========== 1) passage-level truncate (top_psg) ==========
        pids_all = row["pids"]
        lens_all = row["lens"]
        base_all = row["base_scores"]

        pids = pids_all[:top_psg]
        lens = lens_all[:top_psg]
        base_scores = base_all[:top_psg]

        # lens 统一成 CPU long tensor，方便 sum
        if not torch.is_tensor(lens):
            lens = torch.tensor(list(lens), dtype=torch.long)
        else:
            lens = lens.to(torch.long)

        tok_cut = int(lens.sum().item())

        # ========== 2) token-level truncate (VERY IMPORTANT) ==========
        V_all = row["V"]
        codes_all = row["codes"]
        wpids_all = row.get("wpids", None)

        if (V_all is None) or (codes_all is None) or (tok_cut <= 0):
            return pd.DataFrame([{"qid": qid, "query": qtext, "query_vec": Q.unsqueeze(0), "n_exp": 0}])

        V     = V_all[:tok_cut]
        codes = codes_all[:tok_cut]

        # wpids 可能是 None（很常见），必须防御
        wpids = None
        if wpids_all is not None:
            wpids = wpids_all[:tok_cut]

        # ========== 3) 统计 codes 的 tf ==========
        tf_map = tf_from_codes(codes)

        # ========== 4) weighting ==========
        if weighting == "tf-idf":
            assert idf_map is not None
            weights_by_code = weights_tf_idf(tf_map, idf_map=idf_map, default_idf=default_idf)

        elif weighting == "rm1":
            weights_by_code = weights_rm1_from_prf(
                codes, lens, base_scores,
                temperature=temperature,
                normalize_out=True
            )

        elif weighting == "rm3":
            tfidf_w = weights_tf_idf(tf_map, idf_map=idf_map, default_idf=default_idf)
            rm1_w   = weights_rm1_from_prf(codes, lens, base_scores, temperature=temperature, normalize_out=True)
            weights_by_code = weights_rm3(tfidf_w, rm1_w, lam=rm3_lambda)

        elif weighting == "bo1":
            assert (cf_map is not None) and (total_tokens is not None)
            weights_by_code = weights_bo1(tf_map, cf_map=cf_map, total_tokens=total_tokens)

        elif weighting == "dfr":
            assert df_map is not None
            weights_by_code = weights_dfr_rsj(tf_map, df_map=df_map, N_docs=N_docs)

        else:
            raise ValueError(f"Unknown weighting: {weighting}")

        # ========== 5) rel per token ==========
        rel = rel_from_code(
            codes,
            code_rel=weights_by_code,
            device=V.device,
            normalize=True
        )

        # ========== 6) select expansion tokens ==========
        if mmr_selection:
            dedup_wp = wpids if (dedup_same_wp and (wpids is not None)) else None
            selected = mmr_select_unified(
                V, rel,
                top_k=top_exp,
                Q=Q,
                lambda_uni=lambda_div,
                dedup_wpids=dedup_wp
            )
        else:
            k = min(top_exp, len(rel))
            _, selected = torch.topk(rel, k=k)
            selected = selected.tolist()

        if len(selected) == 0:
            return pd.DataFrame([{"qid": qid, "query": qtext, "query_vec": Q.unsqueeze(0), "n_exp": 0}])

        sel_idx = torch.as_tensor(selected, dtype=torch.long, device=V.device)
        E = beta * V[sel_idx]
        Q_new = torch.cat([Q, E.to(Q.device, dtype=Q.dtype)], dim=0).unsqueeze(0)

        # optional debug outputs
        exp_wpids = None
        exp_wptoks = None
        if wpids is not None:
            exp_wpids = [int(wpids[i].item()) for i in selected]
            if output_exptok and (doc_tok is not None):
                exp_wptoks = doc_tok.tok.convert_ids_to_tokens(exp_wpids)

        exp_codes = [int(codes[i].item()) for i in selected]

        return pd.DataFrame([{
            "qid": qid,
            "query": qtext,
            "query_vec": Q_new,
            "n_exp": int(E.size(0)),
            "top_psg": int(top_psg),
            "top_exp": int(top_exp),
            "beta": float(beta),
            "exp_idx": selected,
            "exp_wpids": exp_wpids,
            "exp_wptoks": exp_wptoks,
            "exp_codes": exp_codes,
        }])

    return pt.apply.by_query(_prf, add_ranks=False)




import numpy as np
import pandas as pd
import torch
import time

LAT_COLS = [
    "lat_query_encoding_ms",
    "lat_candidate_generation_ms",
    "lat_score_pids_total_ms",
    # "lat_topk_ms",
    "lat_materialize_ms",
    "lat_total_ms",
]

def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def latency_profile_n_runs(prof_transformer, df_in, n_runs=5, warmup=1,
                   agg_over_queries="median", agg_over_runs="median",
                   lat_cols=LAT_COLS):
    """
    返回：pd.Series，每个 lat_col 一个标量（ms/query）
    - agg_over_queries: "mean" or "median"：一次 run 内，对所有 query 聚合
    - agg_over_runs:    "mean" or "median"：多次 run 之间再聚合
    """

    # --- warmup ---
    for _ in range(warmup):
        _ = prof_transformer(df_in)
        sync()

    # --- repeated runs ---
    per_run_stats = []
    for _ in range(n_runs):
        sync()
        df_prof = prof_transformer(df_in)     # 一次 run，返回每个 query 1 行
        sync()

        if agg_over_queries == "mean":
            s = df_prof[lat_cols].mean()
        elif agg_over_queries == "median":
            s = df_prof[lat_cols].median()
        else:
            raise ValueError("agg_over_queries must be 'mean' or 'median'")

        per_run_stats.append(s)

    per_run_df = pd.DataFrame(per_run_stats)  # shape: (n_runs, len(lat_cols))

    if agg_over_runs == "mean":
        out = per_run_df.mean()
    elif agg_over_runs == "median":
        out = per_run_df.median()
    else:
        raise ValueError("agg_over_runs must be 'mean' or 'median'")

    return out

