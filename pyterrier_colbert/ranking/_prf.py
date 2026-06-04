import math, torch
import torch.nn.functional as F
from typing import Literal, Optional, Iterable, List, Tuple
import pyterrier as pt
import pandas as pd
from collections import Counter, defaultdict
import numpy as np

import math
from typing import List, Optional
import torch

def mmr_select_unified(
    V: torch.Tensor,                 # [M, d] PRF token vectors (L2-normalised, same device)
    rel: torch.Tensor,               # [M]    relevance scores (same device as V)
    top_k: int,
    Q: torch.Tensor,                 # [n_q, d] original query token vectors (L2-normalised)
    lambda_uni: float = 0.3,         # 单一λ：同时控制对 S∪Q 的多样性惩罚
    dedup_wpids: Optional[torch.Tensor] = None,  # [M] CPU long; WordPiece ids for可选去重
) -> List[int]:
    """
    Unified MMR:
      mmr(i) = (1 - λ) * rel(i) - λ * max_{x in (S ∪ Q)} cos(v_i, x)

    其中 S 为已选扩展集合（动态增长），Q 为原始查询 token 集合（固定）。
    Q 的惩罚从一开始就生效；S 的惩罚在选择后逐步生效。
    """
    DEV = V.device
    M = int(V.size(0))
    k = min(top_k, M)

    Qd = Q.to(device=DEV, dtype=V.dtype)

    selected: List[int] = []
    Sel = None  # [t, d] 已选向量
    selected_wp = set()

    # 预计算 v_i 对查询 Q 的最大相似度： sim_q[i] = max_j <v_i, q_j>
    sim_q = (V @ Qd.T).max(dim=1).values if Qd.numel() > 0 else torch.zeros(M, device=DEV, dtype=V.dtype)

    for _ in range(k):
        # v_i 对已选扩展 S 的最大相似度（若 S 为空则为0）
        if Sel is None:
            sim_sel = torch.zeros(M, device=DEV, dtype=V.dtype)
        else:
            sim_sel = (V @ Sel.T).max(dim=1).values

        # 合并抑制源：对 S∪Q 的最大相似度
        sim_combined = torch.maximum(sim_sel, sim_q)

        # 统一λ的 MMR
        mmr = (1.0 - lambda_uni) * rel - lambda_uni * sim_combined

        # ban 已选
        if selected:
            taken = torch.as_tensor(selected, dtype=torch.long, device=DEV)
            mmr.index_fill_(0, taken, float('-inf'))

        # 可选：WP 去重
        if dedup_wpids is not None and selected_wp:
            dup_mask = torch.tensor(
                [int(w.item()) in selected_wp for w in dedup_wpids],
                dtype=torch.bool, device=DEV
            )
            mmr = torch.where(dup_mask, torch.tensor(float('-inf'), device=DEV), mmr)

        i = int(torch.argmax(mmr).item())
        if not math.isfinite(float(mmr[i].item())):
            break

        selected.append(i)
        if dedup_wpids is not None:
            selected_wp.add(int(dedup_wpids[i].item()))
        Sel = V[i:i+1] if Sel is None else torch.cat([Sel, V[i:i+1]], dim=0)

    return selected

import pandas as pd, torch
import torch.nn.functional as F
import pyterrier as pt
from colbert.modeling.tokenization import DocTokenizer


def plaid_prf_end_to_end(factory, k=1000, **kwargs):
    return plaid_prf(factory, **kwargs) >> factory.end_to_end(k=k, query_encoded=True)

def plaid_prf(
    factory, 
    *,
    # PRF & expansion:
    top_psg=5, top_exp=16, beta=0.4, lambda_div=0.3,
    lambda_q=0.3, # TODO can this be removed.
    dedup_same_wp: bool = True,
    mmr_selection: bool = True,
    output_exptok: bool = False,
    dataset = None,
    # resources for  weighting
    weighting: Literal['tf-idf', 'rm1', 'rm3', 'bo1', 'dfr_rsj'] = "tf-idf",
    rm3_lambda: float = 0.5,       # RM3 mixing coefficient
    temperature: float = 1.0,      # RM1 softmax temperature
):
    idf_map, df_map, cf_map, stats = build_global_code_stats(factory.searcher.ranker.index)
    N_global = stats['N']
    N_docs = stats['N'] # TODO do we need both vars, this one is for dfr_rsj
    total_tokens = stats['tokens']
    eps = stats['eps']
    add_one = stats['add_one']
    embS = factory.searcher.ranker.embeddings_strided
    default_idf = compute_default_idf(N_global, eps, add_one)

    # tokenizer（仅用于 id->token 文本）
    cfg = (getattr(factory.searcher, "config", None)
           or getattr(factory.searcher.ranker, "config", None)
           or getattr(factory.searcher.ranker, "colbert_config", None))
    doc_tok = DocTokenizer(config=cfg)  # .tok 是 HF tokenizer

    @torch.no_grad()
    def _expand(dfq):
        qid, qtext = dfq.iloc[0]["qid"], dfq.iloc[0]["query"]

        # 1) encode query
        Q = factory.searcher.encode([qtext]).squeeze(0).to(torch.float32)
        Q = F.normalize(Q, p=2, dim=-1)

        # 2) PRF pids via dense PLAID
        pids, ranks, scores = factory.searcher.dense_search(Q.unsqueeze(0), k=top_psg)
        if not pids:
            return pd.DataFrame([{"qid": qid, "query": qtext, "query_vec": Q.unsqueeze(0)}])
        # caption scores for RM1
        base_scores = torch.tensor(scores, dtype = torch.float32, device = 'cpu')

        # 3) Gather PRF token vectors and compressed codes
        V, lens = embS.lookup_pids(pids)                 # V: [sumL, d]
        if V is None or V.numel() == 0:
            return pd.DataFrame([{"qid": qid, "query": qtext, "query_vec": Q.unsqueeze(0)}])
        V = F.normalize(V.to(torch.float32), p=2, dim=-1)
        codes, _ = embS.lookup_codes(pids)               # [sumL]
        assert int(codes.numel()) == int(V.size(0)) # make sure positionally algined codes and vec
        # print(f"codes before are: {codes}")

        # 4) recover corresponding WP oneline，keep_mask is used to keep codes and wp ids aligned
        if output_exptok:
            assert dataset is not None, "dataset is required for output_exptok to retrieve WP tokens"
            wpids_trim, keep_mask_cpu = build_wpids_and_keepmask_from_corpus(
                factory, dataset, pids, lens, doc_tok
            )
            
            # 若一个都没保住，直接回退到原始查询
            if keep_mask_cpu.sum().item() == 0:
                return pd.DataFrame([{"qid": qid, "query": qtext, "query_vec": Q.unsqueeze(0)}])
    
            
            # 先在 V 上应用 keep_mask（注意设备）
            if V.is_cuda:
                V = V[keep_mask_cpu.to(V.device)]
            else:
                V = V[keep_mask_cpu]
            codes = codes[keep_mask_cpu]          # CPU
            wpids = wpids_trim                # [sum(K_i)]，与裁剪后的 V/codes 完全对齐
        else:
            wpids = None
        
        # 5) Compute weights by the selected scheme (per-code)
        tf_map = tf_from_codes(codes)  # PRF counts per code
        

        # 6) weighting methods
        if weighting == "tf-idf":
            assert idf_map is not None, "idf_map is required for tf-idf weighting"
            weights_by_code = weights_tf_idf(tf_map, idf_map=idf_map, default_idf=default_idf)

        elif weighting == "rm1":
            # needs per-doc lens and base_scores
            weights_by_code = weights_rm1_from_prf(
                codes, lens, base_scores, temperature=temperature, normalize_out=True
            )

        elif weighting == "rm3":
            assert idf_map is not None, "idf_map is required for the tf-idf part of RM3"
            tfidf_w = weights_tf_idf(tf_map, idf_map=idf_map, default_idf=default_idf)
            rm1_w   = weights_rm1_from_prf(
                codes, lens, base_scores, temperature=temperature, normalize_out=True
            )
            weights_by_code = weights_rm3(tfidf_w, rm1_w, lam=rm3_lambda)

        elif weighting == "bo1":
            assert (cf_map is not None) and (total_tokens is not None), \
                "bo1 weighting needs cf_map and total_tokens"
            weights_by_code = weights_bo1(tf_map, cf_map=cf_map, total_tokens=total_tokens)

        elif weighting == "dfr":
            # simple DFR-style proxy (RSJ-IDF)
            assert df_map is not None, "dfr_rsj weighting needs df_map (document frequencies)"
            weights_by_code = weights_dfr_rsj(tf_map, df_map=df_map, N_docs=N_docs)

        else:
            raise ValueError(f"Unknown weighting: {weighting}")
            
        #  Dict to Tensor AND normalize
        rel = rel_from_code(codes, code_rel =weights_by_code , device=V.device, normalize=True)

        # 6） selection methods
        if mmr_selection:

            # MMR 多样性选择（可选去重相同 WP）
            dedup_wp = wpids if (dedup_same_wp and (wpids is not None)) else None
            # selected = mmr_select(V, rel, top_k=top_exp, lambda_div=lambda_div, dedup_wpids=dedup_wp)
            # selected = mmr_select_with_query(V, rel, top_k = top_exp, Q=Q.squeeze(0), lambda_div=lambda_div, lambda_q = lambda_q, dedup_wpids = dedup_wp)
            selected = mmr_select_unified(V, rel, top_k = top_exp, Q=Q.squeeze(0), lambda_uni=lambda_div, dedup_wpids = dedup_wp)
        # print("mmr_sel:", type(selected), selected)
            
        else:
            # only select top k from high to low. rel: torch.tensor
            # print("default selection: {type(rel)}")
            k = min(top_exp, len(rel))
            rel, selected = torch.topk(rel,k=k)
            # print("topk_sel:",type(selected), selected)
            
            
        
        if len(selected) == 0:
            return pd.DataFrame([{"qid": qid, "query": qtext, "query_vec": Q.unsqueeze(0)}])
            
        sel_idx = torch.as_tensor(selected, dtype=torch.long, device=V.device)
        E = beta * V[sel_idx]                                        # on DEV
        Q_new = torch.cat([Q, E.to(Q.device, dtype=Q.dtype)], dim=0).unsqueeze(0)
            
        
        # 7) 质性输出（WP）
        if wpids is not None:
            exp_wpids  = [int(wpids[i].item()) for i in selected]
            exp_wptoks = doc_tok.tok.convert_ids_to_tokens(exp_wpids)  # 原始 WP（含 ## & specials）
        else:
            exp_wpids, exp_wptoks = None, None

        exp_codes = [int(codes[i].item()) for i in selected]
        # for i in selected:
        #     print(i, int(codes[i]))
        
        if output_exptok:
            print(f"QUERY: {qtext} \n EXP_TOKS: {exp_wptoks} \n EXP_codes: {exp_codes}")

        
        return pd.DataFrame([{
            "qid": qid,
            "query": qtext,
            "query_vec": Q_new,
            "n_exp": int(E.size(0)),
            "lambda_div": float(lambda_div),
            "exp_idx": selected,          # 相对当前（过滤后的）候选数组的下标
            "exp_wpids": exp_wpids,
            "exp_wptoks": exp_wptoks,
            "exp_codes": exp_codes
        }])

    return pt.apply.by_query(_expand, add_ranks=False)





##########################
# build global stats
##########################

import math
from collections import defaultdict
from typing import Dict, Tuple
import torch
from tqdm import tqdm
import json

# @torch.no_grad()
def build_global_code_stats(index : 'ColBERTv2Index',
                            batch_size: int = 1024,
                            eps: float = 1.0,
                            add_one: bool = True
                           ) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int], dict]:
    """
    Based on the compression codes in the index, calculate:  
    - `idf_map[c]`: float - the IDF of this code  
    - `df_map[c]`: int - the number of documents in which this code appears (counted at most once per document)  
    - `cf_map[c]`: int - the total number of occurrences of this code in the entire collection (can be >1 per document)  
    - Statistical information: includes total number of documents N, total number of tokens, average length avg_len, etc.  
    
    Rely only on `embeddings_strided.lookup_codes(pids)` and `lens`.
    """

    STATS_FILES = ["stats.json", "idf_map.json", "df_map.json", "cf_map.json"]
    exists_ok = True
    for fname in STATS_FILES:
        if not (index.path / fname).exists():
            exists_ok = False
    if exists_ok:
        _read_json = lambda path: json.load(open(path, "r", encoding="utf-8") )
        idf_map = _read_json(index.path / "idf_map.json")
        df_map  = _read_json(index.path / "df_map.json")
        cf_map  = _read_json(index.path / "cf_map.json")
        stats   = _read_json(index.path / "stats.json")
        return idf_map, df_map, cf_map, stats
    
    embS = index.searcher.ranker.embeddings_strided           # ResidualEmbeddingsStrided
    strided = embS.codes_strided                                # 内部 StridedTensor（有 lengths）
    
    N_docs = strided.lengths.numel() if hasattr(strided.lengths, "numel") else len(strided.lengths)

    df_map = defaultdict(int)   # code -> DF
    cf_map = defaultdict(int)   # code -> CF
    total_tokens = 0            # 累计 codes 数（总 token 数）

    
    with tqdm(total=N_docs, desc="Scanning codes (by docs)", unit="doc") as pbar:
        for start in range(0, N_docs, batch_size):
            end  = min(start + batch_size, N_docs)

            # 显式用 long 张量，避免 _prepare_lookup 的 dtype/assert 问题
            pids = torch.arange(start, end, dtype=torch.long, device="cpu")

            codes, lens = embS.lookup_codes(pids)  # codes: 1D [sum(L_i)] ; lens: per-doc lengths
            # 空批或无效直接跳过
            if (codes is None) or (lens is None):
                pbar.update(end - start)
                continue
            if hasattr(codes, "numel") and codes.numel() == 0:
                pbar.update(end - start)
                continue

            # 统计总 token 数（lens 可能是 list/ndarray/tensor，统一安全求和）
            batch_tokens = int(torch.as_tensor(lens).sum().item())
            total_tokens += batch_tokens

            # CPU 上处理
            codes = codes.cpu()
            lens  = [int(x) for x in torch.as_tensor(lens).cpu().tolist()]

            off = 0
            for L in lens:
                if L > 0:
                    doc_codes = codes[off:off+L]

                    # 同时获取唯一值与计数
                    uniq, counts = torch.unique(doc_codes, return_counts=True)

                    # DF: 每文仅计 1 次
                    for cid in uniq.tolist():
                        df_map[int(cid)] += 1

                    # CF: 计出现次数
                    for cid, cnt in zip(uniq.tolist(), counts.tolist()):
                        cf_map[int(cid)] += int(cnt)

                off += L

            pbar.update(end - start)

    # ---- 计算 IDF（以索引尺度 N_docs 为底）----
    idf_map = {
        c: (math.log((N_docs + eps) / (df + eps)) + (1.0 if add_one else 0.0))
        for c, df in df_map.items()
    }

    stats = {
        "N": N_docs,                              
        "tokens": total_tokens,
        "avg_len": (total_tokens / N_docs) if N_docs > 0 else 0.0,
        "eps": eps,
        "add_one": add_one,
        "num_codes": len(idf_map),
    }
    df_map = dict(df_map)
    cf_map = dict(cf_map)

    _write_json = lambda path, data: json.dump(data, open(path, "w", encoding="utf-8"))
    _write_json(index.path / "idf_map.json", idf_map)
    _write_json(index.path / "df_map.json", dict(df_map))
    _write_json(index.path / "cf_map.json", dict(cf_map))
    _write_json(index.path / "stats.json", stats)

    return idf_map, df_map, cf_map, stats

# ---------- IDF ----------
def compute_default_idf(N_global: int, eps: float = 1.0, add_one: bool = True) -> float:
    return math.log((N_global + eps) / (1 + eps)) + (1.0 if add_one else 0.0)
    
# ---------------------------
# Utilities 
# ---------------------------
def unit(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)
    
def rel_from_code(codes: torch.Tensor, code_rel: dict, device=None, normalize=True):
    rel = torch.tensor([float(code_rel.get(int(c.item()), 0.0)) for c in codes],
                       dtype=torch.float32, device=device or codes.device)
    if normalize and rel.numel() > 0:
        mn, mx = rel.min(), rel.max()
        rel = (rel - mn) / (mx - mn) if mx > mn else torch.zeros_like(rel)
    return rel

def offsets_from_lengths(lens: torch.Tensor) -> torch.Tensor:
    # lens: [num_docs] (CPU tensor)
    offs = torch.zeros(len(lens) + 1, dtype=torch.long)
    offs[1:] = torch.cumsum(lens.to(torch.long), dim=0)
    return offs

def tf_from_codes(codes_1d: torch.Tensor) -> dict:
    # codes_1d: [M] CPU Long 
    # calculate the term frequency for a given code:
    tf = Counter(codes_1d.tolist())
    return tf


def norm_positive(d: dict, eps: float = 1e-12) -> dict:
    s = float(sum(max(0.0, v) for v in d.values())) + eps
    return {k: max(0.0, v) / s for k, v in d.items()}

def to_per_occurrence(weights_by_code: dict, codes_1d: torch.Tensor):
    # Return list[(score, idx)] for each occurrence idx
    # codes_1d: CPU Long [M]
    # 把 dict 转换为list
    out = []
    get = weights_by_code.get
    for j, c in enumerate(codes_1d.tolist()):
        out.append((float(get(c, 0.0)), j))
    return out


# ---------------------------
# TF×IDF (term-level)
# ---------------------------
def weights_tf_idf(tf_by_code: dict, *, idf_map: dict, default_idf: float) -> dict:
    # idf_map can have string or int keys; normalize lookup
    get = lambda c: float(idf_map.get(c, idf_map.get(str(c), default_idf)))
    return {c: tf * get(c) for c, tf in tf_by_code.items()}


# ---------------------------
# RM1 (Lavrenko–Croft)
# P(w|R) = sum_d P(w|d) P(d|q)
# Here: P(w|d) ≈ tf_{w,d} / |d| over PRF; P(d|q) from softmax(base_scores)
# ---------------------------
def weights_rm1_from_prf(codes_1d: torch.Tensor,
                         lens: torch.Tensor,
                         base_scores: torch.Tensor,
                         temperature: float = 1.0,
                         normalize_out: bool = True) -> dict:
    """
    codes_1d: [M] CPU Long (token codes for PRF tokens)
    lens:     [D] CPU Long (token counts per PRF doc, order aligned with pids)
    base_scores: [D] (torch, CPU or GPU) retrieval scores for those PRF docs
    """
    D = len(lens)
    offs = offsets_from_lengths(lens).tolist()  # len D+1
    # soft P(d|q)
    scores = base_scores.detach().float()
    scores = scores - scores.max()                # stability
    Pdq = torch.softmax(scores / float(temperature), dim=0).cpu().numpy()  # [D]

    # accumulate P(w|R)
    PwR = defaultdict(float)
    for d in range(D):
        s, e = offs[d], offs[d+1]
        if e <= s: 
            continue
        doc_codes = codes_1d[s:e].tolist()
        Ld = e - s
        # P(w|d) ~ tf(w,d)/Ld
        tf_d = Counter(doc_codes)
        coef = float(Pdq[d]) / float(Ld)
        for w, tfwd in tf_d.items():
            PwR[w] += coef * float(tfwd)

    if normalize_out:
        PwR = norm_positive(PwR)
    return dict(PwR)


# ---------------------------
# RM3 = (1 - λ) * Base + λ * RM1
# Here we mix TF×IDF (normalized) with RM1
# ---------------------------
def weights_rm3(tfidf_w: dict, rm1_w: dict, lam: float = 0.5) -> dict:
    a = norm_positive(tfidf_w)
    b = norm_positive(rm1_w)
    return {k: (1.0 - lam) * a.get(k, 0.0) + lam * b.get(k, 0.0)
            for k in set(a) | set(b)}


# ---------------------------
# BO1 (Divergence from Randomness, parameter-free)
# Amati's BO1: w = tf_R*log2((1+F)/F) + log2(1+F),
# where F ~ cf/N_tokens (background probability)
# ---------------------------
def weights_bo1(tf_by_code: dict,
                *,
                cf_map: dict,            # collection frequency per code
                total_tokens: int,        # total tokens in collection
                eps: float = 1e-12) -> dict:
    out = {}
    for c, tfR in tf_by_code.items():
        cf = float(cf_map.get(c, cf_map.get(str(c), 0.0)))
        F = cf / max(1.0, float(total_tokens))
        # safe-guard
        F = max(F, eps)
        w = tfR * math.log2((1.0 + F) / F) + math.log2(1.0 + F)
        out[c] = w
    return out


# ---------------------------
# DFR (simple RSJ-style as a proxy): TF * RSJ-IDF
# RSJ-IDF = log( (N - df + 0.5) / (df + 0.5) )
# ---------------------------
def weights_dfr_rsj(tf_by_code: dict,
                    *,
                    df_map: dict,
                    N_docs: int,
                    eps: float = 0.5) -> dict:
    out = {}
    for c, tfR in tf_by_code.items():
        df = float(df_map.get(c, df_map.get(str(c), 0.0)))
        idf_rsj = math.log((N_docs - df + eps) / (df + eps))
        out[c] = tfR * idf_rsj
    return out

def  build_wpids_and_keepmask_from_corpus(
    factory, dataset, pids, lens, doc_tokenizer
):
    """
    Return:
      - wpids_all_trimmed: torch.LongTensor [sum(K_i)]
      - keep_mask_cpu:     torch.BoolTensor [sum(L_i)], cropping the concatenated axis of V/codes
    where K_i = min(L_i, T_i), and T_i is the online tokenization length.
    """
    # 取原文的工具
    def _get_text(docno):
        try:
            df = dataset.get_corpus()
            row = df.loc[df['docno'] == docno]
            if len(row) > 0: return row.iloc[0]['text']
        except Exception:
            pass
        for rec in dataset.get_corpus_iter():
            if rec.get('docno') == docno:
                return rec.get('text')
        return None

    L = lens.to(torch.long).cpu()
    offs = torch.zeros(len(L)+1, dtype=torch.long)
    offs[1:] = torch.cumsum(L, dim=0)
    sumL = int(offs[-1].item())

    keep_mask = torch.zeros(sumL, dtype=torch.bool)  #  V/codes cancatenate
    wp_chunks = []

    for i, pid in enumerate(pids):
        Li = int(L[i])
        if Li == 0:
            continue
        docno = factory.docnos.fwd[int(pid)]
        text  = _get_text(docno)
        if text is None:
            # no text, discard
            continue
        ids, _ = doc_tokenizer.tensorize([text])   #  tokenizer
        Ti = int(ids.size(1))
        Ki = min(Li, Ti)                           # align to min
        if Ki > 0:
            s = int(offs[i]); e = s + Ki
            keep_mask[s:e] = True
            wp_chunks.append(ids[0, :Ki].to(torch.long).cpu())

    wpids_all_trimmed = torch.cat(wp_chunks, dim=0) if wp_chunks else torch.empty(0, dtype=torch.long)

    return wpids_all_trimmed, keep_mask
    


# ---------- compute TF×IDF(code) relevance ----------
def rel_tf_idf_over_codes(
    codes: torch.Tensor,           # [M]  CPU
    idf_map: dict,
    default_idf: float,
    device: torch.device
) -> torch.Tensor:
    codes_list = codes.cpu().tolist()
    tf = {}
    for c in codes_list:
        tf[c] = tf.get(c, 0) + 1
    rel = torch.tensor([tf[c] * float(idf_map.get(str(c), default_idf)) for c in codes_list],
                       dtype=torch.float32, device=device)
    # Normalize to [0,1] (can be stably combined with similarity) preprocessing: V = l2_normalize(V), rel = minmax(rel) (to make scales comparable).
    if torch.isfinite(rel).any():
        rmin, rmax = float(rel.min().item()), float(rel.max().item())
        rel = (rel - rmin) / (rmax - rmin) if rmax > rmin else torch.zeros_like(rel)
    return rel

# ---------- MMR Selection ----------
def mmr_select(
    V: torch.Tensor,               # [M, d] 已 L2 归一化；在同一 device 上
    rel: torch.Tensor,             # [M]    与 V 同 device
    top_k: int,
    lambda_div: float = 0.3,
    dedup_wpids: Optional[torch.Tensor] = None,  # [M] CPU；用于避免相同 WP id（可为 None）
) -> List[int]:
    """
    Return: the selected index list (Python list[int], relative to the local indices of the current V)
    Algorithm：MMR(i) = (1-λ)*rel(i) - λ * max_{j∈S} cos(v_i, v_j)
    """
    DEV = V.device
    M = int(V.size(0))
    k = min(top_k, M)
    selected: List[int] = []
    Sel = None

    # 维护一个已选 WP id 的 set（如需去重）
    selected_wp = set()

    for _ in range(k):
        if Sel is None:
            mmr = (1.0 - lambda_div) * rel
        else:
            sim_max = (V @ Sel.T).max(dim=1).values
            mmr = (1.0 - lambda_div) * rel - lambda_div * sim_max

        # ban 已选
        if selected:
            taken = torch.as_tensor(selected, dtype=torch.long, device=DEV)
            mmr.index_fill_(0, taken, float('-inf'))

        # 可选：避免重复 WP id
        if (dedup_wpids is not None) and (len(selected_wp) > 0):
            dup_mask_cpu = torch.tensor([int(w.item()) in selected_wp for w in dedup_wpids], dtype=torch.bool)
            dup_mask = dup_mask_cpu.to(device=DEV)
            if dup_mask.any():
                mmr = torch.where(dup_mask, torch.tensor(float('-inf'), device=DEV), mmr)

        i = int(torch.argmax(mmr).item())
        if not math.isfinite(float(mmr[i].item())):
            break

        selected.append(i)
        if dedup_wpids is not None:
            selected_wp.add(int(dedup_wpids[i].item()))

        Sel = V[selected] if Sel is None else torch.cat([Sel, V[i:i+1]], dim=0)

    return selected



import math
from typing import List, Optional
import torch

def mmr_select_with_query(
    V: torch.Tensor,                 # [M, d]  PRF token vectors (L2-normalised, same device)
    rel: torch.Tensor,               # [M]     relevance scores (same device as V)
    top_k: int,
    Q: torch.Tensor,                 # [n_q, d] original query token vectors (L2-normalised)
    lambda_div: float = 0.3,         # weight for diversity vs rel (as before)
    lambda_q: float = 0.3,           # NEW: penalty strength vs original query directions
    dedup_wpids: Optional[torch.Tensor] = None,  # [M] CPU long; WordPiece ids for optional dedup
) -> List[int]:
    """
    MMR objective with two diversity terms:
      mmr(i) = (1 - λ_div) * rel(i)
               - λ_div * max_{e in S} cos(v_i, e)
               - λ_q   * max_{q in Q} cos(v_i, q)

    When S is empty, the second term vanishes; the query penalty is active from the start.
    """
    DEV = V.device
    M = int(V.size(0))
    k = min(top_k, M)

    # Ensure Q is on device and float
    Qd = Q.to(device=DEV, dtype=V.dtype)

    selected: List[int] = []
    Sel = None                      # [t, d] grows with selections

    # book-keeping for optional dedup by WordPiece id
    selected_wp = set()

    # precompute similarity to query once (max over query tokens for each PRF token)
    # sim_q[i] = max_j <v_i, q_j>
    sim_q = (V @ Qd.T).max(dim=1).values if Qd.numel() > 0 else torch.zeros(M, device=DEV, dtype=V.dtype)

    for _ in range(k):
        # max similarity to already-selected expansions (0 if none yet)
        if Sel is None:
            sim_sel = torch.zeros(M, device=DEV, dtype=V.dtype)
        else:
            sim_sel = (V @ Sel.T).max(dim=1).values

        # MMR with both penalties
        mmr = (1.0 - lambda_div) * rel - lambda_div * sim_sel - lambda_q * sim_q

        # ban already selected indices
        if selected:
            taken = torch.as_tensor(selected, dtype=torch.long, device=DEV)
            mmr.index_fill_(0, taken, float('-inf'))

        # optional: avoid duplicate WP ids
        if dedup_wpids is not None and selected_wp:
            dup_mask = torch.tensor(
                [int(w.item()) in selected_wp for w in dedup_wpids],
                dtype=torch.bool, device=DEV
            )
            mmr = torch.where(dup_mask, torch.tensor(float('-inf'), device=DEV), mmr)

        i = int(torch.argmax(mmr).item())
        if not math.isfinite(float(mmr[i].item())):
            break

        selected.append(i)
        if dedup_wpids is not None:
            selected_wp.add(int(dedup_wpids[i].item()))

        # grow Sel
        Sel = V[i:i+1] if Sel is None else torch.cat([Sel, V[i:i+1]], dim=0)

    return selected

import math
from typing import List, Optional
import torch

def mmr_select_unified(
    V: torch.Tensor,                 # [M, d] PRF token vectors (L2-normalised, same device)
    rel: torch.Tensor,               # [M]    relevance scores (same device as V)
    top_k: int,
    Q: torch.Tensor,                 # [n_q, d] original query token vectors (L2-normalised)
    lambda_uni: float = 0.3,         # 单一λ：同时控制对 S∪Q 的多样性惩罚
    dedup_wpids: Optional[torch.Tensor] = None,  # [M] CPU long; WordPiece ids for可选去重
) -> List[int]:
    """
    Unified MMR:
      mmr(i) = (1 - λ) * rel(i) - λ * max_{x in (S ∪ Q)} cos(v_i, x)

    其中 S 为已选扩展集合（动态增长），Q 为原始查询 token 集合（固定）。
    Q 的惩罚从一开始就生效；S 的惩罚在选择后逐步生效。
    """
    DEV = V.device
    M = int(V.size(0))
    k = min(top_k, M)

    Qd = Q.to(device=DEV, dtype=V.dtype)

    selected: List[int] = []
    Sel = None  # [t, d] 已选向量
    selected_wp = set()

    # 预计算 v_i 对查询 Q 的最大相似度： sim_q[i] = max_j <v_i, q_j>
    sim_q = (V @ Qd.T).max(dim=1).values if Qd.numel() > 0 else torch.zeros(M, device=DEV, dtype=V.dtype)

    for _ in range(k):
        # v_i 对已选扩展 S 的最大相似度（若 S 为空则为0）
        if Sel is None:
            sim_sel = torch.zeros(M, device=DEV, dtype=V.dtype)
        else:
            sim_sel = (V @ Sel.T).max(dim=1).values

        # 合并抑制源：对 S∪Q 的最大相似度
        sim_combined = torch.maximum(sim_sel, sim_q)

        # 统一λ的 MMR
        mmr = (1.0 - lambda_uni) * rel - lambda_uni * sim_combined

        # ban 已选
        if selected:
            taken = torch.as_tensor(selected, dtype=torch.long, device=DEV)
            mmr.index_fill_(0, taken, float('-inf'))

        # 可选：WP 去重
        if dedup_wpids is not None and selected_wp:
            dup_mask = torch.tensor(
                [int(w.item()) in selected_wp for w in dedup_wpids],
                dtype=torch.bool, device=DEV
            )
            mmr = torch.where(dup_mask, torch.tensor(float('-inf'), device=DEV), mmr)

        i = int(torch.argmax(mmr).item())
        if not math.isfinite(float(mmr[i].item())):
            break

        selected.append(i)
        if dedup_wpids is not None:
            selected_wp.add(int(dedup_wpids[i].item()))
        Sel = V[i:i+1] if Sel is None else torch.cat([Sel, V[i:i+1]], dim=0)

    return selected

