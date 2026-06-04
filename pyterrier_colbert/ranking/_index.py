import warnings # to remove the autocast warning

from . import ColBERTModelOnlyFactory

import pandas as pd
import pyterrier as pt
import os
from typing import Optional
import json
from pyterrier import tqdm
from colbert.searcher import Searcher
from warnings import warn
import torch
from colbert.search.index_storage import StridedTensor #for plaid stage search
from colbert.modeling.colbert import colbert_score_reduce #for plaid stage search

def suppress_amp_autocast_warning(func):
    import warnings
    from functools import wraps
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"`torch\.cuda\.amp\.autocast\(args\.\.\.\)` is deprecated",
                category=FutureWarning,
            )
            return func(*args, **kwargs)

    return wrapper


class ColBERTv2Index(ColBERTModelOnlyFactory, pt.Artifact):

    ARTIFACT_TYPE = 'dense_index'
    ARTIFACT_FORMAT = 'colbert'
    ARTIFACT_PACKAGE_HINT = 'pyterrier_colbert2'

    def __init__(self, index_location : str, plaid_mode=False, colbert : Optional[str] = None,
        ncells=None, centroid_score_threshold=None, ndocs=None, **kwargs):
        
        with open(os.path.join(index_location,'pt_meta.json'), 'rt') as f_meta:
            meta = json.load(f_meta)
            assert meta.get('type') == 'dense_index' and meta['format'] == 'colbert'
            self._meta = meta
            if colbert is None:
                colbert = self._meta.get('model_checkpoint', colbert)

        # call both super-class constructors
        ColBERTModelOnlyFactory.__init__(self, colbert, **kwargs)
        pt.Artifact.__init__(self, index_location)
        self.plaid_mode = plaid_mode
        self.ncells = ncells
        self.centroid_score_threshold = centroid_score_threshold
        self.ndocs = ndocs
        dirs = os.path.split(index_location)
        self.searcher = Searcher(dirs[-1], index_root=os.path.join(*dirs[0:-1]))
        if self.plaid_mode:
            self.searcher.configure(ncells=self.ncells,
                                centroid_score_threshold=self.centroid_score_threshold,
                                ndocs=self.ndocs)
            
        # Load the docno mappings from the permanent file
        docno_file = os.path.join(index_location, "docnos.npids")
        from npids import Lookup
        self.docnos = Lookup(docno_file)

    def __len__(self):
        return len(self.docnos)
    
    #from . import prf
    #plaid_prf_end_to_end = prf.plaid_prf_end_to_end

    """
    End-to-end retrieval wrapper using dense_search. 
    in particular, searcher.dense_search maybe with different configs for colbertv2 and plaid.
    """
    def end_to_end(self, k=1000, decompose=False, query_encoded=False) -> pt.Transformer: 

        @suppress_amp_autocast_warning
        def _search_query_encoded(df_query):
            pt.validate.query_frame(df_query, extra_columns=["query_vec"])
            if len(df_query) == 0:
                return pd.DataFrame(columns=["qid", "query", "docno", "score", "rank"])
            
            assert len(df_query) == 1
            # fetch encoded query vector from the dataframe, and ensure it's a torch tensor on the correct device
            val = df_query.iloc[0]["query_vec"]
            Q = val.detach().clone() if isinstance(val, torch.Tensor) else torch.tensor(val)
            if torch.cuda.is_available():
                Q = Q.cuda()

            # call colbert.Searcher or plaid if plaid_mode is True
            docids, ranks, scores = self.searcher.dense_search(Q, k=k)
            docnos = self.docnos.fwd[docids]

            # ignore the ranks returned by the searcher and re-assign them based on the sorted order of scores, 
            # to ensure consistency between colbertv2 and plaid modes. This is because in plaid mode, the searcher 
            # may return fewer than k results due to pruning; also ensures they start at pt.model.FIRST_RANK
            ranks = ranks[0:len(scores)]
            ranks = [pt.model.FIRST_RANK + i for i in range(len(ranks))]
            return pd.DataFrame({
                "qid": [df_query.iloc[0]["qid"]] * len(docnos),
                "query": [df_query.iloc[0]["query"]] * len(docnos),
                "docno": docnos,
                "score": scores,
                "rank": ranks
            })

        @suppress_amp_autocast_warning
        def _search(df_query):
            pt.validate.query_frame(df_query, extra_columns=["query"])
            if len(df_query) == 0:
                return pd.DataFrame(columns=["qid", "query", "docno", "score", "rank"])
            
            # TODO can we make df_queries into a colbert.Queries object to allow parallelisation?
            assert len(df_query) == 1
            # encode Q
            Q = self.searcher.encode([df_query.iloc[0]["query"]])

            # call colbert.Searcher or plaid if plaid_mode is True
            docids, ranks, scores = self.searcher.dense_search(Q, k=k)
            docnos = self.docnos.fwd[docids]

            # ignore the ranks returned by the searcher and re-assign them based on the sorted order of scores, 
            # to ensure consistency between colbertv2 and plaid modes. This is because in plaid mode, the searcher 
            # may return fewer than k results due to pruning; also ensures they start at pt.model.FIRST_RANK
            ranks = ranks[0:len(scores)]
            ranks = [pt.model.FIRST_RANK + i for i in range(len(ranks))]
            return pd.DataFrame({
                "qid": [df_query.iloc[0]["qid"]] * len(docnos),
                "query": [df_query.iloc[0]["query"]] * len(docnos),
                "docno": docnos,
                "score": scores,
                "rank": ranks
            })

        if decompose:
            assert self.plaid_mode == True, "Decomposed search is only supported in PLAID mode"
            assert not query_encoded, "Decomposed search is not compatible with pre-encoded queries"
            return self.plaid_candidate_generation() >> self.plaid_centroid_interaction() >> self.plaid_centroid_pruning() >> self.plaid_final_scoring(k=k)
        return pt.apply.by_query(_search_query_encoded if query_encoded else _search, add_ranks=False, label="PLAID" if self.plaid_mode else "ColBERTv2")

    """
    More specifically, a PLAID retrieval wrapper using candidate generation
    and centroid interaction and pruning stages.
    Requires an index built with ivf.pid.pt (optimised inverted file).
    """
    def plaid_candidate_generation(self) -> pt.Transformer:
        """
        Stage 1: Generate candidates.  For each query, return a single row with
        the encoded query, the list of pids and the centroid_scores.
        Output columns: qid, query, Q, pids, centroid_scores
        """
        assert self.plaid_mode
        def _generate(df_query):
            pt.validate.query_frame(df_query, extra_columns=["query"])
            if len(df_query) == 0:
                return pd.DataFrame(columns=["qid", "query", "Q_embs", "pids", "score"])
            
            assert len(df_query) == 1
            row = df_query.iloc[0]
            qid, query = row.qid, row.query
            Q = self.searcher.encode([query])
            pids, centroid_scores = self.searcher.ranker.generate_candidates(
                self.searcher.config, Q
            )
            return pd.DataFrame([{
                "qid": qid,
                "query": query,
                "Q_embs": Q, # Q_embs is the query embeddings
                "pids": pids,
                "score": centroid_scores # centroid_scores before centroid interaction
            }])
        return pt.apply.by_query(_generate, label="plaid_candidate_generation")

    def plaid_centroid_interaction(self) -> pt.Transformer:
        """
        Stage 2: Compute approximate scores for each candidate.
        Takes the output of candidate_generation and expands it into one row per candidate.
        Output columns: qid, query, pid, docno, approx_score
        """
        assert self.plaid_mode
        def _interact(df):
            pt.validate.query_frame(df, extra_columns=["query", "pids", "score"])
            if len(df) == 0:
                return pd.DataFrame(columns=["qid", "query", "pid", "docno", "score"])
            rows = []
            for _, r in df.iterrows():
                qid, query = r.qid, r.query
                pids = r.pids
                centroid_scores = r.score #r.score are centroid_scores before centroid interaction
                # lookup token codes for each candidate passage
                codes_packed, codes_lengths = self.searcher.ranker.embeddings_strided.lookup_codes(pids)
                approx_scores_tok = centroid_scores[codes_packed.long()]
                approx_strided = StridedTensor(approx_scores_tok, codes_lengths, use_gpu=False)
                approx_padded, approx_mask = approx_strided.as_padded_tensor()
                approx_scores = colbert_score_reduce(approx_padded, approx_mask, self.searcher.config)
                for pid, approx in zip(pids.tolist(), approx_scores):
                    docno = self.docnos.fwd[pid]
                    # docno = self.docno_mapping.get(pid, "unknown_docno")
                    rows.append({
                        "qid": qid,
                        "query": query,
                        "pid": pid,
                        "docno": docno,
                        "score": approx.item() # approx_score after centroid interaction
                    })
            return pd.DataFrame(rows)
        return pt.apply.generic(_interact, label="plaid_centroid_interaction")

    def plaid_centroid_pruning(self) -> pt.Transformer:
        """
        Stage 3: Prune the approximate scores down to ndocs per query.
        Input columns: qid, query, pid, docno, approx_score
        Output columns: qid, query, pid, docno
        """
        assert self.plaid_mode

        ndocs = self.searcher.config.ndocs
        def _prune(df):
            pt.validate.result_frame(df, extra_columns=["query", "score"])
            if len(df) == 0:
                return pd.DataFrame(columns=["qid", "query", "pid", "docno", "score"])
            pruned_rows = []
            for qid, group in df.groupby("qid"):
                pruned = group.sort_values("score", ascending=False).head(ndocs) # scores are  the approx_scores after centroid interaction
                pruned_rows.append(pruned[["qid", "query", "pid", "docno"]])
            return pd.concat(pruned_rows).reset_index(drop=True)
        return pt.apply.generic(_prune, label="plaid_centroid_pruning")

    def plaid_final_scoring(self, k=1000) -> pt.Transformer:
        """
        Stage 4: Compute full ColBERT scores on the pruned set.
        Input columns: qid, query, pid, docno
        Output columns: qid, docno, score, rank
        """
        assert self.plaid_mode
        def _score(df):
            pt.validate.result_frame(df, extra_columns=["query", "pid"])
            if len(df) == 0:
                return pd.DataFrame(columns=["qid", "query", "docno", "score", "rank"])
            results = []
            for qid, group in df.groupby("qid"):
                query = group["query"].iloc[0]
                Q = self.searcher.encode([query])
                pids = torch.tensor(group["pid"].tolist()).int()
                # Pass centroid_scores=None to disable further pruning and get final scores
                scores, pids_scored = self.searcher.ranker.score_pids(
                    self.searcher.config, Q, pids, centroid_scores=None
                )
                # Keep top k results
                topk = min(k, len(scores))
                top_indices = scores.argsort(descending=True)[:topk]
                for rank, idx in enumerate(top_indices):
                    pid = pids_scored[idx].item()
                    docno = self.docnos.fwd[pid]
                    # docno = self.docno_mapping.get(pid, "unknown_docno")
                    results.append({
                        "qid": qid,
                        "docno": docno,
                        "score": scores[idx].item(),
                        "rank": rank + 1
                    })
            return pd.DataFrame(results, columns=["qid", "docno", "score", "rank"])
        return pt.apply.by_query(_score, label="plaid_final_scoring")

