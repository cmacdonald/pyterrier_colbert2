from . import ColBERTModelOnlyFactory

import pandas as pd
import pyterrier as pt
import os
from pyterrier import tqdm
#from colbert.evaluation.load_model import load_model
#from .. import load_checkpoint
# monkeypatch to use our downloading version
#import colbert.evaluation.loaders
#colbert.evaluation.loaders.load_checkpoint = load_checkpoint
#colbert.evaluation.loaders.load_model.__globals__['load_checkpoint'] = load_checkpoint
from colbert.searcher import Searcher
from warnings import warn
import torch
from colbert.search.index_storage import StridedTensor #for plaid stage search
from colbert.modeling.colbert import colbert_score_reduce #for plaid stage search

class ColBERTv2Index(ColBERTModelOnlyFactory, pt.Artifact):

    ARTIFACT_TYPE = 'dense_index'
    ARTIFACT_FORMAT = 'colbert'
    ARTIFACT_PACKAGE_HINT = 'pyterrier_colbert2'

    def __init__(self, colbert, index_location, plaid_mode=False,
    ncells=None, centroid_score_threshold=None, ndocs=None, **kwargs):
        # TODO do we need the colbert checkpoint....; Searcher will load it too.

        # call both super-class constructors
        ColBERTModelOnlyFactory.__init__(self, colbert, **kwargs)
        pt.Artifact.__init__(self, index_location)
        self.plaid_mode = plaid_mode
        self.ncells = ncells
        self.centroid_score_threshold = centroid_score_threshold
        self.ndocs = ndocs
        dirs = os.path.split(index_location)
        self.searcher = Searcher(dirs[-1], index_root=os.path.join(*dirs[0:-1]))
        if self.plaid_mode == True:
            self.searcher.configure(ncells=self.ncells,
                                centroid_score_threshold=self.centroid_score_threshold,
                                ndocs=self.ndocs)
            
        # Load the docno mappings from the permanent file
        docno_file = os.path.join(index_location, "docnos.npids")
        from npids import Lookup
        self.docnos = Lookup(docno_file)


    """
    End-to-end retrieval wrapper using dense_search. 
    in particular, searcher.dense_search maybe with different configs for colbertv2 and plaid.
    """
    def end_to_end(self, k=1000) -> pt.Transformer: 
        def _search(df_query):
            if len(df_query) == 0:
                return pd.DataFrame(columns=["qid", "query", "docno", "score", "rank"])
            
            # TODO can we make df_queries into a colbert.Queries object to allow parallelisation?
            assert len(df_query) == 1
            # encode Q
            Q = self.searcher.encode([df_query.iloc[0]["query"]])

            # call colbert.Searcher or plaid if plaid_mode is True
            docids, ranks, scores = self.searcher.dense_search(Q, k=k)
            docnos = self.docnos.fwd[docids]

            # interestingly, ranks can be longers, i.e. we can retrieve 
            # less documents than k, but ranks will still be of length k 
            if len(ranks) > len(scores):
                ranks = ranks[0:len(scores)]

            return pd.DataFrame({
                "qid": [df_query.iloc[0]["qid"]] * len(docnos),
                "query": [df_query.iloc[0]["query"]] * len(docnos),
                "docno": docnos,
                "score": scores,
                "rank": ranks
            })
        return pt.apply.by_query(_search, add_ranks=False)

    """
    More specifically, a PLAID retrieval wrapper using  candidate generation
     and centroid interaction and pruning stages.
    Requires an index built with ivf.pid.pt (optimised inverted file).
    """

    def plaid_candidate_generation(self) -> pt.Transformer:
        """
        Stage 1: Generate candidates.  For each query, return a single row with
        the encoded query, the list of pids and the centroid_scores.
        Output columns: qid, query, Q, pids, centroid_scores
        """
        assert self.plaid_mode == True
        def _generate(df_query):
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
        return pt.apply.by_query(_generate)

    def plaid_centroid_interaction(self) -> pt.Transformer:
        """
        Stage 2: Compute approximate scores for each candidate.
        Takes the output of candidate_generation and expands it into one row per candidate.
        Output columns: qid, query, pid, docno, approx_score
        """
        assert self.plaid_mode == True
        def _interact(df):
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
        return pt.apply.generic(_interact)

    def plaid_centroid_pruning(self) -> pt.Transformer:
        """
        Stage 3: Prune the approximate scores down to ndocs per query.
        Input columns: qid, query, pid, docno, approx_score
        Output columns: qid, query, pid, docno
        """
        assert self.plaid_mode == True

        ndocs = self.searcher.config.ndocs
        def _prune(df):
            pruned_rows = []
            for qid, group in df.groupby("qid"):
                pruned = group.sort_values("score", ascending=False).head(ndocs) # scores are  the approx_scores after centroid interaction
                pruned_rows.append(pruned[["qid", "query", "pid", "docno"]])
            return pd.concat(pruned_rows).reset_index(drop=True)
        return pt.apply.generic(_prune)

    def plaid_final_scoring(self, k=1000) -> pt.Transformer:
        """
        Stage 4: Compute full ColBERT scores on the pruned set.
        Input columns: qid, query, pid, docno
        Output columns: qid, docno, score, rank
        """
        assert self.plaid_mode == True
        def _score(df):
            results = []
            for qid, group in df.groupby("qid"):
                query = group["query"].iloc[0]
                Q = self.searcher.encode([query])
                pids = torch.tensor(group["pid"].tolist())
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
        return pt.apply.by_query(_score)
