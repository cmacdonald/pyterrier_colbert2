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
    def colbertv2_end_to_end(self, k=1000) -> pt.Transformer: 
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
    def plaid_end_to_end(self, k=1000) -> pt.Transformer:
        assert self.plaid_mode == True, "plaid_end_to_end should only be used in PLAID mode"
        def _search(df_query):
            assert len(df_query) == 1
            query_text = df_query.iloc[0]["query"]
            qid = df_query.iloc[0]["qid"]

            # Encode the query
            Q = self.searcher.encode([query_text])
            

            # PLAID candidate generation: returns pids and centroid_scores
            # https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/search/candidate_generation.py#L45
            pids, centroid_scores = self.searcher.ranker.generate_candidates(
                self.searcher.config, Q
            )
            # PLAID centroid interaction, pruning and final scoring
            # score_pids returns (scores, pids)
            # https://github.com/stanford-futuredata/ColBERT/blob/main/colbert/search/index_storage.py#L111
            scores, pids = self.searcher.ranker.score_pids(
                self.searcher.config, Q, pids, centroid_scores
            )
                   
            # Extract the top-k results
            topk = min(k, len(pids))
            top_indices = scores.argsort(descending=True)[:topk]
            results = []
            for rank, idx in enumerate(top_indices):
                pid = pids[idx].item()
                docno = self.docnos.fwd[pid]
                # docno = self.docno_mapping.get(pid, "unknown_docno")
                score = scores[idx].item()
                results.append([qid, docno, score, rank + 1])

            return pd.DataFrame(results, columns=["qid", "docno", "score", "rank"])

        return pt.apply.by_query(_search)
    
    

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
            if 'query_vec' in df_query.columns:
                print("generating candaite based on expanded query vecs")
                query_vec = torch.from_numpy(row.query_vec)
                if torch.cuda.is_available():
                    query_vec = query_vec.to('cuda', non_blocking=True).to(dtype=torch.float16)
                pids, centroid_scores = self.searcher.ranker.generate_candidates(self.searcher.config, query_vec )
                return pd.DataFrame([{
                "qid": qid,
                "query": query,
                "Q_embs": query_vec, # Q_embs is the query embeddings
                # "Exp_embs": query_vec # Exp_embs is the expanded query embeddings
                "pids": pids,
                "score": centroid_scores # centroid_scores before centroid interaction           
                }])
            else:
                pids, centroid_scores = self.searcher.ranker.generate_candidates(
                    self.searcher.config, Q )
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
                # packed_codes: 1D LongTensor，列出该 passage 内所有 token 的 code ID（即 embedding 索引）
                # lengths:      1D LongTensor，告诉你每个 passage（这里只有一个）有多少 tokens
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
                        "Q_embs": r.Q_embs, # Q_embs is the query w/o expansion embeddings
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
                pruned_rows.append(pruned[["qid", "query","Q_embs", "pid", "docno"]])
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
                pids = torch.tensor(group["pid"].tolist())
                if 'Q_embs' in df.columns:
                    Q_embs = group["Q_embs"].iloc[0]
                    scores, pids_scored = factory.searcher.ranker.score_pids(
                        factory.searcher.config, Q_embs, pids, centroid_scores=None
                        )
            
                else:
                    Q = self.searcher.encode([query])
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
                        "query": query,
                        "docno": docno,
                        "score": scores[idx].item(),
                        "rank": rank + 1
                    })
            return pd.DataFrame(results, columns=["qid","query", "docno", "score", "rank"])
        return pt.apply.by_query(_score)
    
    
    def get_token_ids_for_passage(self, pid):
        """
        Given a passage ID, retrieve its token IDs from the index.
        Returns:
            packed_eids: 1D LongTensor of token IDs for the passage
            lengths: 1D LongTensor with the length of the passage (number of tokens)
        """
        codes_packed, codes_lengths = self.searcher.ranker.embeddings_strided.lookup_codes(torch.tensor([pid]))
        return codes_packed, codes_lengths
    
    
    
    
    def lookup_and_decompress(self, eids):
        
        from colbert.indexing.codecs import residual_embeddings as residual_embeddings
        """
        emb_ivf : ResidualEmbeddingsStrided (searcher.ranker.embeddings_strided)
        eids    : list[int] or 1D LongTensor of global embedding indexes
        return  : [len(eids), dim] float tensor of decompressed token embeddings
        """
        emb_ivf = self.searcher.ranker.embeddings_strided
        
        # Ensure a LongTensor of IDs
        if not isinstance(eids, torch.Tensor):
            eids = torch.tensor(eids, dtype=torch.long)
        # device = 'cuda' if getattr(emb_ivf, 'use_gpu', False) else 'cpu'
        # eids = eids.to(device)

        # Grab the integer codes for these embeddings (NOT from codes_strided)
        # emb_ivf.codes : 1D tensor [total_embeddings] of residual code IDs
        # emb_ivf.residuals : 1D/packed residual storage aligned with `codes`
        codes_subset     = emb_ivf.codes[eids]
        residuals_subset = emb_ivf.residuals[eids]

        # Use the same path as lookup_pids(): codec.decompress(ResidualEmbeddings(...))
        vecs = emb_ivf.codec.decompress(
            residual_embeddings.ResidualEmbeddings(codes_subset, residuals_subset)
        )
        return vecs
     
    def query_expansion(self,idf_map, top_psg=3, top_exp = 10, beta=0.5) -> pt.Transformer:
        """
        Query expansion using top_psg passages from ColBERTv2 retrieval.
        Expansion terms are selected based on tf-idf scores.
        The final expanded query vector is a weighted combination of the original query and expansion terms.
        Input columns: qid, query
        Output columns: qid, query, query_vec
        """ 
        assert self.plaid_mode == True
        def _expand(df_query):
            assert len(df_query) == 1
            row = df_query.iloc[0]
            qid, query = row.qid, row.query
            Q = self.searcher.encode([query])
            # Retrieve top_psg passages using dense search
            pids, centroid_scores = self.searcher.ranker.generate_candidates( self.searcher.config, Q)
            pids = pids[:top_psg]
            token_ids = []
            for pid in pids:
                packed_eids, lengths = self.get_token_ids_for_passage(pid)
                packed_eids = packed_eids.cpu()  # save some GPU mem
                token_ids.extend(packed_eids.tolist())
            scored_tokens = [(eid, idf_map.get(eid, 0.0)) for eid in token_ids]
            scored_tokens.sort(key = lambda x: x[1], reverse = True)
            expansion_eids = [eid for eid, _ in scored_tokens[:top_exp]]
            expansion_embs = self.lookup_and_decompress(expansion_eids) 
            weights = torch.tensor([idf_map.get(eid, 0.0) for eid in expansion_eids],device = expansion_embs.device) # similar to colbert-prf, no normalisation is applied here
            expansion_embs = expansion_embs * weights.unsqueeze(1) * beta 
            Q_expanded = torch.cat([Q.squeeze(0), expansion_embs], dim=0).unsqueeze(0)
            Q_np = Q_expanded.half().cpu().numpy() # save some GPU memory
            return pd.DataFrame([{
                "qid":qid,
                "query_vec": Q_expanded,
                "query": query
            }])
        return pt.apply.by_query(_expand,add_ranks=False)
                
                
