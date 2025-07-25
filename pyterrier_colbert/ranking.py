
import os
import json
import torch
import pandas as pd
import pyterrier as pt
assert pt.started(), "please run pt.init() before importing pyt_colbert"

from pyterrier import tqdm
from typing import Union, Tuple
from colbert.evaluation.load_model import load_model
from . import load_checkpoint
# monkeypatch to use our downloading version
import colbert.evaluation.loaders
colbert.evaluation.loaders.load_checkpoint = load_checkpoint
colbert.evaluation.loaders.load_model.__globals__['load_checkpoint'] = load_checkpoint
from colbert.modeling.checkpoint import Checkpoint  # modified from colbert.inference import Checkpoint
from colbert.modeling.colbert import ColBERT  # add a new method to use BaseColBERT
from colbert.modeling.colbert import colbert_score #add a new score method
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer # to build query/doc from text manually
from colbert.infra import Run, RunConfig, ColBERTConfig  # add ColBERTConfig
from colbert.indexing.loaders import get_parts, load_doclens
from colbert.searcher import Searcher
import colbert.modeling.colbert
import numpy as np
import pickle
from warnings import warn



class Object(object):
    pass


from typing import List     


class ColBERTModelOnlyFactory():
    
    def __init__(self, colbert_model: Union[str, Tuple[ColBERT, dict]], gpu=True, mask_punctuation=False, dim=128):
        args = Object()
        args.query_maxlen = 32
        args.doc_maxlen = 180
        args.dim = dim  
        args.bsize = 128
        args.similarity = 'cosine'        
        args.amp = True
        args.nprobe = 10
        args.part_range = None
        args.mask_punctuation = mask_punctuation
        args.gpu = gpu  # Add gpu properties
        self.gpu = gpu
        if not gpu:
            warn("Gpu disabled, YMMV")
            import colbert.parameters
            colbert.parameters.DEVICE = torch.device("cpu")

        if isinstance(colbert_model, str):
            args.checkpoint = colbert_model
            colbert_config = ColBERTConfig.load_from_checkpoint(colbert_model)
            args.colbert = Checkpoint(name=args.checkpoint, colbert_config=colbert_config)
        else:
            assert isinstance(colbert_model, tuple)
            args.colbert, args.checkpoint = colbert_model
            assert isinstance(args.colbert, ColBERT)
            assert isinstance(args.checkpoint, dict)
        
        args.inference = args.colbert
        self.args = args

        
    def query_encoder(self, detach=True) -> pt.Transformer:
        def _encode_query(row):
            with torch.no_grad():
                Q = self.args.inference.queryFromText([row.query], bsize=512)
                if detach:
                    Q = Q.cpu()
                return pd.Series([Q[0]])
            
        def row_apply(df):
            if "docno" in df.columns or "docid" in df.columns:
                warn("You are query encoding an R dataframe, the query will be encoded for each row")
            df["query_embs"] = df.apply(_encode_query, axis=1)
            return df
        
        return pt.apply.generic(row_apply)
        
    def text_encoder(self, detach=True, batch_size=8) -> pt.Transformer:
        """
        Returns a transformer that can encode the text using ColBERT's model.
        input: qid, text
        output: qid, text, doc_embs, doc_toks,
        """
        def chunker(seq, size):
            for pos in range(0, len(seq), size):
                yield seq.iloc[pos:pos + size]
        def df_apply(df):
            with torch.no_grad():
                rtr_embs = []
                rtr_toks = []
                for chunk in chunker(df, batch_size):
                    embsD, idsD = self.args.inference.docFromText(chunk.text.tolist(), with_ids=True)
                    if detach:
                        embsD = embsD.cpu()
                    rtr_embs.extend([embsD[i, : ,: ] for i in range(embsD.shape[0])])
                    rtr_toks.extend(idsD)
            df["doc_embs"] = pd.Series(rtr_embs)
            df["doc_toks"] = pd.Series(rtr_toks)
            return df
        return pt.apply.generic(df_apply)

    def explain_text(self, query : str, document : str):
        """
        Provides a diagram explaining the interaction between a query and the text of a document
        """       
        # Use DocTokenizer to process documents and return idsD
        doc_tokenizer = DocTokenizer(config=self.args.colbert.colbert_config)
        idsD, attention_mask = doc_tokenizer.tensorize([document])
        
        # Generate document embedding and mask using inference.doc
        embsD, D_mask = self.args.inference.doc(idsD, attention_mask, keep_dims='return_mask')
        embsD = embsD.cpu()

        return self._explain(query, embsD, idsD)
    
    def _explain(self, query, embsD, idsD):
        inference = self.args.inference
        query_tokenizer = QueryTokenizer(config=self.args.colbert.colbert_config)
        idsQ, attention_mask = query_tokenizer.tensorize(
            [query], context=None, full_length_search=False
        )
        embsQ = inference.query(idsQ, attention_mask).cpu()
        # Ensure that the data types are consistent, converting both embsQ and embsD to float32
        embsQ = embsQ.to(dtype=torch.float32)
        embsD = embsD.to(dtype=torch.float32)
        interaction = (embsQ[0] @ embsD[0].T).cpu().numpy().T
        
        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec

        tokenmap = {"[unused1]" : "[D]", "[unused0]" : "[Q]"}

        fig = plt.figure(figsize=(4, 12)) 
        gs = GridSpec(2, 1, height_ratios=[1, 20]) 

        ax1=fig.add_subplot(gs[0])
        ax2=fig.add_subplot(gs[1])
        
        ax2.matshow(interaction, cmap=plt.cm.Blues)
        qtokens = self.args.inference.query_tokenizer.tok.convert_ids_to_tokens(idsQ[0])
        dtokens = self.args.inference.query_tokenizer.tok.convert_ids_to_tokens(idsD[0])
        qtokens = [tokenmap[t] if t in tokenmap else t for t in qtokens]
        dtokens = [tokenmap[t] if t in tokenmap else t for t in dtokens]

        ax2.set_xticks(range(32), minor=False)
        ax2.set_xticklabels(qtokens, rotation=90)
        ax2.set_yticks(range(len(idsD[0])))
        ax2.set_yticklabels(dtokens)
        ax2.set_anchor("N")

        contributions=[]
        for i in range(32):
            maxpos = np.argmax(interaction[:,i])
            plt.text(i-0.25, maxpos+0.1, "X", fontsize=5)
            contributions.append(interaction[maxpos,i])

        from sklearn.preprocessing import minmax_scale
        ax1.bar([0.5 + i for i in range(0,32)], contributions, color=plt.cm.Blues(minmax_scale(contributions, feature_range=(0.4, 1))))
        ax1.set_xlim([0,32])
        ax1.set_xticklabels([])
        fig.tight_layout()
        #fig.subplots_adjust(hspace=-0.37)
        return fig
    
    def text_scorer(self, query_encoded=False, doc_attr="text", verbose=False) -> pt.Transformer:
        def slow_rerank(args, query, pids, passages):
            colbert = args.colbert
            inference = args.inference

            Q = inference.queryFromText([query]).cpu()

            doc_tokenizer = DocTokenizer(config=colbert.colbert_config)
            input_ids, attention_mask = doc_tokenizer.tensorize(passages)
            D_padded, D_mask = inference.doc(input_ids, attention_mask, keep_dims='return_mask')

            # v2 colbert_score
            scores = colbert_score(Q, D_padded, D_mask, config=colbert.colbert_config).cpu()

            scores = scores.sort(descending=True)
            ranked = scores.indices.tolist()

            ranked_scores = scores.values.tolist()
            ranked_pids = [pids[position] for position in ranked]
            ranked_passages = [passages[position] for position in ranked]

            return list(zip(ranked_scores, ranked_pids, ranked_passages))

        
        def slow_rerank_with_qembs(args, qembs, pids, passages, gpu=True):
            inference = args.inference 
            print(f"Using inference object: {inference}")

            Q = torch.unsqueeze(qembs, 0)
            if gpu:
                Q = Q.cuda()
            Q = Q.half()  # Converts the query embed to the float16 type

            D_tuple = inference.docFromText(passages, bsize=args.bsize, keep_dims=True, to_cpu=not gpu)

            # Unlock the returned tuple to get the actual document embed
            D = D_tuple[0] 
            if gpu:
                D = D.cuda()

            try:
                scores = (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)
                print(f"Scores: {scores}")
            except Exception as e:
                print(f"Error calculating scores: {e}")
                return []

            scores = scores.sort(descending=True)
            ranked = scores.indices.tolist()
            ranked_scores = scores.values.tolist()
            ranked_pids = [pids[position] for position in ranked]
            ranked_passages = [passages[position] for position in ranked]
            return list(zip(ranked_scores, ranked_pids, ranked_passages))

        def _text_scorer(queries_and_docs):
            groupby = queries_and_docs.groupby("qid")
            rtr = []
            with torch.no_grad():
                for qid, group in tqdm(groupby, total=len(groupby), unit="q") if verbose else groupby:
                    query = group["query"].values[0]
                    ranking = slow_rerank(self.args, query, group["docno"].values, group[doc_attr].values.tolist())
                    for rank, (score, pid, passage) in enumerate(ranking):
                        rtr.append([qid, query, pid, score, rank])
            return pd.DataFrame(rtr, columns=["qid", "query", "docno", "score", "rank"])

        def _text_scorer_qembs(queries_and_docs):
            groupby = queries_and_docs.groupby("qid")
            rtr = []
            with torch.no_grad():
                for qid, group in tqdm(groupby, total=len(groupby), unit="q") if verbose else groupby:
                    qembs = group["query_embs"].values[0]
                    query = group["query"].values[0]
                    ranking = slow_rerank_with_qembs(self.args, qembs, group["docno"].values, group[doc_attr].values.tolist(), gpu=self.gpu)
                    for rank, (score, pid, passage) in enumerate(ranking):
                        rtr.append([qid, query, pid, score, rank])
            return pd.DataFrame(rtr, columns=["qid", "query", "docno", "score", "rank"])

        return pt.apply.generic(_text_scorer_qembs if query_encoded else _text_scorer)


    def scorer(factory, add_contributions=False, add_exact_match_contribution=False, verbose=False, gpu=True) -> pt.Transformer:
        """
        Calculates the ColBERT max_sim operator using previous encodings of queries and documents
        input: qid, query_embs, [query_weights], docno, doc_embs
        output: ditto + score, [+ contributions]
        """
        import torch
        import pyterrier as pt
        assert pt.started(), 'PyTerrier must be started'
        cuda0 = torch.device('cuda') if gpu else torch.device('cpu')

        def _build_interaction(row, D):
            doc_embs = row.doc_embs
            doc_len = doc_embs.shape[0]
            D[row.row_index, 0:doc_len, :] = doc_embs
            
        def _build_toks(row, idsD):
            doc_toks = row.doc_toks
            doc_len = doc_toks.shape[0]
            idsD[row.row_index, 0:doc_len] = doc_toks
        
        def _score_query(df):
            with torch.no_grad():
                weightsQ = None
                Q = torch.cat([df.iloc[0].query_embs])
                if "query_weights" in df.columns:
                    weightsQ = df.iloc[0].query_weights
                else:
                    weightsQ = torch.ones(Q.shape[0])
                if gpu:
                    Q = Q.cuda()
                    weightsQ = weightsQ.cuda()        
                D = torch.zeros(len(df), factory.args.doc_maxlen, factory.args.dim, device=cuda0)
                df['row_index'] = range(len(df))
                if verbose:
                    pt.tqdm.pandas(desc='scorer')
                    df.progress_apply(lambda row: _build_interaction(row, D), axis=1)
                else:
                    df.apply(lambda row: _build_interaction(row, D), axis=1)
                maxscoreQ = (Q @ D.permute(0, 2, 1)).max(2).values
                scores = (weightsQ*maxscoreQ).sum(1).cpu()
                df["score"] = scores.tolist()
                if add_contributions:
                    contributions = (Q @ D.permute(0, 2, 1)).max(1).values.cpu()
                    df["contributions"] = contributions.tolist()
                if add_exact_match_contribution:
                    idsQ = torch.cat([df.iloc[0].query_toks]).unsqueeze(0)
                    idsD = torch.zeros(len(df), factory.args.doc_maxlen, dtype=idsQ.dtype)

                    df.apply(lambda row: _build_toks(row, idsD), axis=1)

                    # which places in the query are actual tokens, not specials such as MASKs
                    token_match = (idsQ != 101) & (idsQ != 102) & (idsQ != 103) & (idsQ != 1) & (idsQ != 2)

                    # which places in the interaction have exact matches (not [CLS])
                    exact_match = (idsQ.unsqueeze(1) == idsD.unsqueeze(2)) & (idsQ != 101)
                    
                    # perform the interaction
                    interaction = (Q @ D.permute(0, 2, 1)).cpu()

                    weightsQ = weightsQ.unsqueeze(0).cpu()

                    weighted_maxsim = weightsQ*interaction.max(2).values

                    # mask out query embeddings that arent tokens 
                    weighted_maxsim[:, ~token_match[0,:]] = 0
                    
                    # get the sum
                    denominator = weighted_maxsim.sum(1)

                    # zero out entries that arent exact matches
                    interaction[~ exact_match.permute(0, 2, 1)] = 0

                    weighted_maxsim = weightsQ*interaction.max(2).values
                    # mask out query embeddings that arent tokens 
                    weighted_maxsim[:, ~token_match[0,:]] = 0

                    # get the sum
                    numerator = weighted_maxsim.sum(1)
                    #df["exact_count"] = exact_match
                    df["exact_numer"] = numerator.tolist()
                    df["exact_denom"] = denominator.tolist()
                    df["exact_pct"] = (numerator/denominator).tolist()
            return df
            
        return pt.apply.by_query(_score_query, add_ranks=True)
    

class ColBERTv2Index(ColBERTModelOnlyFactory):

    def __init__(self, colbert, index_location, **kwargs):
        super().__init__(colbert, **kwargs)
        self.searcher = Searcher(index_location)
        self.docno_mapping = {}

        # Load the docno mappings from the permanent file
        docno_file = os.path.join(index_location, "docnos.npids")
        from npids import Lookup
        self.docnos = Lookup(docno_file)


    def end_to_end(self, k=100) -> pt.Transformer:
        def _search(df_query):
            if len(df_query) == 0:
                return pd.DataFrame(columns=["qid", "docno", "score", "rank"])
            
            # TODO can we make df_queries into a colbert.Queries object
            assert len(df_query) == 1
            # encode Q
            Q = self.searcher.encode([df_query.iloc[0]["query"]])
      
            # call colbert.Searcher
            docids, ranks, scores = self.searcher.dense_search(Q, k=k)
            docnos = self.docnos.fwd[docids]
            return pd.DataFrame({
                "qid": df_query.iloc[0]["qid"],
                "docno": docnos,
                "score": scores,
                "rank": ranks
            })
        return pt.apply.by_query(_search)
