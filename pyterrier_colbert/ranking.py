
import os
import json
import torch
import pandas as pd
import pyterrier as pt
assert pt.started(), "please run pt.init() before importing pyt_colbert"

from pyterrier import tqdm
from pyterrier.datasets import Dataset
from typing import Union, Tuple
from colbert.evaluation.load_model import load_model
from . import load_checkpoint
# monkeypatch to use our downloading version
import colbert.evaluation.loaders
colbert.evaluation.loaders.load_checkpoint = load_checkpoint
colbert.evaluation.loaders.load_model.__globals__['load_checkpoint'] = load_checkpoint
# from colbert.modeling.inference import ModelInference
from colbert.modeling.checkpoint import Checkpoint  # modified from colbert.inference.checkpoint import Checkpoint
from colbert.modeling.colbert import ColBERT  # add a new method to use BaseColBERT
# from colbert.evaluation.slow import slow_rerank
from colbert.modeling.colbert import colbert_score
from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.infra import Run, RunConfig, ColBERTConfig  # add ColBERTConfig
from colbert.indexing.loaders import get_parts, load_doclens
from colbert.searcher import Searcher
from colbert.data.queries import Queries
import colbert.modeling.colbert
from collections import defaultdict
import numpy as np
import pickle
from warnings import warn

class file_part_mmap:
    def __init__(self, file_path, file_doclens, dim):
        self.dim = dim
        
        self.doclens = file_doclens
        self.endpos = np.cumsum(self.doclens)
        self.startpos = self.endpos - self.doclens

        mmap_storage = torch.HalfStorage.from_file(file_path, False, sum(self.doclens) * self.dim)
        self.mmap = torch.HalfTensor(mmap_storage).view(sum(self.doclens), self.dim)
 
    def get_embedding(self, pid):
        startpos = self.startpos[pid]
        endpos = self.endpos[pid]
        return self.mmap[startpos:endpos,:]

class file_part_mem:
    def __init__(self, file_path, file_doclens, dim):
        self.dim = dim

        self.doclens = file_doclens
        self.endpos = np.cumsum(self.doclens)
        self.startpos = self.endpos - self.doclens

        self.mmap = torch.load(file_path)
        #print(self.mmap.shape)
 
    def get_embedding(self, pid):
        startpos = self.startpos[pid]
        endpos = self.endpos[pid]
        return self.mmap[startpos:endpos,:]


class Object(object):
    pass


from typing import List     


class re_ranker_mmap:
    def __init__(self, index_path, args, inference, verbose = False, memtype='mmap'):
        self.args = args
        # self.doc_maxlen = args.doc_maxlen

        self.doc_maxlen = args.colbert_config.doc_maxlen   #use Checkpoint

        assert self.doc_maxlen > 0
        self.inference = inference
        # self.dim = args.dim

        self.dim = args.colbert_config.dim  #use Checkpoint

        self.verbose = verbose
    
        # Every pt file gets its own list of doc lengths
        self.part_doclens = load_doclens(index_path, flatten=False)
        assert len(self.part_doclens) > 0, "Did not find any indices at %s" % index_path
        # Local mmapped tensors with local, single file accesses
        self.part_mmap : List[file_part_mmap] = re_ranker_mmap._load_parts(index_path, self.part_doclens, self.dim, memtype)
        
        # last pid (inclusive, e.g., the -1) in each pt file
        # the -1 is used in the np.searchsorted
        # so if each partition has 1000 docs, the array is [999, 1999, ...]
        # this helps us map from passage id to part (inclusive, explaning the -1)
        self.part_pid_end_offsets = np.cumsum([len(x) for x in self.part_doclens]) - 1

        self.segment_sizes = torch.LongTensor([0] + [x.mmap.shape[0] for x in self.part_mmap])
        self.segment_starts = torch.cumsum(self.segment_sizes, 0)
        
        # first pid (inclusive) in each pt file
        tmp = np.cumsum([len(x) for x in self.part_doclens])
        tmp[-1] = 0
        self.part_pid_begin_offsets = np.roll(tmp, 1)
        # [0, 1000, 2000, ...]
        self.part_pid_begin_offsets
    
    @staticmethod
    def _load_parts(index_path, part_doclens, dim, memtype="mmap"):
        # Every pt file is loaded and managed independently, with local pids
        _, all_parts_paths, _ = get_parts(index_path)
        
        if memtype == "mmap":
            all_parts_paths = [ file.replace(".pt", ".store") for file in all_parts_paths ]
            mmaps = [file_part_mmap(path, doclens, dim) for path, doclens in zip(all_parts_paths, part_doclens)]
        elif memtype == "mem":
            mmaps = [file_part_mem(path, doclens, dim) for path, doclens in tqdm(zip(all_parts_paths, part_doclens), total=len(all_parts_paths), desc="Loading index shards to memory", unit="shard")]
        else:
            assert False, "Unknown memtype %s" % memtype
        return mmaps

    def num_docs(self):
        """
        Return number of documents in the index
        """
        return sum([len(x) for x in self.part_doclens])

    def get_embedding(self, pid):
        # In which pt file we need to look the given pid
        part_id = np.searchsorted(self.part_pid_end_offsets, pid)
        # calculate the pid local to the correct pt file
        local_pid = pid - self.part_pid_begin_offsets[part_id]
        # identify the tensor we look for
        disk_tensor = self.part_mmap[part_id].get_embedding(local_pid)
        doclen = disk_tensor.shape[0]
         # only here is there a memory copy from the memory mapped file 
        target = torch.zeros(self.doc_maxlen, self.dim)
        target[:doclen, :] = disk_tensor
        return target
    
    def get_embedding_copy(self, pid, target, index):
        # In which pt file we need to look the given pid
        part_id = np.searchsorted(self.part_pid_end_offsets, pid)
        # calculate the pid local to the correct pt file
        local_pid = pid - self.part_pid_begin_offsets[part_id]
        # identify the tensor we look for
        disk_tensor = self.part_mmap[part_id].get_embedding(local_pid)
        doclen = disk_tensor.shape[0]
        # only here is there a memory copy from the memory mapped file 
        target[index, :doclen, :] = disk_tensor
        return target
    
    def our_rerank(self, query, pids, gpu=True):
        colbert = self.args.colbert
        inference = self.inference

        # Q = inference.queryFromText([query])
        Q = inference.queryFromText([query], to_cpu=not self.args.gpu)  # use Checkpoint
        if self.verbose:
            pid_iter = tqdm(pids, desc="lookups", unit="d")
        else:
            pid_iter = pids

        D_ = torch.zeros(len(pids), self.doc_maxlen, self.dim)
        for offset, pid in enumerate(pid_iter):
            self.get_embedding_copy(pid, D_, offset)

        if gpu:
            D_ = D_.cuda()

        scores = colbert.score(Q, D_).cpu()
        del(D_)
        return scores.tolist()

    def our_rerank_batched(self, query, pids, gpu=True, batch_size=1000):
        import more_itertools
        if len(pids) < batch_size:
            return self.our_rerank(query, pids, gpu=gpu)
        allscores=[]
        for group in more_itertools.chunked(pids, batch_size):
            batch_scores = self.our_rerank(query, group, gpu)
            allscores.extend(batch_scores)
        return allscores
        
        
    def our_rerank_with_embeddings(self, qembs, pids, weightsQ=None, gpu=True):
        """
        input: qid,query, docid, query_tokens, query_embeddings, query_weights 
        
        output: qid, query, docid, score
        """
        colbert = self.args.colbert
        inference = self.inference
        # default is uniform weight for all query embeddings
        if weightsQ is None:
            weightsQ = torch.ones(len(qembs))
        # make to 3d tensor
        Q = torch.unsqueeze(qembs, 0)
        if gpu:
            Q = Q.cuda()
        
        if self.verbose:
            pid_iter = tqdm(pids, desc="lookups", unit="d")
        else:
            pid_iter = pids

        D_ = torch.zeros(len(pids), self.doc_maxlen, self.dim)
        for offset, pid in enumerate(pid_iter):
            self.get_embedding_copy(pid, D_, offset)
        if gpu:
            D_ = D_.cuda()
        maxscoreQ = (Q @ D_.permute(0, 2, 1)).max(2).values.cpu()
        scores = (weightsQ*maxscoreQ).sum(1).cpu()
        return scores.tolist()

    def our_rerank_with_embeddings_batched(self, qembs, pids, weightsQ=None, gpu=True, batch_size=1000):
        import more_itertools
        if len(pids) < batch_size:
            return self.our_rerank_with_embeddings(qembs, pids, weightsQ, gpu)
        allscores=[]
        for group in more_itertools.chunked(pids, batch_size):
            batch_scores = self.our_rerank_with_embeddings(qembs, group, weightsQ, gpu)
            allscores.extend(batch_scores)
        return allscores

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
            # colbert_config = ColBERTConfig.from_existing(ColBERTConfig.load_from_checkpoint(colbert_model))
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
                print(f"Encoded query: {Q[0]}")  # Print debugging information
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
        embsD, idsD = self.args.inference.docFromText([document], with_ids=True)
        return self._explain(query, embsD, idsD)
    
    def _explain(self, query, embsD, idsD):
        embsQ, idsQ, masksQ = self.args.inference.queryFromText([query], with_ids=True)

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

            # print(f"Number of passages: {len(passages)}")
            # for i, passage in enumerate(passages[:5]): 
            #     print(f"Passage {i}: {passage[:100]}...") 

            Q = inference.queryFromText([query]).cpu()
            print(f"Query embedding device: {Q.device}") 
            print(f"Query shape: {Q.shape}") 

            doc_tokenizer = DocTokenizer(config=colbert.colbert_config)
            input_ids, attention_mask = doc_tokenizer.tensorize(passages)
            D_padded, D_mask = inference.doc(input_ids, attention_mask, keep_dims='return_mask')

            # D_padded_tuple = inference.docFromText(passages, bsize=args.bsize, keep_dims="return_mask")
            # D_padded, D_mask = D_padded_tuple

            # if isinstance(D_mask, list):
            #     D_mask = torch.tensor(D_mask)

            # D_padded = D_padded
            # D_mask = D_mask.cpu()
            print(f"D_padded shape: {D_padded.shape}")
            print(f"D_mask shape: {D_mask.shape}")

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
                print(f"Scores: {scores}")  # Print debugging information
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
        docno_file = os.path.join(index_location, "docnos.tsv")

        with open(docno_file, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    line_idx = int(parts[0])
                    docno = parts[1]
                    self.docno_mapping[line_idx] = docno
        
        #delete the temporary text file
        temp_text_file = "temp_texts.tsv"
        if os.path.exists(temp_text_file):
            os.remove(temp_text_file)
            print(f"Temporary text file {temp_text_file} deleted successfully.")
        else:
            print(f"Temporary text file {temp_text_file} not found or already deleted.")

    def end_to_end(self, k=100) -> pt.Transformer:
        def _search(df_query):
            # TODO make df_queries into a colbert.Queries object
            assert len(df_query) == 1
            # encode Q
            Q = self.searcher.encode([df_query.iloc[0]["query"]])
            print("[DEBUG] Encoded Q shape:", Q.shape) 
            print("[DEBUG] Encoded Q:", Q) 

            
            # call colbert.Searcher
            doc_ids, ranks, scores = self.searcher.dense_search(Q, k=k)
            print("[DEBUG] Document IDs:", doc_ids) 
            print("[DEBUG] Scores:", scores) 
            
            # convert ranking into df_results dataframe
            df_results  = []
            for idx, docid in enumerate(doc_ids):
                # print(type(docid))
                # df_results.append([df_query.iloc[0]["qid"], docid, scores[idx], ranks[idx]])
                docno = self.docno_mapping.get(docid, "unknown_docno")
                df_results.append([df_query.iloc[0]["qid"], docno, scores[idx], ranks[idx]])

            df_results = pd.DataFrame(df_results, columns=["qid", "docno", "score", "rank"])
            print("[DEBUG] Result DataFrame:", df_results)
            return df_results
        return pt.apply.by_query(_search)
    

import pandas as pd

class ColbertPRF(pt.Transformer):
    def __init__(self, pytcfactory, k, fb_embs, beta=1, r = 42, return_docs = False, fb_docs=10,  *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.fb_embs = fb_embs
        self.beta = beta
        self.return_docs = return_docs
        self.fb_docs = fb_docs
        self.pytcfactory = pytcfactory
        self.fnt = pytcfactory.nn_term(df=True)
        self.r = r
        import torch
        import numpy as np
        num_docs = self.fnt.num_docs
        self.idfdict = {}
        for tid in pt.tqdm(range(self.fnt.inference.query_tokenizer.tok.vocab_size)):
            df = self.fnt.getDF_by_id(tid)
            idfscore = np.log((1.0+num_docs)/(df+1))
            self.idfdict[tid] = idfscore
        assert self.k > self.fb_embs ,"fb_embs should be smaller than number of clusters"
        self._init_clustering()

    def _init_clustering(self):
        import sklearn
        from packaging.version import Version
        from warnings import warn
        if Version(sklearn.__version__) > Version('0.23.2'):
            warn("You have sklearn version %s - sklearn KMeans clustering changed in 0.24, so performance may differ from those reported in the ICTIR 2021 paper, which used 0.23.2. "
            "See also https://github.com/scikit-learn/scikit-learn/issues/19990" % str(sklearn.__version__))

    def _get_centroids(self, prf_embs):
        from sklearn.cluster import KMeans
        kmn = KMeans(self.k, random_state=self.r)
        kmn.fit(prf_embs)
        return np.float32(kmn.cluster_centers_)
        
    def transform_query(self, topic_and_res : pd.DataFrame) -> pd.DataFrame:
        topic_and_res = topic_and_res.sort_values('rank')
        if 'doc_embs' in topic_and_res.columns:
            prf_embs = torch.cat(topic_and_res.head(self.fb_docs).doc_embs.tolist())
        else:
            prf_embs = torch.cat([self.pytcfactory.rrm.get_embedding(docid) for docid in topic_and_res.head(self.fb_docs).docid.values])
        
        # perform clustering on the document embeddings to identify the representative embeddings
        centroids = self._get_centroids(prf_embs)
        
        # get the most likely tokens for each centroid        
        toks2freqs = self.fnt.get_nearest_tokens_for_embs(centroids)

        # rank the clusters by descending idf
        emb_and_score = []
        for cluster, tok2freq in zip(range(self.k),toks2freqs):
            if len(tok2freq) == 0:
                continue
            most_likely_tok = max(tok2freq, key=tok2freq.get)
            tid = self.fnt.inference.query_tokenizer.tok.convert_tokens_to_ids(most_likely_tok)
            emb_and_score.append( (centroids[cluster], most_likely_tok, tid, self.idfdict[tid]) ) 
        sorted_by_second = sorted(emb_and_score, key=lambda tup: -tup[3])
        

       # build up the new dataframe columns
        toks=[]
        scores=[]
        exp_embds = []
        for i in range(min(self.fb_embs, len(sorted_by_second))):
            emb, tok, tid, score = sorted_by_second[i]
            toks.append(tok)
            scores.append(score)
            exp_embds.append(emb)
        
        first_row = topic_and_res.iloc[0]
        
        # concatenate the new embeddings to the existing query embeddings 
        newemb = torch.cat([
            first_row.query_embs, 
            torch.Tensor(exp_embds)])
        
        # the weights column defines important of each query embedding
        weights = torch.cat([ 
            torch.ones(len(first_row.query_embs)),
            self.beta * torch.Tensor(scores)]
        )
        
        # generate the revised query dataframe row
        rtr = pd.DataFrame([
            [first_row.qid, 
             first_row.query, 
             newemb, 
             toks, 
             weights ]
            ],
            columns=["qid", "query", "query_embs", "query_toks", "query_weights"])
        return rtr

    def transform(self, topics_and_docs : pd.DataFrame) -> pd.DataFrame:
        # validation of the input
        required = ["qid", "query", "docno", "query_embs", "rank"]
        for col in required:
            if not col in topics_and_docs.columns:
                raise KeyError("Input missing column %s, found %s" % (col, str(list(topics_and_docs.columns))) )
        
        #restore the docid column if missing
        if "docid" not in topics_and_docs.columns and "doc_embs" not in topics_and_docs.columns:
            topics_and_docs = self.pytcfactory._add_docids(topics_and_docs)
        
        rtr = []
        for qid, res in topics_and_docs.groupby("qid"):
            new_query_df = self.transform_query(res)     
            if self.return_docs:
                rtr_cols = ["qid"]
                for col in ["doc_embs", "doc_toks", "docid", "docno"]:
                    if col in res.columns:
                        rtr_cols.append(col)
                new_query_df = res[rtr_cols].merge(new_query_df, on=["qid"])
                
            rtr.append(new_query_df)
        return pd.concat(rtr)


def _approx_maxsim_numpy(faiss_scores, faiss_ids, mapping, weights, score_buffer):
    import numpy as np
    faiss_depth = faiss_scores.shape[1]
    pids = mapping[faiss_ids]
    qemb_ids = np.arange(faiss_ids.shape[0])
    for rank in range(faiss_depth):
        rank_pids = pids[:, rank]
        score_buffer[rank_pids, qemb_ids] = np.maximum(score_buffer[rank_pids, qemb_ids], faiss_scores[:, rank])
    all_pids = np.unique(pids)
    final = np.sum(score_buffer[all_pids, : ] * weights, axis=1)
    score_buffer[all_pids, : ] = 0
    return all_pids, final

def _approx_maxsim_defaultdict(all_scores, all_embedding_ids, mapping, qweights, ignore2):
    from collections import defaultdict
    pid2score = defaultdict(float)

    # dont rely on ids.shape here for the number of query embeddings in the query
    for qpos in range(all_scores.shape[0]):
        scores = all_scores[qpos]
        embedding_ids = all_embedding_ids[qpos]
        pids = mapping[embedding_ids]
        qpos_scores = defaultdict(float)
        for (score, pid) in zip(scores, pids):
            _pid = int(pid)
            qpos_scores[_pid] = max(qpos_scores[_pid], score)
        for (pid, score) in qpos_scores.items():
            pid2score[pid] += score * qweights[ qpos].item()
    return list(pid2score.keys()), list(pid2score.values())

def _approx_maxsim_sparse(all_scores, all_embedding_ids, mapping, qweights, ignore2):
    from scipy.sparse import csr_matrix
    import numpy as np
    index_size = 9_000_000 # TODO: use total # of documents here instead of hardcoding
    num_qembs = all_scores.shape[0]
    faiss_depth = all_scores.shape[1]
    all_pids = mapping[all_embedding_ids]

    pid2score = csr_matrix((1, index_size))
    for qpos in range( num_qembs ):
        a = csr_matrix((all_scores[qpos], (np.arange(faiss_depth), all_pids[qpos])), shape=(faiss_depth, index_size))
        pid2score += a.max(axis=0) * qweights[qpos]
    return (pid2score.indices, pid2score.data)
