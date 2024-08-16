import os
import torch
import pyterrier as pt

# Initialize PyTerrier first
if not pt.started():
    pt.init()

import pandas as pd
from colbert.modeling.colbert import ColBERT
from colbert.searcher import Searcher
from colbert.data.queries import Queries

# Define model path
colbert_model_path = '/mnt/c/Users/DJH/Desktop/code/colbertv2.0'
index_location = '/mnt/c/Users/DJH/Desktop/code/experiments/msmarco.nbits=2'


class ColBERTv2Index(ColBERTModelOnlyFactory):

    def __init__(self, colbert, index_location, index_name, **kwargs):
        super().__init__(colbert, **kwargs)
        # LOAD index from index_location - colbert.Searcher
        self.searcher = Searcher(index_location,index_name)

    def end_to_end(self, k=100) -> pt.Transformer():
        def _search(df_queries):
            # TODO make df_queries into a colbert.Queries object
            queries_dict = {row['qid']: row['query'] for _, row in df_queries.iterrows()}
            queries = Queries(data=queries_dict)
            # call colbert.Searcher
            results = self.searcher.search(queries, k=k)
            # convert ranking into df_results dataframe
            results = []
            for qid, query_ranking in ranking.items():
                for rank, (docid, score) in enumerate(query_ranking):
                    results.append([qid, docid, score, rank])

            df_results = pd.DataFrame(results, columns=["qid", "docno", "score", "rank"])
            return df_results
        return pt.apply.generic(_search)
