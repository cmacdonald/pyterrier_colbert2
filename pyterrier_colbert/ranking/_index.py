from ._modelonly import ColBERTModelOnlyFactory

import pandas as pd
import pyterrier as pt

from pyterrier import tqdm
from colbert.evaluation.load_model import load_model
from .. import load_checkpoint
# monkeypatch to use our downloading version
import colbert.evaluation.loaders
colbert.evaluation.loaders.load_checkpoint = load_checkpoint
colbert.evaluation.loaders.load_model.__globals__['load_checkpoint'] = load_checkpoint
from colbert.searcher import Searcher
from warnings import warn


class ColBERTv2Index(ColBERTModelOnlyFactory, pt.Artifact):

    ARTIFACT_TYPE = 'dense_index'
    ARTIFACT_FORMAT = 'colbert'
    ARTIFACT_PACKAGE_HINT = 'pyterrier_colbert2'

    def __init__(self, colbert, index_location, **kwargs):
        # TODO do we need the colbert checkpoint....; Searcher will load it too.

        # call both super-class constructors
        super(ColBERTModelOnlyFactory, self).__init__(colbert, **kwargs)
        super(pt.Artifact, self).__init__(index_location)
        import os
        dirs = os.path.split(index_location)
        self.searcher = Searcher(dirs[-1], index_root=os.path.join(*dirs[0:-1]))
        self.docno_mapping = {}

        # Load the docno mappings from the permanent file
        docno_file = os.path.join(index_location, "docnos.npids")
        from npids import Lookup
        self.docnos = Lookup(docno_file)


    def end_to_end(self, k=1000) -> pt.Transformer:
        def _search(df_query):
            if len(df_query) == 0:
                return pd.DataFrame(columns=["qid", "query", "docno", "score", "rank"])
            
            # TODO can we make df_queries into a colbert.Queries object to allow parallelisation?
            assert len(df_query) == 1
            # encode Q
            Q = self.searcher.encode([df_query.iloc[0]["query"]])
      
            # call colbert.Searcher
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
