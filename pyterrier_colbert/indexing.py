import pyterrier as pt
import os
from warnings import warn

import torch
import numpy as np

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
#from colbert.evaluation.loaders import load_colbert
# from . import load_checkpoint
# # monkeypatch to use our downloading version
# import colbert.evaluation.loaders
# colbert.evaluation.loaders.load_checkpoint = load_checkpoint
# colbert.evaluation.loaders.load_model.__globals__['load_checkpoint'] = load_checkpoint
from colbert.utils.utils import print_message


class Object(object):
  pass

class ColbertV2Indexer(pt.Indexer):

    def __init__(self, index_location, checkpoint, index_name, nbits=2):
        self.index_location = index_location
        self.checkpoint = checkpoint
        self.index_name = index_name
        self.nbits = nbits

    def index(self, iter_dict):

        if not os.path.exists(self.index_location):
            os.makedirs(self.index_location)

        from timeit import default_timer as timer
        starttime = timer()

        docnos = []
        def _gen():
            for i, record in enumerate(iter_dict):
                docnos.append(record['docno'])
                yield f"{i}\t{record['text']}" 

        with Run().context(RunConfig(nranks=1, avoid_fork_if_possible = True, experiment=self.index_name, root=self.index_location)):
            config = ColBERTConfig(
                nbits=self.nbits,
                #avoid_fork_if_possible=True,  # prevent transformers error (no impact)
                )
            print(config)
            indexer = Indexer(checkpoint=self.checkpoint, config=config)
            # colbert needs to know the total number of passages (i.e. length of the iterator), 
            # so assumes the index is a list of strings, not an iterable
            indexer.index(name=self.index_name, collection=list(_gen()), overwrite=True)

        endtime = timer()
        print("#> V2 Indexing complete, Time elapsed %0.2f seconds" % (endtime - starttime))

        full_index_path = os.path.join(self.index_location, self.index_name, "indexes", self.index_name)
        docnos_file = os.path.join(full_index_path, "docnos.npids")

        print("#> V2 recording docnos")
        from npids import Lookup
        Lookup.build(docnos, docnos_file)
        

        print("#> done")
        import pyterrier_colbert.ranking
        ranker = pyterrier_colbert.ranking.ColBERTv2Index(self.checkpoint, full_index_path)
        return ranker
