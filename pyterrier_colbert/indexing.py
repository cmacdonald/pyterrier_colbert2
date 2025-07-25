import pyterrier as pt
import os
import json
import torch
import numpy as np
from colbert.indexing.loaders import load_doclens
import ujson
import random
import copy
import queue
import math
from colbert.utils.parser import Arguments
import colbert.utils.distributed as distributed
from warnings import warn
from colbert.utils.utils import create_directory

import time
import torch
import ujson
import numpy as np

import itertools
import threading
import queue

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer
from colbert.evaluation.loaders import load_colbert
from . import load_checkpoint
# monkeypatch to use our downloading version
import colbert.evaluation.loaders
colbert.evaluation.loaders.load_checkpoint = load_checkpoint
colbert.evaluation.loaders.load_model.__globals__['load_checkpoint'] = load_checkpoint
from colbert.utils.utils import print_message
import pickle
from colbert.indexing.index_manager import IndexManager
from warnings import warn

DEBUG=False


class Object(object):
  pass

class ColbertV2Indexer(pt.Indexer):

    def __init__(self, index_location, checkpoint, index_name, nbits=2):
        self.index_location = index_location
        self.checkpoint = checkpoint
        self.index_name = index_name
        self.nbits = nbits

    def index(self, iter_dict):
        

        # temp_text_file = "temp_texts.tsv"
        # temp_docno_file = "temp_docnos.tsv"

        # # Write docnos to a permanent file and texts to a temporary file
        # with open(temp_text_file, 'w') as text_fout, open(temp_docno_file, 'w') as docno_fout:
        #     for line_idx, line in enumerate(iter_dict):
        #         docno = line["docno"]
        #         cleaned_text = line['text'].replace('\n', ' ').replace('\r', ' ')
        #         text_fout.write(f"{line_idx}\t{cleaned_text}\n")
        #         docno_fout.write(f"{line_idx}\t{docno}\n")

        if not os.path.exists(self.index_location):
            os.makedirs(self.index_location)

        from timeit import default_timer as timer
        starttime = timer()

        docnos = []
        def _gen():
            for i, record in enumerate(iter_dict):
                docnos.append(record['docno'])
                yield f"{i}\t{record['text']}" 

        with Run().context(RunConfig(nranks=1, experiment=self.index_name, root=self.index_location)):
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
        from .ranking import ColBERTv2Index
        ranker = ColBERTv2Index(self.checkpoint, full_index_path)
        return ranker
