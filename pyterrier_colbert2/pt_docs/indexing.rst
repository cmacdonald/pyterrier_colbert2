Indexing
=========================

.. automodule:: pyterrier_colbert.indexing
   :members:
   :undoc-members:
   :show-inheritance:


Overview
-------------------------

The indexing module contains the PyTerrier indexer used to build ColBERTv2
artifacts from a corpus iterator.

Typical usage:

.. code-block:: python

   from pyterrier_colbert.indexing import ColbertV2Indexer

   indexer = ColbertV2Indexer(
       index_location="/path/to/index_root",
       checkpoint="colbert-ir/colbertv2.0",
       index_name="my_index",
   )
   artifact = indexer.index(dataset.get_corpus_iter())
   # or, you can pass an iter-dict of documents to index
   artifact = indexer.index([{"docno": "doc1", "text": "This is a test document."}])


API Documentation
-------------------------

.. autoclass:: pyterrier_colbert.indexing.ColbertV2Indexer
   :members:
   :undoc-members:
   :show-inheritance:
