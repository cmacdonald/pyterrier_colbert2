Reranking
=========================

.. automodule:: pyterrier_colbert.ranking._modelonly
   :members:
   :undoc-members:
   :show-inheritance:


Overview
-------------------------

This files documents ColBERT utilities for query and
document encoding, text scoring, and interaction visualization. These don't 
require an index to be built, and can be used for reranking or other tasks.

Typical usage:

.. code-block:: python

   from pyterrier_colbert.ranking import ColBERTModelOnlyFactory

   factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
   q_encoder = factory.query_encoder()
   d_encoder = factory.text_encoder()
   scorer = factory.text_scorer()


Transformer Examples
-------------------------

The examples below show each transformer and a corresponding PyTerrier
schematic.

Query Encoder (``q_encoder``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encodes the ``query`` column into ``query_embs``.

.. code-block:: python

   from pyterrier_colbert.ranking import ColBERTModelOnlyFactory

   factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
   q_encoder = factory.query_encoder()

.. schematic::
    from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
    factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
    q_encoder = factory.query_encoder()
    q_encoder


Document Encoder (``d_encoder``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Encodes the ``text`` column into ``doc_embs``.

.. code-block:: python

   from pyterrier_colbert.ranking import ColBERTModelOnlyFactory

   factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
   d_encoder = factory.text_encoder()
   d_encoder([{"docno": "doc1", "text": "This is a test document."}])
   # returns an iter-dict with columns: docno, text, doc_embs

.. schematic::
    from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
    factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
    d_encoder = factory.text_encoder()
    d_encoder


Text Scorer (``text_scorer``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Scores candidate documents directly from raw text and query text.

.. code-block:: python

   from pyterrier_colbert.ranking import ColBERTModelOnlyFactory

   factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
   text_scorer = factory.text_scorer()
   text_scorer([{"query": "test query", "text": "test document"}])
   # returns an iter-dict with columns: query, text, score

.. schematic::
    from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
    factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
    text_scorer = factory.text_scorer()
    text_scorer


Embedding Scorer (``scorer``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Builds a scoring pipeline over precomputed query and document embeddings.

.. code-block:: python

   from pyterrier_colbert.ranking import ColBERTModelOnlyFactory

   factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
   emb_pipeline = factory.query_encoder() >> factory.text_encoder() >> factory.scorer(gpu=False)
   emb_pipeline([{"query": "test query", "text": "test document"}])
   # returns an iter-dict with columns: query, text, score

.. schematic::
    from pyterrier_colbert.ranking import ColBERTModelOnlyFactory
    factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
    emb_pipeline = factory.query_encoder() >> factory.text_encoder() >> factory.scorer(gpu=False)
    emb_pipeline



API Documentation
-------------------------

.. autoclass:: pyterrier_colbert.ranking.ColBERTModelOnlyFactory
   :members:
   :undoc-members:
   :show-inheritance:
