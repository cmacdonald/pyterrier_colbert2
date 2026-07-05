PyTerrier ColBERT
=======================================================

`pyterrier-colbert2 <https://github.com/cmacdonald/pyterrier_colbert2>`__ is an extension for
`PyTerrier <https://github.com/terrier-org/pyterrier>`__ that provides ColBERTv2 indexing,
dense retrieval, PLAID retrieval variants, and ColBERT-based text scoring components.

PyTerrier-ColBERT2 supports:

#. End-to-end ColBERTv2 indexing via ``ColbertV2Indexer``
#. Artifact-backed dense retrieval with ``ColBERTv2Index``
#. PLAID retrieval mode with optional decomposed stage-by-stage execution
#. Model-only ColBERT reranking utilities for query encoding, text encoding, scoring, and explanation
#. PLAID PRF components for query expansion pipelines

This repo replaces the original `pyterrier-colbert <https://github.com/terrierteam/pyterrier_colbert>`__ extension, which was based on ColBERTv1.

API Documentation
---------------------------------

.. toctree::
	:maxdepth: 1

	datamodel
	indexing
	ranking
	reranking
	prf
	


Main Components
---------------------------------

**Indexing**

- ``pyterrier_colbert.indexing.ColbertV2Indexer``

	- Builds ColBERTv2 indexes from PyTerrier document iterators
	- Writes artifact metadata for loading via PyTerrier artifact APIs

**Retrieval and Ranking**

- ``pyterrier_colbert.ranking.ColBERTv2Index``

	- ``end_to_end()`` for dense retrieval
	- PLAID mode support (``plaid_mode=True``) with configurable
		``ncells``, ``centroid_score_threshold``, and ``ndocs``
	- Decomposed PLAID stages:

		- ``plaid_candidate_generation()``
		- ``plaid_centroid_interaction()``
		- ``plaid_centroid_pruning()``
		- ``plaid_final_scoring()``

**Reranking Utilities**

- ``pyterrier_colbert.ranking.ColBERTModelOnlyFactory``

	- ``query_encoder()`` for query embeddings
	- ``text_encoder()`` for document embeddings
	- ``text_scorer()`` for ColBERT text scoring (including pre-encoded query mode)
	- ``scorer()`` for max-sim scoring over precomputed embeddings
	- ``explain_text()`` for query-document interaction visualization

**PRF (PLAID)**

- ``pyterrier_colbert.ranking._prf.plaid_prf``
- ``pyterrier_colbert.ranking._prf.plaid_prf_end_to_end``

	These provide PLAID-based pseudo-relevance feedback query expansion operators.


Quick Start
---------------------------------

Create an index:

.. code-block:: python

		from pyterrier_colbert.indexing import ColbertV2Indexer

		indexer = ColbertV2Indexer(
				index_location="/path/to/index_root",
				checkpoint="colbert-ir/colbertv2.0",
				index_name="my_index",
				nbits=2,
		)
		index = indexer.index(dataset.get_corpus_iter())

Run dense retrieval:

.. code-block:: python

		dense = index.end_to_end(k=1000)
		results = dense.search("chemical reactions")

Run PLAID retrieval:

.. code-block:: python

		from pyterrier_colbert.ranking import ColBERTv2Index

		plaid = ColBERTv2Index(
				"/path/to/index_root/my_index/indexes/my_index",
				plaid_mode=True,
				ncells=4,
				centroid_score_threshold=0.5,
				ndocs=256,
		)
		results = plaid.end_to_end(k=1000).search("chemical reactions")

Use model-only text scoring:

.. code-block:: python

		from pyterrier_colbert.ranking import ColBERTModelOnlyFactory

		factory = ColBERTModelOnlyFactory("colbert-ir/colbertv2.0", gpu=False)
		scorer = factory.text_scorer()


Example Notebooks
---------------------------------

Try the notebooks in this repository root to get started:

- ``colbert v2 end to end retrieval.ipynb``
- ``colbert v2-plaid.ipynb``
- ``plaid_prf.ipynb``
- ``reranking v2 trec.ipynb``
- ``vaswani.ipynb``


Credits
---------------------------------

- Craig Macdonald, University of Glasgow
- Nicola Tonellotto, University of Pisa
- Sanjana Karumuri, University of Glasgow
- Xiao Wang, University of Glasgow
- Muhammad Hammad Khan, University of Glasgow
- Sean MacAvaney, University of Glasgow
- Sasha Petrov, University of Glasgow
