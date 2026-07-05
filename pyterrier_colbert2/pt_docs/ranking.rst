Ranking
=========================

.. automodule:: pyterrier_colbert.ranking._index
   :members:
   :undoc-members:
   :show-inheritance:


Overview
-------------------------

The ranking module contains the ColBERTv2 and PLAID retrieval factory used for
dense retrieval and PLAID execution.

Typical usage:

.. code-block:: python

   from pyterrier_colbert.ranking import ColBERTv2Index

   dense = ColBERTv2Index("/path/to/index_root/my_index/indexes/my_index")
   results = dense.end_to_end(k=1000).search("chemical reactions")


   Transformer Schematics
   -------------------------

   Single-stage retrieval transformer
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This schematic shows the single-stage retrieval view used by
   ``ColBERTv2Index.end_to_end()``.

   .. schematic::
      import pyterrier as pt
      single_stage = pt.apply.by_query(lambda df: df, add_ranks=False, label="ColBERTv2Index.end_to_end")
      single_stage


   Decomposed PLAID retrieval pipeline
   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

   This schematic shows the decomposed PLAID execution as a four-stage pipeline,
   equivalent to:

   ``plaid_candidate_generation() >> plaid_centroid_interaction() >> plaid_centroid_pruning() >> plaid_final_scoring()``

   .. schematic::
      import pyterrier as pt
      s1 = pt.apply.by_query(lambda df: df, label="plaid_candidate_generation")
      s2 = pt.apply.generic(lambda df: df, label="plaid_centroid_interaction")
      s3 = pt.apply.generic(lambda df: df, label="plaid_centroid_pruning")
      s4 = pt.apply.by_query(lambda df: df, label="plaid_final_scoring")
      decomposed_plaid = s1 >> s2 >> s3 >> s4
      decomposed_plaid


API Documentation
-------------------------

.. autoclass:: pyterrier_colbert.ranking.ColBERTv2Index
   :members:
   :undoc-members:
   :show-inheritance:

