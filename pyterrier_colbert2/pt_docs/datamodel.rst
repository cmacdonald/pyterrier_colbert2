.. _pyterrier_colbert.datamodel:

ColBERT Data Model
==================

This package extends the standard PyTerrier data model with additional columns
for embedding-based retrieval, reranking, and PLAID/PRF internals.

The table below documents the extra columns used by this repository.

+----------------+------------------------------+-----------------------------+------------------------------------------+
| Column         | Type                         | Produced by                 | Consumed by                              |
+================+==============================+=============================+==========================================+
| ``query_embs`` | Query embedding tensor       | ``query_encoder()``         | ``text_scorer(query_encoded=True)``,     |
|                |                              |                             | ``scorer()``                             |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``doc_embs``   | Document embedding tensor    | ``text_encoder()``          | ``scorer()``                             |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``query_vec``  | Query embedding tensor       | ``plaid_prf()``             | ``end_to_end(query_encoded=True)``       |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``Q_embs``     | Encoded query (PLAID stage) | ``plaid_candidate_generation()`` | Intermediate PLAID stage outputs    |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``pids``       | Candidate passage ids        | ``plaid_candidate_generation()`` | ``plaid_centroid_interaction()``    |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``pid``        | Single passage id            | ``plaid_centroid_interaction()`` | ``plaid_final_scoring()``           |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``query_weights`` | Per-query token weights   | User-provided optional      | ``scorer()``                             |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``query_toks`` | Query token ids              | User-provided optional      | ``scorer(add_exact_match_contribution=True)`` |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``doc_toks``   | Document token ids           | User-provided optional      | ``scorer(add_exact_match_contribution=True)`` |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``contributions`` | Per-query-term max-sim    | ``scorer(add_contributions=True)`` | Downstream analysis/debugging      |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``exact_numer`` | Exact-match numerator       | ``scorer(add_exact_match_contribution=True)`` | Downstream analysis/debugging |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``exact_denom`` | Exact-match denominator     | ``scorer(add_exact_match_contribution=True)`` | Downstream analysis/debugging |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``exact_pct``  | Exact-match ratio            | ``scorer(add_exact_match_contribution=True)`` | Downstream analysis/debugging |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``n_exp``      | Number of expansion vectors  | ``plaid_prf()``             | Downstream analysis/debugging            |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``lambda_div`` | Diversity weight used in PRF | ``plaid_prf()``             | Downstream analysis/debugging            |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``exp_idx``    | Selected expansion indexes   | ``plaid_prf()``             | Downstream analysis/debugging            |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``exp_wpids``  | Selected WordPiece ids       | ``plaid_prf(output_exptok=True)`` | Downstream analysis/debugging      |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``exp_wptoks`` | Selected WordPiece tokens    | ``plaid_prf(output_exptok=True)`` | Downstream analysis/debugging      |
+----------------+------------------------------+-----------------------------+------------------------------------------+
| ``exp_codes``  | Selected compressed codes    | ``plaid_prf()``             | Downstream analysis/debugging            |
+----------------+------------------------------+-----------------------------+------------------------------------------+


Validation Contracts
--------------------

The following methods assert extra columns via ``pt.validate`` checks:

- ``pyterrier_colbert.ranking.ColBERTv2Index.end_to_end(query_encoded=True)`` requires ``query_vec``.
- ``pyterrier_colbert.ranking.ColBERTv2Index.plaid_candidate_generation()`` requires ``query``.
- ``pyterrier_colbert.ranking.ColBERTv2Index.plaid_centroid_interaction()`` requires ``query``, ``pids``, ``score``.
- ``pyterrier_colbert.ranking.ColBERTv2Index.plaid_centroid_pruning()`` requires ``query``, ``score``.
- ``pyterrier_colbert.ranking.ColBERTv2Index.plaid_final_scoring()`` requires ``query``, ``pid``.
- ``pyterrier_colbert.ranking.ColBERTModelOnlyFactory.query_encoder()`` requires ``query`` and produces ``query_embs``.
- ``pyterrier_colbert.ranking.ColBERTModelOnlyFactory.text_encoder()`` requires ``text`` and produces ``doc_embs``.
- ``pyterrier_colbert.ranking.ColBERTModelOnlyFactory.text_scorer()`` requires ``query`` and the configured document text column.
- ``pyterrier_colbert.ranking.ColBERTModelOnlyFactory.text_scorer(query_encoded=True)`` requires ``query_embs``, ``query`` and the configured document text column.
- ``pyterrier_colbert.ranking.ColBERTModelOnlyFactory.scorer()`` requires ``query``, ``query_embs``, ``doc_embs``.
- ``pyterrier_colbert.ranking._prf.plaid_prf()`` requires ``query`` and produces ``query_vec``.


Notes
-----

- Columns such as ``qid``, ``query``, ``docno``, ``score``, and ``rank`` follow the standard PyTerrier data model.
- Some fields (for example ``Q_embs``) are internal intermediate columns intended for staged PLAID pipelines.