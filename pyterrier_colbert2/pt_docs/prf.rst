PLAID PRF
=========================

.. automodule:: pyterrier_colbert.ranking._prf
   :members:
   :undoc-members:
   :show-inheritance:


Overview
-------------------------

The PRF module provides PLAID pseudo-relevance feedback query expansion and the
supporting weighting and selection utilities used to build those pipelines.

Typical usage:

.. code-block:: python

   from pyterrier_colbert.ranking._prf import plaid_prf_end_to_end

   prf_pipe = plaid_prf_end_to_end(factory, top_psg=5, top_exp=16)
   results = prf_pipe.search("chemical reactions")


API Documentation
-------------------------

