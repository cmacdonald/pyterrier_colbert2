# PyTerrier ColBERT v2, PLAID & PLAID-PRF

[PyTerrier](https://github.com/terrier-org/pyterrier) bindings for [ColBERT v2](https://github.com/stanford-futuredata/ColBERT/) dense retrieval, with support for [PLAID](https://arxiv.org/abs/2205.09707) retrieval and [PLAID-PRF](https://eprints.gla.ac.uk/383294/) pseudo-relevance feedback.

This is a successor to the original [PyTerrier ColBERT repository](https://github.com/terrierteam/pyterrier_colbert).

## Installation

Install from PyPI:
```bash
pip install git+https://github.com/cmacdonald/pyterrier_colbert2.git
```

## Quick Start

The easiest way to get started is to use a pre-built index from HuggingFace. The following example demonstrates end-to-end PLAID retrieval on the MS MARCO passages collection:

```python
import pyterrier as pt
import pyterrier_colbert2

# Load a pre-built PLAID index from HuggingFace
index = pt.Artifact.from_hf("pyterrier/msmarco_psg_v1.colbertv2", 
    plaid_mode=True, ncells=4,
    centroid_score_threshold=0.4, ndocs=4096)

# Search using PLAID
retriever = index.end_to_end()
results = retriever.search("what are chemical reactions?")
print(results.head())
```


## Key Features

### PLAID Retrieval

PLAID (Partitioned Learned Index for Approximate matching in Dense retrieval) provides fast approximate nearest neighbor search for ColBERT. Enable it when loading an index:

```python
index = pt.Artifact.from_hf("pyterrier/msmarco_psg_v1.colbertv2", 
    plaid_mode=True, ncells=4,
    centroid_score_threshold=0.4, ndocs=4096)

# Use PLAID for efficient end-to-end retrieval
plaid_retriever = index.end_to_end()
```

### PLAID-PRF (Pseudo-Relevance Feedback)

Improve retrieval effectiveness using pseudo-relevance feedback with ColBERT embeddings:

```python
# PLAID with PRF
prf_retriever = index.plaid_prf_end_to_end(top_psg=3, top_exp=14, beta=0.7)

# Compare both approaches
results = prf_retriever.search("what are chemical reactions?")
```

The parameters control:
- `top_psg`: Number of top passages to use for PRF
- `top_exp`: Number of top expansion terms
- `beta`: Weight for the PRF component

### Evaluation


Use PyTerrier's [pt.Experiment](https://pyterrier.readthedocs.io/en/latest/experiments.html) to quickly conduct evaluation of PyTerrier_ColBERT:

```python
from pyterrier.measures import nDCG
pt.Experiment(
    {
        "PLAID": pt.rewrite.tokenise() >> index.end_to_end(), 
        "PLAID-PRF": pt.rewrite.tokenise() >> index.plaid_prf_end_to_end(top_psg=3, top_exp=14, beta=0.7)
    },
    pt.get_dataset("msmarco_passage").get_topics("test-2019"),
    pt.get_dataset("msmarco_passage").get_qrels("test-2019"),
    eval_metrics=[nDCG@10], 
)
```

## Building Custom Indexes

To build a ColBERT index from your own collection:

```python
from pyterrier_colbert2.indexing import ColbertV2Indexer
import pyterrier as pt

# Create an indexer with a ColBERT checkpoint
indexer = ColbertV2Indexer(
    index_location="/path/to/index",
    checkpoint="colbert-ir/colbertv2.0",
    index_name="my_index"
)

# Index your collection
dataset = pt.get_dataset("msmarco_passage")
index = indexer.index(dataset.get_corpus_iter())
```

Then use your indexed collection:

```python
retriever = index.end_to_end()
results = retriever.search("your query here")
```

Share your index to HuggingFace:
```python
index.to_hf("myorg/myindex")
index = pt.Artifact.from_hf("myorg/myindex")
```

## Requirements

- **GPU**: ColBERT requires a CUDA-capable GPU for inference
- **RAM**: The entire index must fit in memory. PLAID mode reduces memory requirements through its partitioned index structure
- **Python**: >= 3.9
- **OS**: Our experience with FAISS is that Linux is required.

## Examples & Notebooks

- [plaidprf-msmarcov1.ipynb](plaidprf-msmarcov1.ipynb) - PLAID and PLAID-PRF on MS MARCO passages with TREC-DL evaluations

## Resource Requirements

| Index           | Corpus Size   | Inference Time (per query) | Memory    |
|-----------------|---------------| ----------------------- | --------- |
| MSMARCO Passage | 8.8M passages | ~50ms (PLAID)            | ~28 GB    |

## References

If you use this code, please cite the relevant papers:

**ColBERT:**
```bibtex
@inproceedings{khattab2020colbert,
  title={ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT},
  author={Khattab, Omar and Zaharia, Matei},
  booktitle={Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval},
  pages={39--48},
  year={2020}
}
```

**PLAID:**
```bibtex
@inproceedings{zhang2022plaid,
  title={PLAID: An Efficient Engine for Late Interaction Retrieval},
  author={Keshav Santhanam and Omar Khattab and Christopher Potts and Matei Zaharia},
  booktitle={arXiv preprint arXiv:2205.09707},
  year={2022},
}
```

**PLAID-PRF:**
```bibtex
@inproceedings{zhang2021prf,
  title={PLAID-PRF - Pseudo-Relevance Feedback with Centroid-like Tokens in PLAID},
  author={Xiao Wang and Sean MacAvaney and Craig Macdonald},
  booktitle={SIGIR '26},
  year={2026}
}
```


## Credits
 
 - Xiao Wang, University of International Business and Economics
 - Jianhua Dong, University of Glasgow
 - Craig Macdonald, University of Glasgow
 - Sean MacAvaney, University of Glasgow


