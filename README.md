# Semantic Search Engine with Word2Vec, Sentence-BERT, and Hybrid Reranking

A course project exploring semantic document retrieval on the MS MARCO v2.1 dataset. The notebook compares three retrieval strategies: Word2Vec with query expansion, Sentence-BERT (SBERT), and a hybrid pipeline that retrieves candidates with Word2Vec and reranks them with SBERT.

> **Project status:** Research prototype implemented in a Google Colab notebook. The notebook launches a temporary Gradio demo, but there is no permanent hosted demo link in this repository.

## What it does

- Preprocesses a subset of MS MARCO passages and trains a 300-dimensional Word2Vec model.
- Creates passage vectors by averaging Word2Vec token embeddings.
- Encodes passages and queries with `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional embeddings).
- Uses FAISS `IndexFlatIP` indexes over normalized vectors for exact inner-product similarity search (equivalent to cosine ranking when vectors are normalized).
- Expands Word2Vec queries with similar words and compares the expanded query retrieval with SBERT retrieval.
- Implements a hybrid pipeline: Word2Vec retrieves candidate passages, then SBERT reranks them. In the notebook's normal search/evaluation path, the default candidate count is 300.
- Evaluates Precision@k, Recall@k, F1@k, MRR@k, and NDCG@k for k = 5 and 10.
- Provides a Gradio interface to compare results from the three methods.

## Architecture

```text
User query
   |
   +--> Word2Vec + query expansion --> FAISS IndexFlatIP --> ranked passages
   |
   +--> Sentence-BERT query encoder --> FAISS IndexFlatIP --> ranked passages
   |
   +--> Word2Vec candidate retrieval --> candidate passages
                                      --> SBERT reranking --> final ranked passages
```

## Dataset and experimental scope

The notebook loads the `train` split of the `microsoft/ms_marco` Hugging Face dataset with configuration `v2.1`, then selects the first **200,000 query records**. Flattening the passages in those records produced **1,995,277 passages** in the saved run.

The saved model-comparison run evaluates the first 500 query records, but only **305** have at least one passage marked relevant by the dataset's `is_selected` labels; the other **195** are skipped by the evaluation function. Treat the values below as results from this limited prototype evaluation, not as a full-dataset benchmark or a held-out test-set result.

Dataset source: [MS MARCO on Hugging Face](https://huggingface.co/datasets/microsoft/ms_marco) · [MS MARCO project](https://microsoft.github.io/msmarco/)

## Results recorded in the notebook

The following values are transcribed from the notebook's saved evaluation output. Metrics are averaged over the 305 evaluated queries described above.

### At k = 5

| Method | Precision@5 | Recall@5 | F1@5 | MRR@5 | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| Word2Vec + query expansion | 0.0190 | 0.0896 | 0.0311 | 0.0465 | 0.0555 |
| Sentence-BERT (SBERT) | **0.1246** | **0.6060** | **0.2060** | **0.3860** | **0.4376** |
| Hybrid (Word2Vec + SBERT reranking) | 0.0649 | 0.3071 | 0.1064 | 0.2256 | 0.2422 |

### At k = 10

| Method | Precision@10 | Recall@10 | F1@10 | MRR@10 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Word2Vec + query expansion | 0.0115 | 0.1060 | 0.0206 | 0.0488 | 0.0611 |
| Sentence-BERT (SBERT) | **0.0764** | **0.7361** | **0.1380** | **0.4030** | **0.4807** |
| Hybrid (Word2Vec + SBERT reranking) | 0.0351 | 0.3301 | 0.0631 | 0.2285 | 0.2501 |

In this run, SBERT has the strongest retrieval metrics. The hybrid approach improves over the Word2Vec baseline, but does not reach SBERT's scores. The separate query-expansion experiment found that using three related words outperformed expansion sizes of five and ten on the evaluated subset; this observation should be rechecked on a larger, better-controlled evaluation.

## Repository contents

```text
semantic-search-engine/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── semantic_search_engine.ipynb
└── docs/
    └── semantic_search_project_paper.pdf
```

The notebook in this repository has had its saved cell outputs cleared to remove stale runtime logs and a temporary Gradio URL. Run the cells to regenerate outputs.

## Run the notebook

The current implementation is designed for Google Colab and Google Drive. It processes nearly two million passages and writes large embedding/index files, so a runtime with sufficient RAM and Drive storage is recommended. GPU acceleration can reduce Sentence-BERT embedding time when available.

1. Upload this repository to GitHub, then open `notebooks/semantic_search_engine.ipynb` in Google Colab (or open the notebook directly from your GitHub repository).
2. In a Colab code cell, clone your repository and install dependencies. Replace the placeholders with your GitHub account and repository name:

   ```python
   !git clone https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPOSITORY>.git /content/semantic-search-engine
   %pip install -q -r /content/semantic-search-engine/requirements.txt
   ```

   If you are using a local checkout instead, run `pip install -r requirements.txt` from the repository root.
3. Run the notebook from the top. It mounts Google Drive and downloads the MS MARCO dataset and the Sentence-BERT model when they are not already cached.
4. In the notebook's project-folder cell, change `PROJECT_DIR` if you want to use a different folder in your own Google Drive. The notebook stores the dataset cache and generated artifacts in that folder.
5. Run the final Gradio cell to start an interactive comparison. The default `share=True` URL is temporary and expires; it is not a permanent deployment.

The MS MARCO dataset is public; a Hugging Face token is optional in the notebook. Do not add API tokens, `.env` files, private credentials, or exported model/index files to GitHub.

## Important implementation notes

- The code uses FAISS `IndexFlatIP`, which performs exact flat inner-product search. It is **not** an approximate-nearest-neighbor index such as IVF or HNSW. The accompanying paper uses the term “ANN”; update the paper or change the index implementation if you want the description and code to match.
- The submitted paper says the corpus contains 1,000,000 passages, while the saved notebook run uses 200,000 query records and produces 1,995,277 flattened passages. These figures should be reconciled in the paper.
- The paper describes the hybrid method as reranking 100 candidates; the normal notebook evaluation and Gradio path use `rerank_k=300`. The helper used for the one-query timing comparison uses 100 candidates. Make this explicit when reporting results.
- The notebook's saved latency comparison is for one query on one runtime. Do not present it as a robust speed benchmark without repeated warm-up runs and aggregate statistics.

## Limitations

- Experiments use a subset of the MS MARCO training split and a limited set of queries with available positive labels in the selected slice.
- Word2Vec mean pooling loses word order and context; query expansion may introduce noisy terms.
- The hybrid method cannot recover a relevant passage that was not retrieved in its initial candidate stage.
- Generated embeddings and FAISS indexes are large and are intentionally not included in this repository. Recreate them by running the notebook.
- The notebook is the current implementation; this repository does not yet provide a modular command-line service or permanent web deployment.

## Project paper

The original three-page paper is available at [`docs/semantic_search_project_paper.pdf`](docs/semantic_search_project_paper.pdf). It contains the project team's names and institutional email addresses; check with co-authors before making the PDF public, or remove it from the public repository if needed.
