# Semantic RAG & Vector Search Benchmark

A local, production-style Retrieval-Augmented Generation (RAG) assessment project that compares:

- Strategy A: Raw Vector Search
- Strategy B: AI-Enhanced Retrieval (Query Expansion/Rewriting + Hybrid rerank)

The project is fully local and has no cloud dependency.

## Tech Stack

- Python 3.11+
- sentence-transformers (`all-MiniLM-L6-v2` preferred)
- FAISS (`faiss-cpu`)
- NumPy
- Pytest
- Rich / tabulate
- Pandas (optional)

## Project Structure

```text
semantic-rag-benchmark/
├── app/
│   ├── embedding/
│   │   ├── embedder.py
│   │   └── mock_vertex_embedding.py
│   ├── vectorstore/
│   │   ├── faiss_store.py
│   │   └── similarity.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── query_expander.py
│   │   └── benchmark.py
│   ├── ingestion/
│   │   ├── loader.py
│   │   └── chunker.py
│   ├── data/
│   │   └── technical_docs.txt
│   └── main.py
├── tests/
├── retrieval_benchmark.md
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Benchmark

```bash
python app/main.py
```

Optional CLI flags:

```bash
python app/main.py --top-k 3 --chunk-size 80 --chunk-overlap 20
```

Outputs generated:

- `benchmark_results.json`
- `retrieval_benchmark.md`
- `artifacts/index/faiss.index` + metadata

## Run Tests

```bash
pytest -v
```

## Architecture Overview

1. Ingestion: Loads technical paragraphs and chunks with overlap.
2. Embedding: `EmbeddingService` uses sentence-transformers locally.
3. Vector Index: `FAISSVectorStore` stores normalized vectors + chunk metadata.
4. Retrieval A: Direct query embedding + top-k similarity.
5. Retrieval B: Query expansion + embedding + hybrid semantic/keyword rerank.
6. Benchmark: Compares both strategies and exports JSON/Markdown + rich table.

## Retrieval Flow

1. User query enters the retriever.
2. Strategy A embeds original query and runs FAISS search.
3. Strategy B expands query with domain terminology.
4. Expanded query is embedded and searched.
5. Optional hybrid rerank blends semantic score with keyword overlap.
6. Benchmark runner logs comparative observations.

## How Embeddings Work

`sentence-transformers` converts text into dense vectors where semantically similar text is closer in vector space. The default model here is `all-MiniLM-L6-v2`.

If the model cannot be loaded in a constrained environment, the implementation gracefully falls back to a deterministic local mock embedding backend for continuity.

## FAISS Explanation

FAISS (Facebook AI Similarity Search) is a high-performance vector indexing library. This project uses `IndexFlatIP` with normalized vectors so inner product becomes cosine similarity.

## Cosine Similarity

For vectors $a$ and $b$:

$$
\text{cosine}(a,b) = \frac{a \cdot b}{\|a\|\|b\|}
$$

With normalized vectors ($\|a\| = \|b\| = 1$), cosine similarity simplifies to dot product.

## Euclidean vs Cosine

- Euclidean distance measures absolute geometric distance.
- Cosine similarity measures angular alignment (direction) and is usually better for text embeddings.
- In text retrieval, cosine is often preferred because vector magnitude can be less meaningful than direction.

## Vertex AI Migration Path (Matching Engine)

To migrate this local benchmark to Google Vertex AI Vector Search:

1. Replace `EmbeddingService` backend with Vertex Text Embeddings API.
2. Replace local `FAISSVectorStore` with Matching Engine index endpoints.
3. Keep retrieval abstractions (`Retriever`, `BenchmarkRunner`) unchanged.
4. Replace mocks with real Vertex SDK clients and service account auth.
5. Add index deployment and endpoint lifecycle scripts.

Because the project is modular, migration is mostly implementation swapping behind interfaces.
