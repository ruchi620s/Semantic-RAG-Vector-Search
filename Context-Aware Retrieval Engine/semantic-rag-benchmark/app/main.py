from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import sys

import numpy as np

# Support direct script execution from any terminal working directory.
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = BASE_DIR / "data" / "technical_docs.txt"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
BENCHMARK_JSON_PATH = PROJECT_ROOT / "benchmark_results.json"
BENCHMARK_MD_PATH = PROJECT_ROOT / "retrieval_benchmark.md"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.embedding.embedder import EmbeddingConfig, EmbeddingService
from app.ingestion.chunker import TextChunker
from app.ingestion.loader import load_technical_paragraphs
from app.retrieval.benchmark import BenchmarkRunner
from app.retrieval.query_expander import QueryExpansionService
from app.retrieval.retriever import Retriever
from app.vectorstore.faiss_store import DocumentChunk, FAISSVectorStore

LOGGER = logging.getLogger(__name__)

DEFAULT_QUERIES = [
    "How does the system handle peak load?",
    "What improves API response speed?",
    "How are failures isolated in distributed services?",
]


def build_retriever(data_file: Path, chunk_size: int, overlap: int) -> Retriever:
    paragraphs = load_technical_paragraphs(str(data_file))
    chunker = TextChunker(chunk_size_words=chunk_size, overlap_words=overlap)
    chunks = chunker.chunk_documents(paragraphs)

    embedding_service = EmbeddingService(
        EmbeddingConfig(
            model_name="all-MiniLM-L6-v2",
            backend="sentence-transformers",
            cache_path=str(PROJECT_ROOT / ".cache" / "embedding_cache.json"),
            use_cache=True,
            local_files_only=True,
        )
    )

    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = np.array(embedding_service.generate_embeddings(chunk_texts), dtype=np.float32)

    if embeddings.ndim != 2:
        raise RuntimeError("Embedding generation failed: expected 2D array")

    documents = [
        DocumentChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            metadata={
                "source_doc_id": chunk.source_doc_id,
                "token_start": chunk.token_start,
                "token_end": chunk.token_end,
            },
        )
        for chunk in chunks
    ]

    store = FAISSVectorStore(dimension=embeddings.shape[1], normalize=True)
    store.add_documents(documents=documents, embeddings=embeddings)

    expander = QueryExpansionService(model_name="gemini-mock")
    retriever = Retriever(embedding_service=embedding_service, vector_store=store, query_expander=expander)

    # Persist index to demonstrate production-style lifecycle.
    store.save_index(str(ARTIFACTS_DIR / "index"))

    return retriever


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic RAG & Vector Search Benchmark")
    parser.add_argument("--top-k", type=int, default=3, help="Number of results per strategy")
    parser.add_argument("--chunk-size", type=int, default=80, help="Chunk size in words")
    parser.add_argument("--chunk-overlap", type=int, default=20, help="Chunk overlap in words")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    args = parse_args()
    LOGGER.info("Current Working Directory: %s", os.getcwd())
    LOGGER.info("Script Directory: %s", BASE_DIR)
    LOGGER.info("Resolved Data Path: %s", DATA_PATH)
    LOGGER.info("File Exists: %s", DATA_PATH.exists())

    # Root cause: relative paths were resolved against terminal CWD, not this script location.
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found: {DATA_PATH}")

    retriever = build_retriever(data_file=DATA_PATH, chunk_size=args.chunk_size, overlap=args.chunk_overlap)
    benchmark = BenchmarkRunner(retriever=retriever)

    results = benchmark.run(queries=DEFAULT_QUERIES, top_k=args.top_k)
    benchmark.print_console_table(results)
    benchmark.export_json(results, str(BENCHMARK_JSON_PATH))
    benchmark.export_markdown(results, str(BENCHMARK_MD_PATH))
    retriever.embedding_service.save_cache()

    LOGGER.info("Benchmark completed. JSON: %s | Markdown: %s", BENCHMARK_JSON_PATH, BENCHMARK_MD_PATH)


if __name__ == "__main__":
    main()
