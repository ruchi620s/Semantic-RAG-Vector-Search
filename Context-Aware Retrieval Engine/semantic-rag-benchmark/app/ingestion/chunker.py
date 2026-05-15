from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    text: str
    source_doc_id: str
    token_start: int
    token_end: int


class TextChunker:
    """Word-based chunker with configurable overlap."""

    def __init__(self, chunk_size_words: int = 80, overlap_words: int = 20) -> None:
        if overlap_words >= chunk_size_words:
            raise ValueError("overlap_words must be smaller than chunk_size_words")
        self.chunk_size_words = chunk_size_words
        self.overlap_words = overlap_words

    def chunk_documents(self, docs: List[str]) -> List[ChunkRecord]:
        chunks: List[ChunkRecord] = []
        for doc_idx, doc in enumerate(docs):
            tokens = doc.split()
            step = self.chunk_size_words - self.overlap_words
            start = 0
            chunk_idx = 0
            while start < len(tokens):
                end = min(start + self.chunk_size_words, len(tokens))
                chunk_tokens = tokens[start:end]
                chunk_text = " ".join(chunk_tokens)
                chunks.append(
                    ChunkRecord(
                        chunk_id=f"doc{doc_idx}_chunk{chunk_idx}",
                        text=chunk_text,
                        source_doc_id=f"doc{doc_idx}",
                        token_start=start,
                        token_end=end,
                    )
                )
                if end >= len(tokens):
                    break
                start += step
                chunk_idx += 1
        return chunks
