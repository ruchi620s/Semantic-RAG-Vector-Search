from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console
from rich.table import Table

from app.retrieval.retriever import Retriever


class BenchmarkRunner:
    """Runs comparative retrieval benchmark and exports JSON/Markdown reports."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.console = Console()

    @staticmethod
    def _observation(raw_results: List[Dict[str, Any]], enhanced_results: List[Dict[str, Any]]) -> str:
        if not raw_results or not enhanced_results:
            return "Insufficient results for comparison."

        raw_avg = sum(item["score"] for item in raw_results) / len(raw_results)
        enh_avg = sum(item["score"] for item in enhanced_results) / len(enhanced_results)

        if enh_avg > raw_avg:
            return "Enhanced retrieval produced stronger semantic alignment on average."
        if enh_avg < raw_avg:
            return "Raw retrieval scored higher; query expansion may have added noise for this query."
        return "Both strategies scored similarly for this query."

    def run(self, queries: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []

        for query in queries:
            raw = self.retriever.raw_search(query=query, top_k=top_k)
            enhanced = self.retriever.enhanced_search(query=query, top_k=top_k, hybrid=True)

            entry = {
                "original_query": query,
                "expanded_query": enhanced["expanded_query"],
                "strategy_a": raw,
                "strategy_b": enhanced,
                "observation": self._observation(raw["results"], enhanced["results"]),
            }
            records.append(entry)

        return records

    def print_console_table(self, records: List[Dict[str, Any]]) -> None:
        table = Table(title="Semantic RAG & Vector Search Benchmark", show_lines=True)
        table.add_column("Query", style="cyan", overflow="fold")
        table.add_column("Strategy A Top-1", style="white", overflow="fold")
        table.add_column("A Score", justify="right", style="magenta")
        table.add_column("Strategy B Top-1", style="white", overflow="fold")
        table.add_column("B Score", justify="right", style="green")
        table.add_column("Observation", style="yellow", overflow="fold")

        for row in records:
            a1 = row["strategy_a"]["results"][0] if row["strategy_a"]["results"] else {"text": "-", "score": 0}
            b1 = row["strategy_b"]["results"][0] if row["strategy_b"]["results"] else {"text": "-", "score": 0}
            table.add_row(
                row["original_query"],
                a1["text"][:110] + ("..." if len(a1["text"]) > 110 else ""),
                f"{a1['score']:.4f}",
                b1["text"][:110] + ("..." if len(b1["text"]) > 110 else ""),
                f"{b1['score']:.4f}",
                row["observation"],
            )

        self.console.print(table)

    @staticmethod
    def export_json(records: List[Dict[str, Any]], output_path: str) -> None:
        Path(output_path).write_text(json.dumps(records, indent=2), encoding="utf-8")

    @staticmethod
    def export_markdown(records: List[Dict[str, Any]], output_path: str) -> None:
        lines: List[str] = [
            "# Retrieval Benchmark Report",
            "",
            "This report compares raw vector retrieval against AI-enhanced query expansion retrieval.",
            "",
        ]

        for idx, row in enumerate(records, start=1):
            lines.append(f"## Query {idx}")
            lines.append("")
            lines.append(f"- Original Query: {row['original_query']}")
            lines.append(f"- Expanded Query: {row['expanded_query']}")
            lines.append(f"- Observation: {row['observation']}")
            lines.append("")
            lines.append("### Strategy A - Raw Vector Search")
            for item in row["strategy_a"]["results"]:
                lines.append(
                    f"- [{item['chunk_id']}] score={item['score']:.4f} | {item['text'][:140]}"
                )
            lines.append("")
            lines.append("### Strategy B - Enhanced Retrieval")
            for item in row["strategy_b"]["results"]:
                lines.append(
                    f"- [{item['chunk_id']}] score={item['score']:.4f} | {item['text'][:140]}"
                )
            lines.append("")

        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
