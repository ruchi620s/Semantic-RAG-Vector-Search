from __future__ import annotations

from app.retrieval.query_expander import QueryExpansionService


def test_query_expansion_enriches_peak_load_query() -> None:
    service = QueryExpansionService()
    expanded = service.expand_query("How does the system handle peak load?")

    assert "autoscaling" in expanded.lower()
    assert "load balancing" in expanded.lower()
    assert "traffic spikes" in expanded.lower()


def test_query_expansion_default_enrichment() -> None:
    service = QueryExpansionService()
    expanded = service.expand_query("Explain architecture")

    assert "distributed systems" in expanded.lower()
    assert "fault tolerance" in expanded.lower()
