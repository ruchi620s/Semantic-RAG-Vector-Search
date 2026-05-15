from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(slots=True)
class MockGenerationResponse:
    text: str


class MockGenerativeModel:
    """Local mock of vertexai.generative_models.GenerativeModel."""

    def __init__(self, model_name: str = "gemini-mock") -> None:
        self.model_name = model_name

    def generate_content(self, prompt: str) -> MockGenerationResponse:
        return MockGenerationResponse(text=prompt)


class QueryExpansionService:
    """Expands short/vague infrastructure questions into richer technical queries."""

    def __init__(self, model_name: str = "gemini-mock") -> None:
        self.model = MockGenerativeModel(model_name=model_name)
        self.topic_map: Dict[str, List[str]] = {
            "peak load": [
                "traffic spikes",
                "autoscaling",
                "horizontal pod autoscaler",
                "load balancing",
                "high concurrent requests",
                "queue backpressure",
            ],
            "response speed": [
                "caching",
                "cache invalidation",
                "api gateway routing",
                "connection pooling",
                "latency optimization",
                "cdn edge caching",
            ],
            "failures isolated": [
                "fault isolation",
                "circuit breaker",
                "bulkhead pattern",
                "retry with exponential backoff",
                "graceful degradation",
                "microservice boundaries",
            ],
        }

    def _keyword_enrichment(self, query: str) -> List[str]:
        q = query.lower()
        enrichments: List[str] = []
        if "peak" in q or "load" in q or "spike" in q:
            enrichments.extend(self.topic_map["peak load"])
        if "response" in q or "speed" in q or "latency" in q:
            enrichments.extend(self.topic_map["response speed"])
        if "failure" in q or "isolate" in q or "resilien" in q:
            enrichments.extend(self.topic_map["failures isolated"])
        if "kubernetes" in q or "k8s" in q:
            enrichments.extend(["pod disruption budget", "readiness probes", "liveness probes"])
        if "api" in q:
            enrichments.extend(["rate limiting", "throttling", "request routing", "auth offloading"])
        return sorted(set(enrichments))

    def expand_query(self, query: str) -> str:
        enrichments = self._keyword_enrichment(query)
        if not enrichments:
            enrichments = [
                "distributed systems",
                "scalability",
                "fault tolerance",
                "observability",
            ]

        expanded = (
            f"{query.strip()} Focus on distributed systems behavior including "
            f"{', '.join(enrichments)}."
        )

        # Keep a call to the mock model to emulate SDK usage pattern.
        response = self.model.generate_content(expanded)
        return response.text
