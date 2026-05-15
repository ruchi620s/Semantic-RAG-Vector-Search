# Retrieval Benchmark Report

This report compares raw vector retrieval against AI-enhanced query expansion retrieval.

## Query 1

- Original Query: How does the system handle peak load?
- Expanded Query: How does the system handle peak load? Focus on distributed systems behavior including autoscaling, high concurrent requests, horizontal pod autoscaler, load balancing, queue backpressure, traffic spikes.
- Observation: Enhanced retrieval produced stronger semantic alignment on average.

### Strategy A - Raw Vector Search
- [doc4_chunk0] score=0.0458 | Load balancing strategies include round robin, least connections, and latency-aware routing. At scale, health checks and outlier detection a
- [doc6_chunk0] score=0.0047 | Fault tolerance in distributed services depends on graceful degradation and failure isolation. If one dependency fails, the calling service 
- [doc3_chunk0] score=-0.0106 | API gateways centralize cross-cutting concerns including authentication, rate limiting, routing, request transformation, and observability. 

### Strategy B - Enhanced Retrieval
- [doc3_chunk0] score=0.1045 | API gateways centralize cross-cutting concerns including authentication, rate limiting, routing, request transformation, and observability. 
- [doc4_chunk0] score=0.0890 | Load balancing strategies include round robin, least connections, and latency-aware routing. At scale, health checks and outlier detection a
- [doc1_chunk0] score=0.0814 | Autoscaling in Kubernetes typically combines Horizontal Pod Autoscaler metrics with cluster node autoscaling. Pods scale horizontally when C

## Query 2

- Original Query: What improves API response speed?
- Expanded Query: What improves API response speed? Focus on distributed systems behavior including api gateway routing, auth offloading, cache invalidation, caching, cdn edge caching, connection pooling, latency optimization, rate limiting, request routing, throttling.
- Observation: Raw retrieval scored higher; query expansion may have added noise for this query.

### Strategy A - Raw Vector Search
- [doc0_chunk0] score=0.1144 | Distributed systems are built to handle traffic across many independent services. A common reliability pattern is to keep services stateless
- [doc2_chunk0] score=0.0402 | Caching improves API response speed by avoiding repeated expensive computations and database round trips. Application-level caches, distribu
- [doc1_chunk0] score=0.0086 | Autoscaling in Kubernetes typically combines Horizontal Pod Autoscaler metrics with cluster node autoscaling. Pods scale horizontally when C

### Strategy B - Enhanced Retrieval
- [doc4_chunk0] score=0.0964 | Load balancing strategies include round robin, least connections, and latency-aware routing. At scale, health checks and outlier detection a
- [doc2_chunk0] score=0.0456 | Caching improves API response speed by avoiding repeated expensive computations and database round trips. Application-level caches, distribu
- [doc3_chunk0] score=0.0202 | API gateways centralize cross-cutting concerns including authentication, rate limiting, routing, request transformation, and observability. 

## Query 3

- Original Query: How are failures isolated in distributed services?
- Expanded Query: How are failures isolated in distributed services? Focus on distributed systems behavior including bulkhead pattern, circuit breaker, fault isolation, graceful degradation, microservice boundaries, retry with exponential backoff.
- Observation: Enhanced retrieval produced stronger semantic alignment on average.

### Strategy A - Raw Vector Search
- [doc7_chunk0] score=0.0722 | Observability combines logs, metrics, and traces to diagnose production incidents quickly. High-cardinality metrics can reveal hot partition
- [doc5_chunk0] score=0.0598 | Microservices isolate business capabilities into independently deployable units. Resilience patterns such as circuit breakers, retries with 
- [doc3_chunk0] score=0.0416 | API gateways centralize cross-cutting concerns including authentication, rate limiting, routing, request transformation, and observability. 

### Strategy B - Enhanced Retrieval
- [doc6_chunk0] score=0.0907 | Fault tolerance in distributed services depends on graceful degradation and failure isolation. If one dependency fails, the calling service 
- [doc2_chunk0] score=0.0616 | Caching improves API response speed by avoiding repeated expensive computations and database round trips. Application-level caches, distribu
- [doc0_chunk0] score=0.0597 | Distributed systems are built to handle traffic across many independent services. A common reliability pattern is to keep services stateless
