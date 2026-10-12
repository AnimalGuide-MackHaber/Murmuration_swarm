---
name: semantic-rag-traversal
description: "Optimize and evaluate Knowledge Graph query search, TF-IDF scoring, and BFS traversal retrieval."
---

# Semantic RAG & Traversal Skill

## Overview
This skill handles how context is retrieved from JSON-based Knowledge Graphs. It guides agents on evaluating cosine similarities of query texts against node summaries, followed by traversing children linkages to assembly a comprehensive prompt context.

## Codebase References
- [knowledge_graph.py:L441-L541](file:///Users/mike/Documents/code/Murmur_swarm/core/knowledge_graph.py#L441-L541) - Implementation of `retrieve_context` using `TfidfVectorizer` and BFS traversal logic.

## Usage Guide
1. Tune standard retrieval depths (e.g., `max_retrieval_depth`) to capture connected details.
2. Verify that retrieval context formats do not overflow prompt token boundaries.
3. Test edge-case queries to confirm semantic links are traversed correctly.

## Tooling & Scripts
- Run query traversal diagnostic test: `python skills/semantic-rag-traversal/scripts/evaluate_rag_retrieval.py` (when implemented).
