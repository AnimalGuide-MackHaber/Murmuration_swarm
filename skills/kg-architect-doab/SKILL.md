---
name: kg-architect-doab
description: "Run, audit, and expand the open-access book document ingestion and Knowledge Graph generation pipeline."
---

# Knowledge Graph & DOAB Ingestion Skill

## Overview
This skill outlines how to ingest PDFs/Markdown files, decompose texts using MarkItDown, chunk document bodies, query LLMs to build hierarchical concept schemas, and resolve merge conflicts across JSON-based knowledge graphs.

## Codebase References
- [knowledge_graph.py](file:///Users/mike/Documents/code/Murmur_swarm/core/knowledge_graph.py) - Text chunking, metadata extraction, auto-prompting, and TF-IDF semantic searches.
- [DOAB_client.py](file:///Users/mike/Documents/code/Murmur_swarm/core/DOAB_client.py) - Client searching and downloading PDFs from Open Access directories.
- [backend.py](file:///Users/mike/Documents/code/Murmur_swarm/backend.py) - Defines the asynchronous pipeline endpoint `/api/kgs/doab` for search terms.

## Usage Guide
1. Identify academic target topics or search prompts.
2. Trigger the DOAB pipeline to download raw PDFs.
3. Call `build_graph_async` to structure the unstructured text into domain-scoped semantic nodes.
4. Verify graph JSON compliance using integrity scripts.

## Tooling & Scripts
- Ingest documents manually: `python -c "from core.knowledge_graph import KnowledgeArchitect; KnowledgeArchitect().build_graph(['path_to_pdf'], 'domain_name')"`
- Run integrity tests on created graphs: `python skills/kg-architect-doab/scripts/verify_kg_integrity.py` (when implemented).
