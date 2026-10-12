Specification 02: I/O Manager & Knowledge Architect

1. File System Hierarchy

The system must maintain the following directory structure:

/swarm_data
    /agents             # Stores AgentConfig JSONs
    /knowledge_graphs   # Stores Hierarchical KG JSONs
    /snapshots          # Stores SwarmSnapshot JSONs
    /settings           # Stores global settings


2. Module SwarmIO

Responsibility: Handle all disk operations using Pydantic's .model_dump_json() and .model_validate_json().

Key Methods:

save_agent_config(config: AgentConfig)

load_agent_config(agent_id: str) -> AgentConfig

save_snapshot(snapshot: SwarmSnapshot)

load_snapshot(filepath: str) -> SwarmSnapshot

list_available_agents() -> List[str]

3. Module KnowledgeArchitect

Responsibility: Convert raw PDF documents into structured Knowledge Graphs (KG).

3.1 PDF Ingestion

Library: Use pypdf or LlamaParse.

Logic: Extract text page-by-page. Clean formatting (remove headers/footers).

3.2 Graph Generation (The "Architect" Agent)

Model: Gemini 1.5 Pro (Large Context).

Process:

Feed entire PDF text into context.

System Prompt: "You are a Knowledge Architect. Convert this text into a hierarchical JSON graph. Structure: Root Domain -> Core Concepts -> Sub-concepts -> Atomic Facts (with page refs)."

Output Format (JSON Enforced):

{
  "nodes": {
    "node_id": {
      "label": "String",
      "summary": "String",
      "type": "concept|fact",
      "children": ["child_id_1", "child_id_2"]
    }
  }
}


Save file to /knowledge_graphs/{domain}.json.

3.3 Graph Traversal Tool

Responsibility: Provide a function for the active agents to query this graph.

Function: query_graph(graph_path: str, node_id: str, depth: int)

Returns the summary of the target node and the label of its immediate children.