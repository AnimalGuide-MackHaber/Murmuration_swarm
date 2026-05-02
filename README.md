# Murmur Swarm 🌌

**Murmur Swarm** is a sophisticated multi-agent orchestration framework that blends **spatial physics** with **agentic cognition**. Inspired by the emergent behaviors of starlings (murmurations), this system enables a collective of specialized AI agents to interact in a 3D virtual space, influencing each other through proximity while collaboratively solving complex research problems.

---

## 🏗 System Architecture

The swarm is managed by a centralized **Orchestrator** powered by [LangGraph](https://github.com/langchain-ai/langgraph). The simulation follows a structured lifecycle where agents move, perceive, and think in iterative cycles.

```mermaid
graph TD
    Start((Start Simulation)) --> Physics[Physics Update]
    Physics --> Perceive[Spatial Perception]
    Perceive --> Think[Agent Thinking Loop]
    Think --> Continue{Iterations Complete?}
    Continue -- No --> Physics
    Continue -- Yes --> Vote[Consortium Vote]
    Vote --> Enrich[Expert Enrichment]
    Enrich --> KG[Collaborative KG Generation]
    KG --> End((Conclusion))
```

---

## 🛸 Spatial Dynamics (Physics Engine)

The swarm isn't just a list of agents; it's a **dynamic topology**. Each agent occupies a position in 3D space, and their movement is governed by a modified **Boids Algorithm**:

1.  **Separation**: Prevent crowding by maintaining distance from neighbors.
2.  **Alignment**: Steer towards the average heading of local peers.
3.  **Cohesion**: Move towards the center of mass of local peers.
4.  **Goal Seeking**: A gentle pull towards the center of the problem space (0,0,0).

### The Influence Matrix
Proximity determines **Influence**. An agent only "hears" the thoughts of others within its `visual_range`. The Orchestrator maintains a real-time `influence_matrix` where scores decay quadratically with distance. This ensures that clusters of agents develop localized sub-narratives before they merge into a global consensus.

---

## 🧠 Cognitive Dynamics (Agent Thinking)

Each agent in the swarm is a specialized persona (e.g., Structural Engineer, Iconoclast, Synthesizer). Their cognitive process follows a **ReAct (Reason + Act)** pattern:

-   **Perception**: The agent retrieves the latest thoughts from its influential neighbors.
-   **Memory Retrieval**: It queries its local **Knowledge Graph (KG)** and **Long-Term Memory (LTM)** for domain-specific context relevant to the current discussion.
-   **Internal Thought**: The agent generates a new contribution, balancing the global goal with its specific domain expertise and the local peer context.

### Parallel Execution
The system supports both sequential and high-concurrency parallel execution of agent thoughts, respecting global rate limits (RPM/TPM) across different LLM providers (Gemini, NVIDIA, Ollama).

---

## 📝 Consensus & Synthesis

Once the spatial-cognitive loop completes, the swarm enters the **Synthesis Phase**:

1.  **Consortium Vote**: Agents aggregate their final positions. A lead model synthesizes these into a primary consensus.
2.  **Expert Enrichment**: Each agent performs a technical "pass" over the consensus, adding domain-specific addendums and methodology critiques without deleting others' work.
3.  **Collaborative Knowledge Graph**: The agents collaboratively build a structured JSON Knowledge Graph representing the final conclusion, which is then persisted for future simulations.
    
---

## 🕸 Knowledge Graph Architecture

The **Knowledge Architect** transforms unstructured research material into a high-fidelity semantic network. This process ensures that agents have access to structured, hierarchical, and cross-linked domain expertise.

```mermaid
graph TD
    subgraph Ingestion["1. Data Ingestion & Decomposition"]
        PDF[PDF Documents] --> MID[MarkItDown Conversion]
        MD[Markdown Files] --> Clean[Text Cleaning]
        MID --> Chunk[Sliding Window Chunking]
        Clean --> Chunk
    end

    subgraph Extraction["2. Cognitive Extraction (LLM)"]
        Chunk --> Prompt[Architect Prompting]
        Prompt --> Concepts[Identify Core Concepts]
        Prompt --> Facts[Extract Atomic Facts]
        Prompt --> Meta[Metadata & Page Refs]
    end

    subgraph Structuring["3. Hierarchical Synthesis"]
        Concepts --> Root[Domain Root]
        Root --> Parent[Primary Nodes]
        Parent --> Child[Sub-Concepts]
        Child --> Atomic[Atomic Facts]
    end

    subgraph Linking["4. Semantic Interlinking"]
        Atomic --> Relation{Semantic Relationship?}
        Relation -- "is-a" --> Child
        Relation -- "relates-to" --> Cross[Cross-Domain Linking]
        Parent -.->|Refining Pass| Cross
    end

    subgraph Persistence["5. Graph Persistence"]
        Structuring --> JSON[Valid JSON Structure]
        Linking --> JSON
        JSON --> Storage[(Knowledge Base)]
    end

    %% Styling for visual clarity
    style Ingestion fill:#2d3436,stroke:#00cec9,stroke-width:2px,color:#fff
    style Extraction fill:#2d3436,stroke:#0984e3,stroke-width:2px,color:#fff
    style Structuring fill:#2d3436,stroke:#6c5ce7,stroke-width:2px,color:#fff
    style Linking fill:#2d3436,stroke:#e84393,stroke-width:2px,color:#fff
    style Persistence fill:#2d3436,stroke:#f1c40f,stroke-width:2px,color:#fff
```

---

## 🌌 Knowledge Graph Example

Below is a simple conceptual example of how a Knowledge Graph links semantic concepts (e.g., in a Farm domain). Nodes represent entities or concepts, and edges represent their semantic relationships.

```mermaid
graph LR
    Farm((The Farm)) --- Barn((Barn))
    Farm --- Livestock((Livestock))
    Farm --- Infrastructure((Infrastructure))
    Farm --- Personnel((Personnel))
    
    Livestock --- Cows((Cows))
    Livestock --- Bulls((Bulls))
    Cows --- Milk((Milk))
    Cows --- Calves((Calves))
    
    Infrastructure --- Tractor((Tractor))
    Infrastructure --- Fields((Fields))
    Fields --- Grass((Grass))
    Fields --- Crops((Crops))
    
    Personnel --- Farmer((Farmer))
    Personnel --- Vet((Vet))
    
    %% Semantic Interlinking
    Cows -.->|Lives in| Barn
    Cows -.->|Eats| Grass
    Bulls -.->|Sires| Calves
    Farmer -.->|Drives| Tractor
    Farmer -.->|Manages| Livestock
    Tractor -.->|Ploughs| Fields
    Grass -.->|Mowed for| Hay((Hay))
    Hay -.->|Stored in| Barn
    Vet -.->|Treats| Livestock
    Milk -.->|Stored in| Barn
    
    %% Styling
    classDef default fill:#2d3436,stroke:#00cec9,stroke-width:2px,color:#fff;
    class Farm,Barn,Livestock,Infrastructure,Personnel,Cows,Bulls,Milk,Calves,Tractor,Fields,Grass,Crops,Farmer,Vet,Hay default;
```

---

## 🔍 Knowledge Retrieval & Traversal

When an agent is asked a question, it doesn't just look for a single keyword. It performs a **semantic traversal** of the Knowledge Graph to gather holistic context.

### Example: "How do the cows get fed when it's raining?"

```mermaid
graph TD
    Prompt[/"User Prompt: 'How are cows fed during rain?'"/] --> Vectorize[Vectorize Prompt]
    Vectorize --> Similarity{Cosine Similarity}
    
    subgraph Traversal["Graph Traversal (BFS)"]
        Similarity -- "Seed Match (Score: 0.92)" --> Node1((Cows))
        Node1 -- "Hop 1: Lives in" --> Node2((Barn))
        Node1 -- "Hop 1: Eats" --> Node3((Grass))
        Node3 -- "Hop 2: Mowed for" --> Node4((Hay))
        Node4 -- "Hop 3: Stored in" --> Node2
    end
    
    Node1 --> Context[Assemble Context]
    Node2 --> Context
    Node3 --> Context
    Node4 --> Context
    
    Context --> LLM[/"Agent Thought: 'Cows eat Grass, but since it's raining, they are likely in the Barn eating the stored Hay...'"/]

    %% Styling
    style Prompt fill:#2d3436,stroke:#00cec9,stroke-width:2px,color:#fff
    style LLM fill:#2d3436,stroke:#f1c40f,stroke-width:2px,color:#fff
    style Node1 fill:#6c5ce7,stroke:#fff,stroke-width:3px,color:#fff
    style Node2 fill:#2d3436,stroke:#00cec9,stroke-width:2px,color:#fff
    style Node3 fill:#2d3436,stroke:#00cec9,stroke-width:2px,color:#fff
    style Node4 fill:#2d3436,stroke:#00cec9,stroke-width:2px,color:#fff
```

1.  **Seed Match**: The system finds the node most semantically similar to the prompt (e.g., **Cows**).
2.  **Breadth-First Search**: It traverses connected edges (e.g., **Grass**, **Barn**) up to a defined `max_retrieval_depth`.
3.  **Holistic Context**: The agent receives not just facts about "cows," but also about their environment (**Barn**) and food sources (**Hay**), enabling a much more "intelligent" response.

---

## 📂 Core Orchestration Files

| File | Responsibility |
| :--- | :--- |
| [`swarm_orchestrator.py`](file:///Users/mike/Documents/code/Murmur_swarm/swarm_orchestrator.py) | The "brain" of the system. Manages the LangGraph state machine and agent lifecycle. |
| [`swarm_physics.py`](file:///Users/mike/Documents/code/Murmur_swarm/swarm_physics.py) | Handles 3D vector math, Boids behaviors, and the Influence Matrix. |
| [`swarm_schema.py`](file:///Users/mike/Documents/code/Murmur_swarm/swarm_schema.py) | Defines the Pydantic data models for Snapshots, Agents, and Vectors. |
| [`knowledge_graph.py`](file:///Users/mike/Documents/code/Murmur_swarm/knowledge_graph.py) | Logic for RAG retrieval and collaborative graph construction. |
| [`swarm_io.py`](file:///Users/mike/Documents/code/Murmur_swarm/swarm_io.py) | Manages persistence for agent configs, snapshots, and conclusions. |

---

## 🚀 Getting Started

To launch the Swarm Intelligence Control Tower (Streamlit UI):

```bash
./.venv/bin/streamlit run app.py
```

All simulation data, including agent thoughts, exported JSONs, and final research papers, are stored in the [`swarm_data/`](file:///Users/mike/Documents/code/Murmur_swarm/swarm_data) directory.
