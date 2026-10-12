MAAI System Technical Specification

1. System Overview

This document outlines the technical architecture for a Multi-Agent AI (MAAI) system designed for continuous open-ended exploration without succumbing to mode collapse or premature convergence. The system replaces standard static multi-agent routing with a dynamic, graph-based orchestration layer. It utilizes continuous asynchronous context updates, Low-Rank Adapter (LoRA) swapping, and targeted knowledge pruning to maintain system flexibility.

2. Core Architecture & Agent Definition

2.1 Agent Composition

Agents are not monolithic, continuously updated base models. Real-time backpropagation on base model weights is computationally infeasible. Instead, an agent is defined as:

Base Model: A frozen, shared Large Language Model (LLM) policy network.

Dynamic State: A dynamically loaded Low-Rank Adapter (LoRA) specific to the agent's current role and objective.

Memory Buffer: A local, short-term context window (the "Frontier Buffer") managed via standard Retrieval-Augmented Generation (RAG).

2.2 State Space and Orchestration

Distributed Ledger of Epistemic Commitments: The global state is managed as a shared Vector Database (e.g., Milvus, Pinecone) functioning as a Knowledge Graph. Nodes represent axioms, assertions, or data clusters; edges represent semantic relationships.

Hypergraph Routing: Agent communication is routed via a Graph Neural Network (GNN) layer. Communication pathways are weighted based on semantic relevance, allowing for a non-rigid, multiplex network topology.

Cross-Frequency Coupling: The system operates on two asynchronous loops. Local agent tasks execute at high frequency (low latency), while global context synchronization and LoRA updates occur at a lower frequency to prevent global phase-locking (system-wide repetitive loops).

3. Epistemic Trophic Cascades (ETC) - Pipeline & Credit Assignment

The system divides compute and roles into a directed pipeline to process information.

3.1 Role Definitions

Producers: Ingest raw external data and generate high-entropy heuristic proposals.

Consumers: Filter and categorize Producer outputs against the current knowledge graph.

Synthesizers: Aggregate filtered data into higher-order assertions, generating training data for future LoRA updates.

Iconoclasts (Garbage Collectors): Execute gradient-based topological pruning. They identify and flag highly redundant nodes or circular logic loops within the Knowledge Graph for deletion or compression.

3.2 Compute Economy & Credit Assignment

Compute budget (inference priority) is dynamically allocated based on measured utility, replacing standard scalar MARL rewards.

Transfer Entropy (TE) Metric: The reward signal is quantified by measuring the reduction in uncertainty (predictive information gain) a Producer's output provides to a Consumer's model.

Diversity Bonus: To prevent homogeneous outputs, the reward function includes a Determinantal Point Process (DPP) penalty to ensure agents occupy distinct semantic regions.

Decomposition (Recycling): When an Iconoclast prunes a node, the underlying tokens are not merely deleted but are compressed into low-dimensional embeddings and returned to the Producer pool to seed new queries.

4. Epistemic Regulation (VSE & XOH)

To prevent the system from over-optimizing on a static dataset, the architecture enforces artificial exploration and data isolation.

4.1 Knowledge Pruning (Void-State Epistemology)

Hyper-Network Masking: Rather than permanently deleting network weights, the system uses a secondary network to generate attention masks. These masks artificially block access to highly saturated regions of the latent space (forcing "blind spots").

Novelty Tracking: This masking is governed by a Random Network Distillation (RND) module that tracks state visitation. Highly visited semantic concepts are temporarily masked, forcing the LLMs to generate novel connections.

4.2 Exogenous Sandboxing (XOH)

Isolated Environments: "Out-of-Distribution" (OOD) data, adversarial prompts, and high-temperature stochastic generations are quarantined in isolated vector database namespaces.

Localized Logic Fusion: The core system does not globally merge with this sandbox. Instead, an asynchronous process monitors for "Semantic Overlap." If a high-value correlation is found between the core graph and the sandbox, specific context shards or LoRA weight-deltas are extracted and merged into the main pipeline via SLERP (Spherical Linear Interpolation).

5. System Telemetry & Control (Criticality Governor)

Real-time Topological Data Analysis (TDA) and Lyapunov Exponent calculations on full LLM latent spaces are computationally unviable ($O(n^3)$). The system utilizes the following approximation pipeline:

5.1 Dimensionality Reduction & TDA

Topological Sketching: The orchestrator periodically samples the active communication graph and agent hidden states, projecting them into a low-dimensional space using UMAP (Uniform Manifold Approximation and Projection).

Mapper Algorithm: The system applies the Mapper algorithm to this compressed space to calculate Betti numbers ($\beta_0$ for connected components, $\beta_1$ for cycles/holes).

Proxy Lyapunov Exponents: Instead of full trajectory divergence, the system monitors the Spectral Radius of the Jacobian of the GNN communication matrix.

5.2 The Control Loop

A global "Criticality Governor" script monitors these metrics to adjust system parameters automatically:

Stagnation Trigger: If Betti numbers drop (indicating a loss of complexity) or the Spectral Radius drops below 1 (indicating mode collapse), the Governor increases the injection rate of OOD data from the XOH sandboxes and increases the attention masking rate (VSE).

Chaos Trigger: If the Spectral Radius significantly exceeds 1 (indicating divergent, incoherent outputs), the Governor increases the Iconoclast pruning rate and restricts compute budgets to focus solely on the Synthesizer layers.