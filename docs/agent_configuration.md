# Agent Configuration & Settings Management

The Murmur Swarm is highly configurable, designed to be completely driven by an external JSON file (`swarm_settings.json`). This decoupling allows users to shape the physical properties of the environment, alter API rate limits per-provider, and inject totally new agents without touching the core Python codebase.

## The Settings Manager

The `SettingsManager` class parses the local configuration on initialization and establishes the global ruleset.

### 1. Physical Environment Tuning
The swarm's fluid dynamics can be heavily modified. Within the `physics` block of the JSON, users can alter:

- **Vector Weights (`separation`, `alignment`, `cohesion`, `goal_seeking`)**: These floating point values are the coefficients used in calculating the Net Acceleration. Increasing cohesion makes the swarm tighten into a ball; increasing separation forces it to scatter.
- **`max_speed`**: Hard caps the velocity vector magnitude.
- **`visual_range`**: The simulated radius within which an agent can detect peers.
- **`influence_decay_alpha`**: Modifies the steepness of the $1 / (1 + \alpha \cdot d^2)$ curve, controlling how fast attention drops off with distance.

### 2. Agent Instantiation
The `agents` array defines the genetic makeup of the swarm. Each JSON object initializes a unique Pydantic `AgentConfig`.

```json
{
  "id": "chem_01",
  "name": "Dr. Bonds",
  "domain": "Chemist",
  "interaction_style": "Analytical, data-driven, skeptical.",
  "knowledge_graph_path": "swarm_data/kbase/chemistry.faiss",
  "system_prompt_template": "You are a swarm agent participating in a spatial simulation..."
}
```

- **Domain & Interaction Style**: These fields dynamically alter the system prompt fed to the LLM. "Aggressive/Contrarian" agents will actively attempt to dismantle the theories of others during the ReAct phase.
- **Knowledge Graph Path**: Tells the orchestrator where to query FAISS/semantic DBs when this specific agent requests contextual knowledge during the simulation.

## The State Configuration (`SwarmSnapshot`)

When `app.py` begins a run, it takes the `swarm_settings.json` and uses it to construct the genesis frame: a `SwarmSnapshot`. 

This snapshot spawns `AgentRuntimeState` objects for every defined agent, randomly placing them across the 3D grid with random initial velocities, waiting for the Physics Engine to take over.
