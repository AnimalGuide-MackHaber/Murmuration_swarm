Specification 01: Core Data Structures & Type System

1. Overview

This module defines the strict Pydantic models that serve as the "Source of Truth" for the entire application. All modules (Physics, IO, UI, Orchestrator) must import and use these types to ensure type safety and serialization compatibility.

2. Geometric Primitives

2.1 Class Vector3

Description: Represents a point or vector in 3D Cartesian space.

Fields:

x: float

y: float

z: float

Methods:

to_array() -> List[float]: Returns [x, y, z] for Numpy compatibility.

from_array(arr: List[float]) -> Vector3: Static factory method.

distance_to(other: Vector3) -> float: Euclidean distance calculation.

3. Agent Definition (Static)

3.1 Class AgentConfig (The "DNA")

Description: Represents the immutable configuration of an agent. Stored as agent_{name}.json.

Fields:

id: UUID string (Primary Key).

name: string (Display Name).

domain: string (e.g., "Structural Engineering").

knowledge_graph_path: string (Relative path to the KG JSON).

system_prompt_template: string (Base instructions, e.g., "You are a skeptic...").

interaction_style: Enum [Collaborative, Debater, Socratic, Critic].

color: string (Hex code for UI visualization, e.g., "#FF0000").

perception_range: float (Default: 25.0).

4. Simulation State (Dynamic)

4.1 Class ChatMessage

Description: A single unit of communication or thought.

Fields:

id: UUID string.

timestamp: datetime.

source_agent_id: string.

content: string.

type: Enum [internal_thought, broadcast_message, system_log].

distance_at_receipt: Optional[float]. (If type is broadcast_message, this stores how far the listener was from the speaker. Used for weighting importance).

4.2 Class AgentRuntimeState

Description: The mutable state of an agent during a simulation tick.

Fields:

config_id: string (Reference to AgentConfig.id).

position: Vector3.

velocity: Vector3.

acceleration: Vector3 (Reset every tick).

private_memory: List[ChatMessage] (The agent's subjective history).

latest_thought: Optional[string] (The most recent output, exposed for peers to potentially "hear").

4.3 Class SwarmSnapshot (The "Save File")

Description: The serializable root object for the entire simulation state.

Fields:

simulation_id: string.

iteration: integer (Current time step).

max_iterations: integer.

universal_prompt: string (The "Goal" of the swarm).

agent_configs: Dict[str, AgentConfig] (Registry of all involved agents).

agent_states: Dict[str, AgentRuntimeState] (Current positions/memories).

influence_matrix: Dict[str, Dict[str, float]] (Current adjacency weights).