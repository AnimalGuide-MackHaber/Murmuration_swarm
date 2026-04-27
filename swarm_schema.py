from __future__ import annotations
from typing import List, Optional, Dict, Literal
from uuid import UUID, uuid4
from datetime import datetime
from pydantic import BaseModel, Field
import numpy as np

# -----------------------------------------------------------------------------
# 2. Geometric Primitives
# -----------------------------------------------------------------------------

class Vector3(BaseModel):
    """Represents a point or vector in 3D Cartesian space."""
    x: float
    y: float
    z: float

    def to_array(self) -> List[float]:
        """Returns [x, y, z] for Numpy compatibility."""
        return [self.x, self.y, self.z]

    @staticmethod
    def from_array(arr: List[float]) -> Vector3:
        """Static factory method."""
        if len(arr) != 3:
            raise ValueError("Array must have exactly 3 elements")
        return Vector3(x=arr[0], y=arr[1], z=arr[2])

    def distance_to(self, other: Vector3) -> float:
        """Euclidean distance calculation."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)


# -----------------------------------------------------------------------------
# 3. Agent Definition (Static)
# -----------------------------------------------------------------------------

class AgentConfig(BaseModel):
    """Represents the immutable configuration of an agent. Stored as agent_{name}.json."""
    id: UUID = Field(default_factory=uuid4)
    name: str
    domain: str
    knowledge_graph_path: str = "" # Relative path to the KG JSON
    system_prompt_template: str
    interaction_style: Literal['Collaborative', 'Debater', 'Socratic', 'Critic'] = 'Collaborative'
    color: str # Hex code
    perception_range: float = 25.0


# -----------------------------------------------------------------------------
# 4. Simulation State (Dynamic)
# -----------------------------------------------------------------------------

class ChatMessage(BaseModel):
    """A single unit of communication or thought."""
    id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=datetime.now)
    source_agent_id: str
    content: str
    type: Literal['internal_thought', 'broadcast_message', 'system_log']
    distance_at_receipt: Optional[float] = None


class AgentRuntimeState(BaseModel):
    """The mutable state of an agent during a simulation tick."""
    config_id: str # Reference to AgentConfig.id
    position: Vector3
    velocity: Vector3
    acceleration: Vector3
    private_memory: List[ChatMessage] = Field(default_factory=list)
    latest_thought: Optional[str] = None


class SwarmSnapshot(BaseModel):
    """The serializable root object for the entire simulation state."""
    simulation_id: str = Field(default_factory=lambda: str(uuid4()))
    iteration: int
    max_iterations: int
    universal_prompt: str
    agent_configs: Dict[str, AgentConfig]
    agent_states: Dict[str, AgentRuntimeState]
    influence_matrix: Dict[str, Dict[str, float]]
