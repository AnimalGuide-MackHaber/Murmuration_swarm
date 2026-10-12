from pydantic import BaseModel, Field
from typing import List

class EpistemicNode(BaseModel):
    id: str = Field(description="Unique UUID for the node")
    content: str = Field(description="The core assertion, heuristic, or data point")
    source_layer: str = Field(description="Producer, Synthesizer, or XOH")
    confidence_score: float = Field(description="Consumer-assigned confidence (0.0 - 1.0)")
    semantic_tags: List[str] = Field(description="3-5 categorical tags for filtering")
    visitation_count: int = Field(default=0, description="Used for VSE masking")

class EpistemicNetwork(BaseModel):
    nodes: List[EpistemicNode] = Field(description="List of epistemic nodes")