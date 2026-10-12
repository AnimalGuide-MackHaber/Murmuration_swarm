import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
from core.swarm_schema import SwarmSnapshot, AgentRuntimeState, Vector3, AgentConfig
from core.swarm_orchestrator import SwarmOrchestrator
from datetime import datetime

class TestSwarmOrchestrator(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.orchestrator = SwarmOrchestrator()
        
    async def test_orchestrator_flow(self):
        # Mock LLM
        mock_model_instance = MagicMock()
        # Mock async generate_content_async
        mock_response = MagicMock()
        mock_response.text = "Mock Thought"
        mock_model_instance.generate_content_async = AsyncMock(return_value=mock_response)
        
        self.orchestrator.model = mock_model_instance
        
        # Setup Initial State
        from uuid import uuid4
        config_uuid = uuid4()
        config_id_str = str(config_uuid)
        
        agent1 = AgentRuntimeState(
            config_id=config_id_str,
            position=Vector3(x=10, y=0, z=0),
            velocity=Vector3(x=0, y=0, z=0),
            acceleration=Vector3(x=0, y=0, z=0)
        )
        config1 = AgentConfig(
            id=config_uuid,
            name="Structural",
            domain="Engineering",
            system_prompt_template="You are an engineer.",
            interaction_style="Collaborative",
            color="#FF0000"
        )
        
        snapshot = SwarmSnapshot(
            iteration=0,
            max_iterations=2, # Short run
            universal_prompt="Build a bridge",
            agent_configs={config_id_str: config1},
            agent_states={"a1": agent1},
            influence_matrix={}
        )
        
        # Run
        history, export_path = await self.orchestrator.run_full_simulation(snapshot)
        final_state = history[-1]
        
        # Verify Iterations
        self.assertGreaterEqual(final_state.iteration, 2)
        
        # Verify Thoughts
        agent_final = final_state.agent_states['a1']
        self.assertEqual(agent_final.latest_thought, "Mock Thought")
        self.assertGreater(len(agent_final.private_memory), 0)

if __name__ == '__main__':
    unittest.main()
