import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import json
from uuid import uuid4
from core.swarm_schema import SwarmSnapshot, AgentRuntimeState, Vector3, AgentConfig
from core.swarm_orchestrator import SwarmOrchestrator
from core.swarm_io import SwarmIO

class TestDynamicOrchestration(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.orchestrator = SwarmOrchestrator()

    @patch('core.swarm_io.SwarmIO.list_available_agents')
    @patch('core.swarm_io.SwarmIO.load_agent_config')
    async def test_allocate_agents(self, mock_load_config, mock_list_agents):
        # Setup mock agents library
        mock_list_agents.return_value = ["Agent_A", "Agent_B", "Agent_C"]
        
        def load_config_side_effect(name):
            if name == "Agent_A":
                return AgentConfig(
                    id=uuid4(), name="Agent_A", domain="Domain_A",
                    system_prompt_template="You are Agent A expert.",
                    interaction_style="Collaborative", color="#FF0000"
                )
            elif name == "Agent_B":
                return AgentConfig(
                    id=uuid4(), name="Agent_B", domain="Domain_B",
                    system_prompt_template="You are Agent B expert.",
                    interaction_style="Collaborative", color="#00FF00"
                )
            return None
        
        mock_load_config.side_effect = load_config_side_effect
        
        # Mock LLM
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "active_agents": ["Agent_A", "Agent_B"],
            "reasoning": "We need Domain A and Domain B.",
            "delegation_instructions": "Compare A and B."
        })
        
        self.orchestrator.model = AsyncMock()
        self.orchestrator.model.generate_content_async = AsyncMock(return_value=mock_response)
        
        result = await self.orchestrator._allocate_agents("Test prompt", [], [])
        
        self.assertEqual(result["active_agents"], ["Agent_A", "Agent_B"])
        self.assertEqual(result["reasoning"], "We need Domain A and Domain B.")
        self.assertEqual(result["delegation_instructions"], "Compare A and B.")

    @patch('core.swarm_io.SwarmIO.list_available_agents')
    @patch('core.swarm_io.SwarmIO.load_agent_config')
    @patch('core.swarm_io.SwarmIO.save_live_state')
    @patch('core.swarm_io.SwarmIO.save_snapshot')
    @patch('core.swarm_io.SwarmIO.save_simulation_export')
    @patch('core.swarm_io.SwarmIO.save_conclusion_markdown')
    async def test_run_dynamic_simulation(self, mock_save_md, mock_save_export, mock_save_snap, mock_save_live, mock_load_config, mock_list_agents):
        # Setup mock agents library
        mock_list_agents.return_value = ["Agent_A", "Agent_B"]
        
        uuid_a = uuid4()
        uuid_b = uuid4()
        
        def load_config_side_effect(name):
            if name == "Agent_A":
                return AgentConfig(
                    id=uuid_a, name="Agent_A", domain="Domain_A",
                    system_prompt_template="You are Agent A expert.",
                    interaction_style="Collaborative", color="#FF0000"
                )
            elif name == "Agent_B":
                return AgentConfig(
                    id=uuid_b, name="Agent_B", domain="Domain_B",
                    system_prompt_template="You are Agent B expert.",
                    interaction_style="Collaborative", color="#00FF00"
                )
            return None
        
        mock_load_config.side_effect = load_config_side_effect
        mock_save_export.return_value = "mock_export_path.json"
        
        # Mock LLM
        mock_alloc_response = MagicMock()
        mock_alloc_response.text = json.dumps({
            "active_agents": ["Agent_A", "Agent_B"],
            "reasoning": "Reason",
            "delegation_instructions": "Focus on topic."
        })
        
        mock_think_response = MagicMock()
        mock_think_response.text = "Thinking output"
        
        mock_vote_response = MagicMock()
        mock_vote_response.text = "Consensus statement"
        
        self.orchestrator.model = AsyncMock()
        # Mock returns for allocation calls (1 call), then thinking (several times), then vote
        self.orchestrator.model.generate_content_async.side_effect = [
            mock_alloc_response, # allocation
            mock_think_response, # Agent_A think step 1
            mock_think_response, # Agent_B think step 1
            mock_think_response, # Agent_A think step 2
            mock_think_response, # Agent_B think step 2
            mock_vote_response,  # consortium vote
        ]
        
        initial_state = SwarmSnapshot(
            iteration=0,
            max_iterations=2,
            universal_prompt="Solve goal",
            agent_configs={},
            agent_states={},
            influence_matrix={}
        )
        
        # Run 1 phase with 2 steps
        history, export_path = await self.orchestrator.run_dynamic_simulation(
            initial_state,
            auto_allocate_iterations=1,
            steps_per_phase=2
        )
        
        # Verify history length
        # 1 initial snapshot + 2 simulation steps
        self.assertEqual(len(history), 3)
        self.assertEqual(export_path, "mock_export_path.json")
        
        # Verify final snapshot properties
        final_snap = history[-1]
        self.assertIn(str(uuid_a), final_snap.agent_states)
        self.assertIn(str(uuid_b), final_snap.agent_states)
        
        # Verify influence matrix override (fully connected)
        for rid, row in final_snap.influence_matrix.items():
            for sid, val in row.items():
                self.assertEqual(val, 1.0)

if __name__ == '__main__':
    unittest.main()
