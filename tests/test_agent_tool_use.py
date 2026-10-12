import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import uuid
import sys
import os

# Mock dependencies
sys.modules["langgraph"] = MagicMock()
sys.modules["langgraph.graph"] = MagicMock()
sys.modules["markitdown"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["openai"] = MagicMock()

# Add parent directory to path so we can import swarm modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.swarm_orchestrator import SwarmOrchestrator
from core.swarm_schema import SwarmSnapshot, AgentConfig, AgentRuntimeState, Vector3
from core.knowledge_graph import KnowledgeArchitect 

# We need a real KnowledgeArchitect for this test to test the search logic
# But we need to mock file I/O

class TestAgentToolUse(unittest.IsolatedAsyncioTestCase):
    async def test_agent_tool_search(self):
        # 1. Setup Mock Orchestrator
        with patch('core.swarm_orchestrator.SettingsManager'), \
             patch('core.swarm_orchestrator.SwarmLogger'), \
             patch('core.swarm_orchestrator.RateLimiter') as MockRateLimiter, \
             patch('core.swarm_orchestrator.OllamaWrapper'): 
             # NOT patching KnowledgeArchitect, using real one but mocking open/json
            
            # Configure Rate Limiter Mock
            mock_rate_limiter = MockRateLimiter.return_value
            mock_rate_limiter.acquire = AsyncMock()
            mock_rate_limiter.update_usage = AsyncMock()
            
            # Initialize Orchestrator
            orchestrator = SwarmOrchestrator()
            # Inject real architect
            orchestrator.architect = KnowledgeArchitect()
            
            # Manually inject a Mock LLM
            mock_llm = AsyncMock()
            orchestrator.model = mock_llm
            
            # Scenario:
            # Automatic context injection will run before model call
            response = MagicMock()
            response.text = "Gravity is a force of attraction."
            response.usage_metadata = {}
            
            mock_llm.generate_content_async = AsyncMock(return_value=response)
            
            # 2. Setup State
            agent_uuid = uuid.uuid4()
            agent_id = str(agent_uuid)
            
            config = AgentConfig(
                id=agent_uuid,
                name="TestAgent",
                domain="Testing",
                knowledge_graph_paths=["test_graph.json"],
                system_prompt_template="You are a test agent.",
                color="#000000"
            )
            
            agent_state = AgentRuntimeState(
                config_id=agent_id,
                position=Vector3(x=0, y=0, z=0),
                velocity=Vector3(x=0, y=0, z=0),
                acceleration=Vector3(x=0, y=0, z=0)
            )
            
            state = SwarmSnapshot(
                iteration=1,
                max_iterations=5,
                universal_prompt="gravity",
                agent_configs={agent_id: config},
                agent_states={agent_id: agent_state},
                influence_matrix={}
            )
            
            # Mock File I/O for KnowledgeArchitect
            mock_graph_data = {
                "nodes": {
                    "node_1": {
                        "label": "Physics",
                        "summary": "Study of matter and energy.",
                        "children": []
                    },
                    "node_2": {
                        "label": "Gravity",
                        "summary": "Gravity is a force of attraction.",
                         "children": []
                    }
                }
            }
            
            import json
            mock_open = unittest.mock.mock_open(read_data=json.dumps(mock_graph_data))
            
            # Use patch for os.path.exists and open
            with patch('os.path.exists', return_value=True), \
                 patch('builtins.open', mock_open):
                
                # 3. Run Agent Think
                result_id, result_agent = await orchestrator._single_agent_think(agent_id, state)
            
                # 4. Assertions
                # Verify prompt contains the search result (injected context)
                call_args_list = mock_llm.generate_content_async.call_args_list
                first_call_prompt = call_args_list[0][0][0]
                
                # We expect the context from "node_2" (Gravity) to be automatically injected
                self.assertIn("RELEVANT KNOWLEDGE FROM test_graph.json", first_call_prompt)
                self.assertIn("Gravity is a force of attraction.", first_call_prompt)
                
                # Verify Final Thought
                self.assertEqual(result_agent.latest_thought, "Gravity is a force of attraction.")
                print("Test passed: Automatic context injection verified!")

if __name__ == "__main__":
    unittest.main()
