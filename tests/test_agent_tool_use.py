import unittest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch
import uuid
import sys
import os

# Mock dependencies
sys.modules["langgraph"] = MagicMock()
sys.modules["langgraph.graph"] = MagicMock()
sys.modules["numpy"] = MagicMock()
sys.modules["markitdown"] = MagicMock()
sys.modules["aiohttp"] = MagicMock()
sys.modules["openai"] = MagicMock()

# Mock numpy sqrt specifically
sys.modules["numpy"].sqrt = lambda x: x**0.5

# Add parent directory to path so we can import swarm modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from swarm_orchestrator import SwarmOrchestrator
from swarm_schema import SwarmSnapshot, AgentConfig, AgentRuntimeState, Vector3
from knowledge_graph import KnowledgeArchitect 

# We need a real KnowledgeArchitect for this test to test the search logic
# But we need to mock file I/O

class TestAgentToolUse(unittest.IsolatedAsyncioTestCase):
    async def test_agent_tool_search(self):
        # 1. Setup Mock Orchestrator
        with patch('swarm_orchestrator.SettingsManager'), \
             patch('swarm_orchestrator.SwarmLogger'), \
             patch('swarm_orchestrator.RateLimiter') as MockRateLimiter, \
             patch('swarm_orchestrator.OllamaWrapper'): 
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
            # Call 1: Agent searches for "gravity"
            # Call 2: Agent gives final thought
            response1 = MagicMock()
            response1.text = "I need info. [[QUERY: gravity]]"
            response1.usage_metadata = {}
            
            response2 = MagicMock()
            response2.text = "Gravity is a fundamental force."
            response2.usage_metadata = {}
            
            mock_llm.generate_content_async = AsyncMock(side_effect=[response1, response2])
            
            # 2. Setup State
            agent_uuid = uuid.uuid4()
            agent_id = str(agent_uuid)
            
            config = AgentConfig(
                id=agent_uuid,
                name="TestAgent",
                domain="Testing",
                knowledge_graph_path="test_graph.json",
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
                universal_prompt="Test Goal",
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
                        "summary": "A force of attraction.",
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
                # Verify prompt of second call contains the search result
                call_args_list = mock_llm.generate_content_async.call_args_list
                second_call_prompt = call_args_list[1][0][0]
                
                # We expect "node_2" (Gravity) to be found
                self.assertIn("search_results", second_call_prompt)
                self.assertIn("node_2", second_call_prompt)
                self.assertIn("Gravity", second_call_prompt)
                
                # Verify Final Thought
                self.assertEqual(result_agent.latest_thought, "Gravity is a fundamental force.")
                print("Test passed: Keyword search verified!")

if __name__ == "__main__":
    unittest.main()
