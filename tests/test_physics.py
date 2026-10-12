import unittest
from core.swarm_schema import SwarmSnapshot, AgentRuntimeState, Vector3, AgentConfig
from core.swarm_physics import SwarmPhysicsEngine
from core.settings_manager import SettingsManager
import numpy as np

class TestSwarmPhysics(unittest.TestCase):
    def setUp(self):
        # Reset settings to default for testing
        SettingsManager._instance = None
        self.engine = SwarmPhysicsEngine()
        
    def test_basic_update(self):
        # Setup 2 agents
        agent1 = AgentRuntimeState(
            config_id="c1",
            position=Vector3(x=10, y=0, z=0),
            velocity=Vector3(x=0, y=0, z=0),
            acceleration=Vector3(x=0, y=0, z=0)
        )
        agent2 = AgentRuntimeState(
            config_id="c2",
            position=Vector3(x=-10, y=0, z=0),
            velocity=Vector3(x=0, y=0, z=0),
            acceleration=Vector3(x=0, y=0, z=0)
        )
        
        snapshot = SwarmSnapshot(
            iteration=0,
            max_iterations=10,
            universal_prompt="test",
            agent_configs={},
            agent_states={"a1": agent1, "a2": agent2},
            influence_matrix={}
        )
        
        # Run update
        new_snapshot = self.engine.update(snapshot)
        
        # Check positions changed (Goal seeking should pull them to 0,0,0)
        p1 = new_snapshot.agent_states["a1"].position
        p2 = new_snapshot.agent_states["a2"].position
        
        # Agent 1 should move towards 0 (x < 10)
        self.assertLess(p1.x, 10.0)
        # Agent 2 should move towards 0 (x > -10)
        self.assertGreater(p2.x, -10.0)
        
        # Check Influence Matrix
        # Distance is 20 approx (started at 20, moved slightly closer)
        # Visual range is 25. So they should influence each other.
        inf_matrix = new_snapshot.influence_matrix
        score = inf_matrix["a1"]["a2"]
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0) # Should be decayed
        
    def test_visual_range_cutoff(self):
        # Agents far apart
        agent1 = AgentRuntimeState(
            config_id="c1",
            position=Vector3(x=100, y=0, z=0),
            velocity=Vector3(x=0, y=0, z=0),
            acceleration=Vector3(x=0, y=0, z=0)
        )
        agent2 = AgentRuntimeState(
            config_id="c2",
            position=Vector3(x=-100, y=0, z=0),
            velocity=Vector3(x=0, y=0, z=0),
            acceleration=Vector3(x=0, y=0, z=0)
        )
        
        snapshot = SwarmSnapshot(
            iteration=0,
            max_iterations=10,
            universal_prompt="test",
            agent_configs={},
            agent_states={"a1": agent1, "a2": agent2},
            influence_matrix={}
        )
        
        new_snapshot = self.engine.update(snapshot)
        
        # Influence should be 0
        self.assertEqual(new_snapshot.influence_matrix["a1"]["a2"], 0.0)

if __name__ == '__main__':
    unittest.main()
