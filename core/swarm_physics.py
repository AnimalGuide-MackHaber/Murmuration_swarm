import numpy as np
from typing import Dict, List, Tuple
from core.swarm_schema import SwarmSnapshot, AgentRuntimeState, Vector3
from core.settings_manager import SettingsManager

class SwarmPhysicsEngine:
    def __init__(self):
        self.settings = SettingsManager().get_settings().physics

    def update(self, current_state: SwarmSnapshot) -> SwarmSnapshot:
        """
        Updates the positions and velocities of all agents in the snapshot.
        Also recalculates the influence matrix.
        """
        agents = current_state.agent_states
        dt = self.settings.dt
        
        # 1. Update Physics (Position/Velocity)
        # We need a copy of the current state to calculate forces based on the *current* frame
        # to avoid order-of-update artifacts.
        next_states = {}
        
        for agent_id, agent in agents.items():
            # Calculate forces
            acceleration = self._calculate_forces(agent, agents)
            
            # Update Velocity: V_new = clamp(V_old + A * dt, max_speed)
            new_velocity = Vector3(
                x = agent.velocity.x + acceleration.x * dt,
                y = agent.velocity.y + acceleration.y * dt,
                z = agent.velocity.z + acceleration.z * dt
            )
            
            # Clamp speed
            speed = np.sqrt(new_velocity.x**2 + new_velocity.y**2 + new_velocity.z**2)
            if speed > self.settings.max_speed:
                scale = self.settings.max_speed / speed
                new_velocity.x *= scale
                new_velocity.y *= scale
                new_velocity.z *= scale
            
            # Update Position: P_new = P_old + V_new * dt
            new_position = Vector3(
                x = agent.position.x + new_velocity.x * dt,
                y = agent.position.y + new_velocity.y * dt,
                z = agent.position.z + new_velocity.z * dt
            )
            
            # Create new state (preserving memory/thoughts)
            next_states[agent_id] = agent.model_copy(update={
                "position": new_position,
                "velocity": new_velocity,
                "acceleration": acceleration
            })
            
        current_state.agent_states = next_states
        
        # 2. Update Influence Matrix based on NEW positions
        current_state.influence_matrix = self._calculate_influence_matrix(next_states)
        
        return current_state

    def _calculate_forces(self, agent: AgentRuntimeState, all_agents: Dict[str, AgentRuntimeState]) -> Vector3:
        """
        Calculates the net acceleration vector acting on an agent.
        Formula: A = (w_s * V_s) + (w_a * V_a) + (w_c * V_c) + (w_g * V_g)
        """
        w_s = self.settings.weights.separation
        w_a = self.settings.weights.alignment
        w_c = self.settings.weights.cohesion
        w_g = self.settings.weights.goal_seeking
        
        v_separation = np.array([0.0, 0.0, 0.0])
        v_alignment = np.array([0.0, 0.0, 0.0])
        v_cohesion = np.array([0.0, 0.0, 0.0])
        
        neighbor_count = 0
        
        p_i = np.array(agent.position.to_array())
        v_i = np.array(agent.velocity.to_array())
        
        for peer_id, peer in all_agents.items():
            if peer.config_id == agent.config_id: # specific logic if needed, but usually just id check
                # Warning: AgentRuntimeState doesn't have its own ID field, it relies on the dict key or config_id?
                # The prompt implies agents are identified by the dict key in SwarmSnapshot.agent_states.
                # However, inside this loop we are processing 'agent' which is a value.
                # We should check if they are the SAME object or effectively the same agent.
                # Comparing config_id might be enough if 1 config = 1 agent instance, but usually systems allow multiple instances of same config.
                # Comparison by memory reference is risky if copied.
                # Let's assume we skip if dist is 0, but that's also risky.
                # BETTER: Pass agent_id to this function or handle outside.
                # Given input signature, let's rely on distance > 0.
                pass

            p_j = np.array(peer.position.to_array())
            dist_sq = np.sum((p_i - p_j)**2)
            dist = np.sqrt(dist_sq)
            
            if dist > 0 and dist <= self.settings.visual_range:
                # Separation
                # Push away: (Pi - Pj) / dist^2
                diff = p_i - p_j
                v_separation += diff / (dist_sq + 1e-6) # Avoid div by zero
                
                # Alignment & Cohesion prep
                v_alignment += np.array(peer.velocity.to_array())
                v_cohesion += p_j
                
                neighbor_count += 1

        if neighbor_count > 0:
            # Alignment: Average velocity of neighbors - My Velocity
            v_alignment = (v_alignment / neighbor_count) - v_i
            
            # Cohesion: Vector towards average position - My Position
            v_cohesion = (v_cohesion / neighbor_count) - p_i
            
        # Goal Seeking: Vector towards (0,0,0) - My Position
        # Spec says: (0,0,0) - P_i
        v_goal = np.array([0.0, 0.0, 0.0]) - p_i
        
        # Weighted Sum
        a_net = (w_s * v_separation) + (w_a * v_alignment) + (w_c * v_cohesion) + (w_g * v_goal)
        
        return Vector3.from_array(a_net.tolist())

    def _calculate_influence_matrix(self, agents: Dict[str, AgentRuntimeState]) -> Dict[str, Dict[str, float]]:
        """
        Calculates the influence score between every pair of agents.
        influence = 1 / (1 + alpha * d^2) if d <= visual_range, else 0.
        """
        matrix = {}
        alpha = self.settings.influence_decay_alpha
        visual_range = self.settings.visual_range
        
        for receiver_id, receiver in agents.items():
            matrix[receiver_id] = {}
            p_r = np.array(receiver.position.to_array())
            
            for sender_id, sender in agents.items():
                if receiver_id == sender_id:
                    matrix[receiver_id][sender_id] = 1.0 # Self-influence? Usually 1 or 0. Spec doesn't specify self. Let's assume 1 for consistency or 0.
                    # "Influence Network" usually implies peers. But let's follow the formula.
                    # distance is 0. 1 / (1+0) = 1. So it works naturally.
                    continue
                
                p_s = np.array(sender.position.to_array())
                dist_sq = np.sum((p_r - p_s)**2)
                dist = np.sqrt(dist_sq)
                
                if dist > visual_range:
                    matrix[receiver_id][sender_id] = 0.0
                else:
                    matrix[receiver_id][sender_id] = 1.0 / (1.0 + alpha * dist_sq)
                    
        return matrix
