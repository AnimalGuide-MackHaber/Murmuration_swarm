Specification 03: Swarm Physics Engine

1. Objective

Simulate the spatial movement of agents to determine their "Influence Network." The engine runs one "Tick" for every cognitive step of the AI.

2. The Boid Algorithm

The engine must update the velocity and position of every AgentRuntimeState based on the following weighted vectors:

Separation ($V_s$): $\sum_{j \neq i} \frac{P_i - P_j}{|P_i - P_j|^2}$ (Push away from neighbors if distance < visual_range).

Alignment ($V_a$): Average velocity of neighbors ($V_{avg} - V_i$).

Cohesion ($V_c$): Vector towards average position of neighbors ($P_{avg} - P_i$).

Goal Seeking ($V_g$): Vector towards the global prompt origin $(0,0,0) - P_i$.

Update Formula:


$$A_{new} = (w_s \cdot V_s) + (w_a \cdot V_a) + (w_c \cdot V_c) + (w_g \cdot V_g)$$

$$V_{new} = \text{clamp}(V_{old} + A_{new} \cdot dt, \text{max\_speed})$$

$$P_{new} = P_{old} + V_{new} \cdot dt$$

3. Influence Matrix Calculation

Input: All Agent Positions.

Output: A square matrix (Dictionary of Dictionaries) {Receiver_ID: {Sender_ID: Influence_Score}}.

Logic:

Calculate Euclidean distance $d$ between Agent A and B.

If $d > \text{visual\_range}$, Influence = 0.

If $d \le \text{visual\_range}$, Influence = $\frac{1}{1 + \alpha \cdot d^2}$.

Where $\alpha$ (alpha) is the influence_decay parameter (controls how sharp the drop-off is).

4. Integration

The Physics Engine does not handle LLM calls.

It accepts a list of AgentRuntimeState.

It mutates the position and velocity fields in place.

It returns the InfluenceMatrix for the Orchestrator to use.