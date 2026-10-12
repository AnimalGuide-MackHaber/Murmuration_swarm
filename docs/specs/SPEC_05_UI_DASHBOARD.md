Specification 05: UI & Visualization

1. Tech Stack

Framework: Streamlit.

Visualization: Plotly Graph Objects (3D Scatter).

2. Tab 1: Agent Forge

Inputs:

Name (Text).

Domain (Text).

PDF Upload (FileUploader).

Personality (Selectbox).

Action:

On Submit, trigger KnowledgeArchitect to parse PDF.

Create AgentConfig JSON.

Save to /swarm_data/agents.

3. Tab 2: Swarm Simulation

Configuration:

Multi-select: Choose Agents from /swarm_data/agents.

Text Area: Universal Prompt.

Slider: Iterations ($n$).

Execution:

Button: "Initialize Swarm".

Logic: Instantiate SwarmOrchestrator. Run graph.

Persistence: Store every step's SwarmSnapshot in st.session_state.

Visualization (The "Player"):

Slider: "Time Step" (0 to $n$).

3D Plot:

Trace 1 (Agents): Scatter3D. X/Y/Z from snapshot.agent_states. Color by Domain.

Trace 2 (Goal): Fixed point at (0,0,0). Gold Star marker.

Trace 3 (Influence): Line3D. Iterate snapshot.influence_matrix. Draw line between A and B if score > threshold. Opacity = score.

Metrics: Display "Current Consensus" or "Active Thoughts" in a side panel for the selected time step.

4. Sidebar: Physics Tuner

Widgets: Sliders for Separation, Alignment, Cohesion, Visual Range.

Action: These update the GlobalSettings singleton, which the Physics_Update node reads in real-time.