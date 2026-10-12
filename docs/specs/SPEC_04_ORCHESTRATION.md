Specification 04: Orchestration (LangGraph)

1. Overview

The orchestrator manages the cognitive loop. It uses LangGraph to define a cyclic state machine that alternates between Physical Movement and Cognitive Processing.

2. Graph State

The LangGraph State object must match the SwarmSnapshot schema defined in Spec 01.

3. Nodes

Node A: Physics_Update

Action: Instantiates SwarmPhysicsEngine. Calls .update().

Output: Updates agent_states (positions) and influence_matrix. Increments iteration.

Node B: Agent_Think (The "Swarm Brain")

Execution: Parallel execution (using asyncio.gather or ThreadPool) for every agent in agent_states.

Subjective Reality Logic (CRITICAL):

Retrieve History: Get agent.private_memory.

Perceive Peers: Iterate through influence_matrix[agent_id].

For every peer_id where score > 0:

Get peer.latest_thought.

Append to a temporary "Heard Context" block: "[Heard from {peer_id} at dist {d}]: {thought}".

Construct Prompt:

System: AgentConfig.system_prompt_template

Context: private_memory + Heard Context + Universal Prompt.

Task: "Based on your domain and what you just heard, what is your next thought? Use your Knowledge Graph tool if needed."

Invoke LLM: Gemini 1.5 Pro.

Update State: Append result to private_memory and update latest_thought.

Node C: Consortium_Vote (Convergence)

Trigger: When iteration >= max_iterations.

Action:

Gather latest_thought from ALL agents (ignore physics/distance).

Broadcast all thoughts to all agents.

Prompt: "Review these final positions. Rank the top 3 solutions and synthesize a final consensus."

Aggregation: Orchestrator aggregates votes/syntheses into final_result.

4. Edges

Start -> Physics_Update

Physics_Update -> Agent_Think

Agent_Think -> (Conditional: if iter < max) -> Physics_Update

Agent_Think -> (Conditional: if iter >= max) -> Consortium_Vote

Consortium_Vote -> End