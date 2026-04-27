import asyncio
import os
import time
from datetime import datetime
from typing import Dict, List, Any
# import google.generativeai as genai  <-- REMOVED
# from swarm_ollama import OllamaWrapper # <--- REMOVED direct import to use dynamic selection or keep both
from swarm_ollama import OllamaWrapper
from swarm_nvidia import NvidiaLLM
from swarm_gemini import GeminiLLM
from langgraph.graph import StateGraph, END
from swarm_schema import SwarmSnapshot, AgentRuntimeState, ChatMessage, Vector3
from swarm_physics import SwarmPhysicsEngine
from swarm_io import SwarmIO
import json
from knowledge_graph import KnowledgeArchitect

from settings_manager import SettingsManager
from swarm_logger import SwarmLogger
from rate_limiter import RateLimiter

class SwarmOrchestrator:
    def __init__(self):
        self.physics_engine = SwarmPhysicsEngine()
        self.settings_manager = SettingsManager()
        self.logger = SwarmLogger()
        self.settings = self.settings_manager.get_settings()
        self.architect = KnowledgeArchitect()
        
        # Initialize Rate Limiter (per-model)
        rl_settings = self.settings_manager.get_rate_limiting_for_provider(self.settings.llm_provider)
        self.rate_limiter = RateLimiter(
            requests_per_minute=rl_settings.requests_per_minute,
            tokens_per_minute=rl_settings.tokens_per_minute
        )
        self.request_delay = rl_settings.request_delay_seconds

        # Initialize LLM Provider
        self.model = None
        self._init_provider(self.settings.llm_provider, self.settings.model_name)

        # Circuit Breaker State
        self.failure_count = 0
        self.circuit_open_until = 0

        self.workflow = self._setup_graph()

    def _init_provider(self, provider: str, model_name: str):
        """Initializes the LLM provider by name."""
        provider = provider.lower()
        self.model = None
        
        try:
            if provider == "nvidia":
                self.logger.log("Initializing Nvidia LLM...", "INFO", "SwarmOrchestrator")
                api_key = self.settings_manager.get_nvidia_api_key()
                self.model = NvidiaLLM(
                    model_name=model_name,
                    base_url=self.settings.nvidia_api.base_url,
                    api_key=api_key
                )
                self.model.configure()
                self.logger.log(f"Nvidia LLM initialized with model {model_name}", "INFO", "SwarmOrchestrator")
                
            elif provider == "ollama":
                self.logger.log("Initializing Ollama LLM...", "INFO", "SwarmOrchestrator")
                self.model = OllamaWrapper(model_name=model_name)
                self.logger.log("Ollama Wrapper initialized.", "INFO", "SwarmOrchestrator")
            
            elif provider == "gemini":
                self.logger.log("Initializing Gemini LLM...", "INFO", "SwarmOrchestrator")
                api_key = self.settings_manager.get_api_key()
                self.model = GeminiLLM(model_name=model_name, api_key=api_key)
                self.model.configure()
                self.logger.log(f"Gemini LLM initialized with model {model_name}", "INFO", "SwarmOrchestrator")
            
            else:
                self.logger.log(f"Unknown LLM provider: {provider}", "ERROR", "SwarmOrchestrator")

        except Exception as e:
            self.logger.log(f"Failed to initialize LLM provider ({provider}): {e}", "ERROR", "SwarmOrchestrator")
            self.model = None

    def switch_provider(self, provider: str, model_name: str):
        """Switches the LLM provider and model at runtime, updating rate limits."""
        self.logger.log(f"Switching provider to {provider} with model {model_name}", "INFO", "SwarmOrchestrator")
        
        # Update rate limiter for new provider
        rl_settings = self.settings_manager.get_rate_limiting_for_provider(provider)
        self.rate_limiter = RateLimiter(
            requests_per_minute=rl_settings.requests_per_minute,
            tokens_per_minute=rl_settings.tokens_per_minute
        )
        self.request_delay = rl_settings.request_delay_seconds
        
        # Reset circuit breaker
        self.failure_count = 0
        self.circuit_open_until = 0
        
        # Initialize the new provider
        self._init_provider(provider, model_name)

    def _setup_graph(self) -> StateGraph:
        """Defines the LangGraph state machine."""
        workflow = StateGraph(SwarmSnapshot)

        # Add Nodes
        workflow.add_node("physics_update", self.node_physics_update)
        workflow.add_node("agent_think", self.node_agent_think)
        workflow.add_node("consortium_vote", self.node_consortium_vote)

        # Set Entry Point
        workflow.set_entry_point("physics_update")

        # Add Edges
        workflow.add_edge("physics_update", "agent_think")
        
        # Conditional Edge from Agent Think
        workflow.add_conditional_edges(
            "agent_think",
            self._check_termination,
            {
                "continue": "physics_update",
                "vote": "consortium_vote"
            }
        )

        workflow.add_edge("consortium_vote", END)

        return workflow.compile()

    def _check_termination(self, state: SwarmSnapshot) -> str:
        """Decides whether to loop or end based on iterations."""
        if state.iteration < state.max_iterations:
            return "continue"
        return "vote"

    def node_physics_update(self, state: SwarmSnapshot) -> Dict[str, Any]:
        """Runs the physics engine to update positions and influence."""
        self.logger.log(f"Physics Update - Iteration {state.iteration}", "INFO", "Orchestrator")
        
        # Update positions & influence
        new_state = self.physics_engine.update(state)
        
        new_state.iteration += 1
        return new_state.model_dump()

    async def node_agent_think(self, state: SwarmSnapshot) -> Dict[str, Any]:
        """Orchestrates the parallel thinking process of all agents."""
        self.logger.log(f"Agents Thinking - Iteration {state.iteration}", "INFO", "Orchestrator")
        
        agents = state.agent_states
        influence = state.influence_matrix
        
        # Prepare tasks for parallel execution
        # Prepare tasks for sequential execution
        new_agent_states = state.agent_states.copy()
        agent_ids = list(agents.keys())
        
        for i, agent_id in enumerate(agent_ids):
            self.logger.log(f"Agent {i+1}/{len(agent_ids)} ({agent_id}) thinking...", "DEBUG", "Orchestrator")
            _, updated_agent_state = await self._single_agent_think(agent_id, state)
            new_agent_states[agent_id] = updated_agent_state
            
        # Return update
        return {"agent_states": new_agent_states}

    async def _single_agent_think(self, agent_id: str, state: SwarmSnapshot) -> Any:
        """The cognitive logic for a single agent with ReAct capability."""
        agent = state.agent_states[agent_id]
        
        # Guard: If config missing (e.g. valid ID in state but not in config), handle gracefully
        if agent.config_id not in state.agent_configs:
            self.logger.log(f"Missing config for agent {agent_id}", "ERROR", "Orchestrator")
            return (agent_id, agent) # Return unchanged
            
        config = state.agent_configs[agent.config_id]
        
        # 1. Perceive Peers
        heard_context = []
        if agent_id in state.influence_matrix:
            for peer_id, score in state.influence_matrix[agent_id].items():
                if score > 0.050 and peer_id != agent_id:
                    if peer_id in state.agent_states:
                        peer = state.agent_states[peer_id]
                        if peer.latest_thought:
                            dist = agent.position.distance_to(peer.position)
                            heard_context.append(f"[Heard from {peer_id} at dist {dist:.2f}]: {peer.latest_thought}")
        
        # 2. Construct Base Prompt
        system_prompt = config.system_prompt_template
        
        # Context Retrieval
        kg_context = ""
        if config.knowledge_graph_path:
            # Formulate query from state and history to grab best semantic match
            recent_thought = heard_context[-1] if heard_context else ""
            query_sources = [state.universal_prompt, config.domain, recent_thought]
            search_query = " ".join([q for q in query_sources if q.strip()])
            
            if search_query:
                retrieved = self.architect.retrieve_context(config.knowledge_graph_path, search_query)
                kg_context = f"\n--- RELEVANT KNOWLEDGE ---\n{retrieved}\n"
        
        history_str = "\n".join([f"[{m.type}] {m.content}" for m in agent.private_memory[-5:]]) 
        heard_str = "\n".join(heard_context)
        
        base_prompt = (
            f"System: {system_prompt}\n"
            f"Global Goal: {state.universal_prompt}\n"
            f"My Domain: {config.domain}\n"
            f"My Interaction Style: {config.interaction_style}\n"
            f"{kg_context}\n"
            f"--- INTERNAL HISTORY ---\n{history_str}\n\n"
            f"--- HEARD FROM PEERS ---\n{heard_str}\n\n"
            f"Task: Based on your domain, the relevant knowledge, and what you specifically heard (if anything), what is your next unique thought or contribution? "
            f"Be concise. If you heard nothing, generate an initial idea."
        )

        
        # 3. ReAct Loop (Think -> Act -> Observe -> Think)
        max_steps = 3
        current_step = 0
        final_thought = "(Thinking...)"
        conversation_history = [base_prompt] # Keep track of turn
        
        while current_step < max_steps:
            current_step += 1
            
            # Check Circuit Breaker
            if time.time() < self.circuit_open_until:
                 final_thought = "(Thinking paused: Circuit Breaker Open)"
                 break
            
            try:
                if self.model:
                    # Construct full prompt from history so far
                    # Provide last few turns if we are in a loop
                    if current_step > 1:
                        full_input = "\n".join(conversation_history)
                    else:
                        full_input = base_prompt
                        
                    estimated_tokens = len(full_input) // 4 
                    # Retry Logic with Exponential Backoff
                    max_retries = 5
                    base_delay = 5
                    
                    for attempt in range(max_retries + 1):
                        await self.rate_limiter.acquire(estimated_tokens)
                        
                        try:
                            response = await self.model.generate_content_async(full_input)
                        except Exception as e:
                            # Log exception but allow retry logic if it's considered empty/failure? 
                            # Or raise if it's an API error (e.g. 500)
                            # Here we focus on "empty" responses which usually don't raise exceptions but return empty text
                            # However, sometimes network quirks happen.
                            # For safety, let's catch distinct errors separately if needed, 
                            # but the user requested specifically "no text response".
                            # Re-raising if it fails completely is standard unless we want to retry on exceptions too.
                            # Let's stick to the user's specific request: "no text response".
                            raise e 

                        if response.text and response.text.strip():
                            thought_content = response.text.strip()
                            break
                        
                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)
                            self.logger.log(f"Agent {agent_id} received empty response. Retrying in {delay}s (Attempt {attempt+1}/{max_retries})", "WARNING", "Orchestrator")
                            await asyncio.sleep(delay)
                        else:
                             self.logger.log(f"Agent {agent_id} failed to get response after {max_retries} retries.", "ERROR", "Orchestrator")
                             thought_content = "(No response from model)"
                    
                    # Update usage 
                    # If response.text was None, len is 0. 
                    response_tokens = len(thought_content) // 4
                    await self.rate_limiter.update_usage(estimated_tokens + response_tokens)
                    
                    # Reset failure count on success
                    self.failure_count = 0 
                    
                    # Tool Extension Check removed - we now do automatic context injection before LLM call
                    
                    # If no tool call, this is the final thought
                    final_thought = thought_content
                    break
                    
                else:
                    final_thought = "(Auto-thought: Model not connected)"
                    break
                    
            except Exception as e:
                self.failure_count += 1
                error_msg = str(e)
                
                if "Rate Limit" in error_msg:
                     final_thought = "(Thinking paused: Rate Limit Reached)"
                     self.logger.log(f"Agent {agent_id} rate limited.", "WARNING", "Orchestrator")
                else:
                     final_thought = "(Thinking failed due to error)"
                     self.logger.log(f"Agent {agent_id} error: {e}", "ERROR", "Orchestrator")
                
                cb_settings = self.settings.circuit_breaker
                max_failures = cb_settings.max_failures if cb_settings else 5
                open_duration = cb_settings.open_duration_seconds if cb_settings else 600

                if self.failure_count >= max_failures:
                    self.circuit_open_until = time.time() + open_duration
                    self.logger.log(f"Circuit Breaker ACTIVATED: Pausing LLM calls for {open_duration}s.", "ERROR", "Orchestrator")
                break

        
        self.logger.log(f"Agent {agent_id} final thought: {final_thought}...", "DEBUG", "Orchestrator")
            
        # 4. Update State
        new_message = ChatMessage(
            source_agent_id=config.name, 
            content=final_thought,
            type='internal_thought'
        )
        
        updated_agent = agent.model_copy()
        updated_agent.private_memory.append(new_message)
        updated_agent.latest_thought = final_thought
        
        return (agent_id, updated_agent)

    async def node_consortium_vote(self, state: SwarmSnapshot) -> Dict[str, Any]:
        """Final aggregation and consensus optimization."""
        self.logger.log("Consortium Vote Started", "INFO", "Orchestrator")
        
        # Gather all latest thoughts
        final_thoughts = []
        agent_contributions = []
        for agent_id, agent in state.agent_states.items():
            config = state.agent_configs.get(agent.config_id)
            agent_name = config.name if config else agent.config_id
            final_thoughts.append(f"{agent.config_id}: {agent.latest_thought}")
            agent_contributions.append({
                "agent_id": str(agent_id),
                "agent_name": agent_name,
                "final_position": agent.latest_thought
            })
            
        joined_thoughts = "\n\n".join(final_thoughts)
        
        prompt = (
            f"Review these final positions from the swarm experts:\n{joined_thoughts}\n\n"
            f"Rank the top 3 solutions and synthesize a final consensus statement."
        )
        
        try:
            if self.model and time.time() >= self.circuit_open_until:
                estimated_tokens = len(prompt) // 4
                await self.rate_limiter.acquire(estimated_tokens)
                
                response = await self.model.generate_content_async(prompt)
                consensus = response.text
                
                response_tokens = len(consensus) // 4
                await self.rate_limiter.update_usage(estimated_tokens + response_tokens)
            else:
                consensus = "Consensus impossible (No Model or Circuit Open)."
        except Exception as e:
            consensus = f"Error voting: {e}"
            
        self.logger.log(f"Consensus Reached: {consensus[:100]}...", "SUCCESS", "Orchestrator")
        print(f"=== CONSENSUS REACHED ===\n{consensus}\n=========================")
        
        return {
            "consensus_text": consensus,
            "agent_contributions": agent_contributions
        }

    async def node_expert_enrichment(self, state: SwarmSnapshot, consensus: str) -> Dict[str, Any]:
        """Each agent reviews the consensus and enriches it with domain-specific technical detail."""
        self.logger.log("Expert Enrichment Pass Started", "INFO", "Orchestrator")
        
        enriched_document = consensus
        enrichment_log = []
        agent_ids = list(state.agent_states.keys())
        
        for i, agent_id in enumerate(agent_ids):
            agent = state.agent_states[agent_id]
            
            if agent.config_id not in state.agent_configs:
                continue
                
            config = state.agent_configs[agent.config_id]
            
            self.logger.log(
                f"Expert Enrichment {i+1}/{len(agent_ids)} - {config.name} ({config.domain}) reviewing...",
                "DEBUG", "Orchestrator"
            )
            
            # Build the enrichment prompt
            prompt = (
                f"You are {config.name}, an expert in {config.domain}.\n"
                f"Interaction Style: {config.interaction_style}\n\n"
                f"The following consensus document was produced by a multi-agent swarm discussion "
                f"on the topic: \"{state.universal_prompt}\"\n\n"
                f"--- CURRENT DOCUMENT ---\n{enriched_document}\n--- END DOCUMENT ---\n\n"
                f"Your task: Review this document and produce an UPDATED version that integrates "
                f"specific technical details, references, methodologies, or insights from your "
                f"domain of expertise ({config.domain}) wherever they are relevant. "
                f"Preserve the existing structure and content — add to it, refine it, and make it "
                f"more technically rigorous from your perspective. "
                f"Do NOT remove other experts' contributions. "
                f"Output ONLY the updated document, no preamble."
            )
            
            try:
                if self.model and time.time() >= self.circuit_open_until:
                    estimated_tokens = len(prompt) // 4
                    await self.rate_limiter.acquire(estimated_tokens)
                    
                    response = await self.model.generate_content_async(prompt)
                    enriched_text = response.text.strip() if response.text else ""
                    
                    response_tokens = len(enriched_text) // 4
                    await self.rate_limiter.update_usage(estimated_tokens + response_tokens)
                    
                    if enriched_text:
                        enriched_document = enriched_text
                        enrichment_log.append({
                            "agent_name": config.name,
                            "domain": config.domain,
                            "status": "enriched"
                        })
                        self.logger.log(f"{config.name} enriched the document.", "SUCCESS", "Orchestrator")
                    else:
                        enrichment_log.append({
                            "agent_name": config.name,
                            "domain": config.domain,
                            "status": "no_changes"
                        })
                else:
                    enrichment_log.append({
                        "agent_name": config.name,
                        "domain": config.domain,
                        "status": "skipped (no model or circuit open)"
                    })
            except Exception as e:
                self.logger.log(f"Enrichment error from {config.name}: {e}", "ERROR", "Orchestrator")
                enrichment_log.append({
                    "agent_name": config.name,
                    "domain": config.domain,
                    "status": f"error: {e}"
                })
        
        self.logger.log("Expert Enrichment Pass Complete", "SUCCESS", "Orchestrator")
        
        return {
            "enriched_conclusion": enriched_document,
            "enrichment_log": enrichment_log
        }

    async def synthesize_paper(self, history: List[SwarmSnapshot], topic: str) -> str:
        """
        Synthesizes a research paper from the simulation history.
        """
        self.logger.log("Synthesizing Research Paper...", "INFO", "Orchestrator")
        
        # Construct Context
        timeline = []
        for snap in history:
            iteration_thoughts = []
            for aid, agent in snap.agent_states.items():
                if agent.latest_thought:
                    name = snap.agent_configs[agent.config_id].name
                    iteration_thoughts.append(f"{name}: {agent.latest_thought}")
            
            if iteration_thoughts:
                timeline.append(f"### Iteration {snap.iteration}\n" + "\n".join(iteration_thoughts))
                
        full_context = "\n\n".join(timeline)
        
        system_prompt = (
            "You are a Lead Research Scientist. Your task is to write a comprehensive research paper "
            "based on the following multi-agent simulation logs. "
            "The agents have debated and collaborated on a specific topic. "
            "Your paper must include:\n"
            "1. Abstract\n"
            "2. Introduction (Problem Statement)\n"
            "3. Methodology (Agent Perspectives & Interaction Dynamics)\n"
            "4. Results (Key Insights & Arguments Emerging from the Swarm)\n"
            "5. Conclusion & Consensus\n"
            "6. References (Cite specific agents as contributors)\n\n"
            "Format the output as clean Markdown."
        )
        
        user_prompt = f"Topic: {topic}\n\nSimulation Logs:\n{full_context}"
        
        try:
             if self.model and time.time() >= self.circuit_open_until:
                full_input = system_prompt + user_prompt
                estimated_tokens = len(full_input) // 4
                await self.rate_limiter.acquire(estimated_tokens)

                response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
                paper = response.text
                
                response_tokens = len(paper) // 4
                await self.rate_limiter.update_usage(estimated_tokens + response_tokens)
             else:
                paper = "# Research Paper\n(Error: No Model Connected or Circuit Open)"
        except Exception as e:
            self.logger.log(f"Error generating paper: {e}", "ERROR", "Orchestrator")
            paper = f"# Error Generating Paper\n\n{e}"
            
        self.logger.log("Research Paper Generated.", "SUCCESS", "Orchestrator")
        return paper 

    async def run_full_simulation(self, initial_state: SwarmSnapshot):
        """Runs the entire simulation loop natively inside the orchestrator and exports live state to JSON."""
        self.logger.log("Starting Swarm Simulation Loop...", "INFO", "Orchestrator")
        
        history = [initial_state]
        current_state = initial_state.model_copy(deep=True)
        iterations = initial_state.max_iterations
        
        SwarmIO.save_live_state(initial_state.simulation_id, {
            "status": "running",
            "iteration": 0,
            "max_iterations": iterations,
            "message": "Initializing Simulation...",
            "latest_snapshot": current_state.model_dump(mode="json")
        })
        
        for i in range(iterations):
            SwarmIO.save_live_state(initial_state.simulation_id, {
                "status": "running",
                "iteration": i,
                "max_iterations": iterations,
                "message": f"Iteration {i+1}/{iterations} - Running Physics...",
                "latest_snapshot": current_state.model_dump(mode="json")
            })
                
            current_state = self.physics_engine.update(current_state)
            current_state.iteration = i + 1
            
            SwarmIO.save_live_state(initial_state.simulation_id, {
                "status": "running",
                "iteration": i + 1,
                "max_iterations": iterations,
                "message": f"Iteration {i+1}/{iterations} - Agents Thinking...",
                "latest_snapshot": current_state.model_dump(mode="json")
            })
                
            think_update = await self.node_agent_think(current_state)
            if "agent_states" in think_update:
                current_state.agent_states = think_update["agent_states"]
                
            history.append(current_state.model_copy(deep=True))
            
            SwarmIO.save_live_state(initial_state.simulation_id, {
                "status": "running",
                "iteration": i + 1,
                "max_iterations": iterations,
                "message": f"Iteration {i+1}/{iterations} - Complete",
                "latest_snapshot": current_state.model_dump(mode="json")
            })

        SwarmIO.save_live_state(initial_state.simulation_id, {
            "status": "voting",
            "iteration": iterations,
            "max_iterations": iterations,
            "message": "Conducting Consortium Vote...",
            "latest_snapshot": current_state.model_dump(mode="json")
        })
        vote_result = await self.node_consortium_vote(current_state)
        
        # --- Expert Enrichment Pass ---
        raw_consensus = vote_result.get("consensus_text", "")
        enrichment_result = {"enriched_conclusion": raw_consensus, "enrichment_log": []}
        
        if raw_consensus and self.model and time.time() >= self.circuit_open_until:
            SwarmIO.save_live_state(initial_state.simulation_id, {
                "status": "enriching",
                "iteration": iterations,
                "max_iterations": iterations,
                "message": "Expert Enrichment — Agents adding domain expertise...",
                "latest_snapshot": current_state.model_dump(mode="json")
            })
            enrichment_result = await self.node_expert_enrichment(current_state, raw_consensus)
        
        SwarmIO.save_live_state(initial_state.simulation_id, {
            "status": "saving",
            "iteration": iterations,
            "max_iterations": iterations,
            "message": "Saving Simulation Data...",
            "latest_snapshot": current_state.model_dump(mode="json")
        })
            
        export_data = {
            "simulation_id": initial_state.simulation_id,
            "timestamp": str(time.time()),
            "steps": []
        }
        
        for snap in history:
            step_data = {"iteration": snap.iteration, "agents": []}
            for aid, agent in snap.agent_states.items():
                cfg = snap.agent_configs[agent.config_id]
                step_data["agents"].append({
                    "id": str(cfg.id),
                    "name": cfg.name,
                    "position": agent.position.model_dump(),
                    "velocity": agent.velocity.model_dump(),
                    "thought": agent.latest_thought
                })
            export_data["steps"].append(step_data)
        
        # --- Consensus & Agent Conversation History ---
        # Build per-agent conversation logs from the final state's private_memory
        agent_conversations = []
        for aid, agent in current_state.agent_states.items():
            cfg = current_state.agent_configs.get(agent.config_id)
            agent_name = cfg.name if cfg else agent.config_id
            messages = []
            for msg in agent.private_memory:
                messages.append({
                    "source": msg.source_agent_id,
                    "type": msg.type,
                    "content": msg.content,
                    "timestamp": str(msg.timestamp)
                })
            agent_conversations.append({
                "agent_id": str(aid),
                "agent_name": agent_name,
                "conversation": messages
            })
        
        export_data["consensus"] = {
            "raw_conclusion": raw_consensus,
            "enriched_conclusion": enrichment_result.get("enriched_conclusion", raw_consensus),
            "enrichment_log": enrichment_result.get("enrichment_log", []),
            "agent_final_positions": vote_result.get("agent_contributions", []),
            "agent_conversations": agent_conversations
        }
            
        export_path = SwarmIO.save_simulation_export(export_data, initial_state.simulation_id)
        
        # Save a standalone markdown conclusion file with timestamp (uses enriched version)
        enriched_conclusion = enrichment_result.get("enriched_conclusion", raw_consensus)
        if enriched_conclusion:
            timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            conclusion_md = f"# Swarm Consensus — {timestamp_str}\n\n"
            conclusion_md += f"**Simulation ID:** `{initial_state.simulation_id}`\n\n"
            conclusion_md += f"**Prompt:** {initial_state.universal_prompt}\n\n"
            conclusion_md += f"---\n\n## Conclusion (Expert-Enriched)\n\n{enriched_conclusion}\n"
            SwarmIO.save_conclusion_markdown(conclusion_md, timestamp_str)
        
        SwarmIO.save_live_state(initial_state.simulation_id, {
            "status": "complete",
            "iteration": iterations,
            "max_iterations": iterations,
            "message": "Simulation Complete!",
            "export_path": export_path,
            "history": [s.model_dump(mode="json") for s in history]
        })
            
        return history, export_path

    async def generate_collaborative_kg(self, history: List[SwarmSnapshot], conclusion: str, domain: str, progress_callback=None) -> str:
        """
        Coordinates the agents to build a domain-specific Knowledge Graph from the consensus.
        1. base_kg = self.architect.generate_base_kg_from_text(conclusion)
        2. agents enrich the base_kg sequentially.
        3. final_kg = self.architect.add_linking_edges(enriched_kg)
        """
        self.logger.log(f"Starting Collaborative KG Generation for domain: {domain}", "INFO", "Orchestrator")
        
        # Step 1: Base Graph Generation
        if progress_callback: progress_callback("Generating base graph structure...", 0.1)
        kg_data = await self.architect.generate_base_kg_from_text(conclusion)
        
        if not kg_data.get("nodes"):
            self.logger.log("Base KG generation resulted in an empty graph.", "WARNING", "Orchestrator")
            # We'll continue anyway, so agents can build it from scratch if needed

        # Step 2: Agent Enrichment
        if not history:
             return "Error: No simulation history provided."

        final_snapshot = history[-1]
        agent_ids = list(final_snapshot.agent_states.keys())
        total_steps = len(agent_ids) + 2  # base + agents + final linking

        for i, agent_id in enumerate(agent_ids):
            agent = final_snapshot.agent_states[agent_id]
            if agent.config_id not in final_snapshot.agent_configs:
                continue
                
            config = final_snapshot.agent_configs[agent.config_id]
            
            if progress_callback: 
                progress = 0.1 + (0.7 * (i / len(agent_ids)))
                progress_callback(f"Agent {config.name} ({config.domain}) adding expertise to KG...", progress)

            self.logger.log(f"Agent {config.name} enriching KG...", "INFO", "Orchestrator")
            
            prompt = (
                f"You are {config.name}, an expert in {config.domain}.\n"
                f"The following JSON represents a Knowledge Graph being collaboratively built from "
                f"a recent Swarm Consensus. Review the 'nodes' and add new structurally sound nodes "
                f"or expand existing ones with specific technical details, mechanisms, or methodologies "
                f"from your domain ({config.domain}).\n"
                f"Maintain the exact JSON structure:\n"
                f"```json\n"
                f"{{\n"
                f"  'nodes': {{\n"
                f"    'node_id': {{\n"
                f"      'label': 'String',\n"
                f"      'summary': 'String',\n"
                f"      'type': 'concept|fact',\n"
                f"      'children': ['child_id_1', 'child_id_2']\n"
                f"    }}\n"
                f"  }}\n"
                f"}}\n"
                f"```\n"
                f"Do NOT remove other agents' nodes. Provide ONLY the updated JSON string.\n\n"
                f"--- CURRENT GRAPH ---\n{json.dumps(kg_data, indent=2)}"
            )

            try:
                if self.model and time.time() >= self.circuit_open_until:
                    estimated_tokens = len(prompt) // 4
                    await self.rate_limiter.acquire(estimated_tokens)
                    
                    response = await self.model.generate_content_async(prompt)
                    json_str = response.text.strip() if response.text else ""
                    
                    response_tokens = len(json_str) // 4
                    await self.rate_limiter.update_usage(estimated_tokens + response_tokens)
                    
                    if json_str:
                        if json_str.startswith("```json"): json_str = json_str[7:]
                        if json_str.endswith("```"): json_str = json_str[:-3]
                        json_str = json_str.strip()
                        
                        updated_data = json.loads(json_str)
                        if "nodes" in updated_data:
                            self.architect._merge_graphs(kg_data, updated_data)
                        self.logger.log(f"{config.name} successfully updated KG.", "SUCCESS", "Orchestrator")
                else:
                    self.logger.log(f"Skipping {config.name} (model/circuit issue).", "WARNING", "Orchestrator")
            except Exception as e:
                self.logger.log(f"Error from {config.name} during KG enrichment: {e}", "ERROR", "Orchestrator")

        # Step 3: Final Linking Pass
        if progress_callback: progress_callback("Finalizing graph links & relationships...", 0.85)
        self.logger.log("Final Linking Pass...", "INFO", "Orchestrator")
        kg_data = await self.architect.add_linking_edges(kg_data)

        # Step 4: Save Graph
        if progress_callback: progress_callback("Saving Collaborative Knowledge Graph...", 0.95)
        
        safe_domain = "".join([c for c in domain if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "_")
        if not safe_domain: safe_domain = "Collaborative_KG"
        
        filename = f"{safe_domain}.json"
        
        # To avoid overwriting unconditionally, let's append timestamp if file exists
        filepath = os.path.join(SwarmIO.KG_DIR, filename)
        if os.path.exists(filepath):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_domain}_{ts}.json"
            filepath = os.path.join(SwarmIO.KG_DIR, filename)

        try:
            with open(filepath, 'w') as f:
                json.dump(kg_data, f, indent=2)
            self.logger.log(f"Collaborative KG saved to {filename}", "SUCCESS", "Orchestrator")
            if progress_callback: progress_callback("Complete!", 1.0)
            return filename
        except Exception as e:
            self.logger.log(f"Error saving Collaborative KG: {e}", "ERROR", "Orchestrator")
            if progress_callback: progress_callback("Error saving graph.", 1.0)
            return f"Error: {e}"
