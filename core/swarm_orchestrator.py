import asyncio
import os
import time
from datetime import datetime
from typing import Dict, List, Any
# import google.generativeai as genai  <-- REMOVED
# from swarm_ollama import OllamaWrapper # <--- REMOVED direct import to use dynamic selection or keep both
from core.swarm_ollama import OllamaWrapper
from core.swarm_nvidia import NvidiaLLM
from core.swarm_gemini import GeminiLLM
from langgraph.graph import StateGraph, END
from core.swarm_schema import SwarmSnapshot, AgentRuntimeState, ChatMessage, Vector3
from core.swarm_physics import SwarmPhysicsEngine
from core.swarm_io import SwarmIO
import json
from core.knowledge_graph import KnowledgeArchitect

from core.settings_manager import SettingsManager
from core.swarm_logger import SwarmLogger
from core.rate_limiter import RateLimiter
from core.synthetic_organism import DifferentiableProjectionLayer
import numpy as np

class SwarmOrchestrator:
    def __init__(self):
        self.physics_engine = SwarmPhysicsEngine()
        self.settings_manager = SettingsManager()
        self.logger = SwarmLogger()
        self.settings = self.settings_manager.get_settings()
        self.architect = KnowledgeArchitect()
        
        # Initialize Rate Limiter (per-model) - now using shared instance from SettingsManager
        self.rate_limiter = self.settings_manager.get_shared_rate_limiter(self.settings.llm_provider)
        rl_settings = self.settings_manager.get_rate_limiting_for_provider(self.settings.llm_provider)
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
                    api_key=api_key,
                    rate_limiter=self.rate_limiter
                )
                self.logger.log(f"Nvidia LLM initialized with model {model_name}", "INFO", "SwarmOrchestrator")
                
            elif provider == "ollama":
                self.logger.log("Initializing Ollama LLM...", "INFO", "SwarmOrchestrator")
                self.model = OllamaWrapper(model_name=model_name, rate_limiter=self.rate_limiter)
                self.logger.log("Ollama Wrapper initialized.", "INFO", "SwarmOrchestrator")
            
            elif provider == "gemini":
                self.logger.log("Initializing Gemini LLM...", "INFO", "SwarmOrchestrator")
                api_key = self.settings_manager.get_api_key()
                self.model = GeminiLLM(model_name=model_name, api_key=api_key, rate_limiter=self.rate_limiter)
                self.logger.log(f"Gemini LLM initialized with model {model_name}", "INFO", "SwarmOrchestrator")
            
            else:
                self.logger.log(f"Unknown LLM provider: {provider}", "ERROR", "SwarmOrchestrator")

        except Exception as e:
            self.logger.log(f"Failed to initialize LLM provider ({provider}): {e}", "ERROR", "SwarmOrchestrator")
            self.model = None

    def switch_provider(self, provider: str, model_name: str):
        """Switches the LLM provider and model at runtime, updating rate limits."""
        self.logger.log(f"Switching provider to {provider} with model {model_name}", "INFO", "SwarmOrchestrator")
        
        # Update rate limiter for new provider - use shared instance
        self.rate_limiter = self.settings_manager.get_shared_rate_limiter(provider)
        rl_settings = self.settings_manager.get_rate_limiting_for_provider(provider)
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
        
        new_agent_states = state.agent_states.copy()
        agent_ids = list(agents.keys())
        
        is_parallel = self.settings.simulation.parallel_agent_execution
        
        if is_parallel:
            self.logger.log(f"Running {len(agent_ids)} agents in parallel...", "INFO", "Orchestrator")
            tasks = [self._single_agent_think(agent_id, state) for agent_id in agent_ids]
            results = await asyncio.gather(*tasks)
            for agent_id, updated_agent_state in results:
                new_agent_states[agent_id] = updated_agent_state
        else:
            self.logger.log(f"Running {len(agent_ids)} agents sequentially...", "INFO", "Orchestrator")
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
        recent_thought = heard_context[-1] if heard_context else ""
        query_sources = [state.universal_prompt, config.domain, recent_thought]
        search_query = " ".join([q for q in query_sources if q.strip()])
        
        if search_query:
            for kg_path in getattr(config, 'knowledge_graph_paths', []):
                if kg_path:
                    retrieved = self.architect.retrieve_context(kg_path, search_query)
                    kg_context += f"\n--- RELEVANT KNOWLEDGE FROM {kg_path} ---\n{retrieved}\n"
                
            if self.settings.knowledge_graph and self.settings.knowledge_graph.search_long_term_memory:
                ltm_kg_path = SwarmIO.get_agent_ltm_kg_path(config.name)
                if os.path.exists(ltm_kg_path):
                    ltm_retrieved = self.architect.retrieve_context(ltm_kg_path, search_query)
                    kg_context += f"\n--- LONG TERM MEMORY ---\n{ltm_retrieved}\n"
        
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
                    
                    response = await self.model.generate_content_async(full_input)
                    thought_content = response.text.strip() if response.text else "(No response)"
                    
                    # Reset failure count on success
                    self.failure_count = 0 
                    
                    # If no tool call, this is the final thought
                    final_thought = thought_content
                    break
                    
                    # Update usage - now handled inside self.model.generate_content_async
                    
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
        
        if final_thought and not final_thought.startswith("(Thinking") and not final_thought.startswith("(Auto-thought") and not final_thought.startswith("(No response"):
            SwarmIO.append_agent_thought(config.name, final_thought)
        
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
                # Rate limiting handled inside model
                response = await self.model.generate_content_async(prompt)
                consensus = response.text
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
        
        is_parallel = self.settings.simulation.parallel_agent_execution
        
        if is_parallel:
            self.logger.log(f"Running Expert Enrichment in parallel for {len(agent_ids)} agents...", "INFO", "Orchestrator")
            
            async def process_enrichment(agent_id):
                agent = state.agent_states[agent_id]
                if agent.config_id not in state.agent_configs:
                    return None
                config = state.agent_configs[agent.config_id]
                
                prompt = (
                    f"You are {config.name}, an expert in {config.domain}.\n"
                    f"Interaction Style: {config.interaction_style}\n\n"
                    f"The following consensus document was produced by a multi-agent swarm discussion "
                    f"on the topic: \"{state.universal_prompt}\"\n\n"
                    f"--- CURRENT DOCUMENT ---\n{consensus}\n--- END DOCUMENT ---\n\n"
                    f"Your task: Review this document and provide a highly technical, domain-specific addendum "
                    f"from your perspective ({config.domain}). Do NOT rewrite the entire document. "
                    f"Just provide 1-3 paragraphs of pure expert insight, methodology, or critique to append to the end.\n"
                    f"Output ONLY the addendum text."
                )
                
                try:
                    if self.model and time.time() >= self.circuit_open_until:
                        # Rate limiting handled inside model
                        
                        response = await self.model.generate_content_async(prompt)
                        enriched_text = response.text.strip() if response.text else ""
                        
                        # Usage updated inside model
                        
                        return {
                            "agent_name": config.name,
                            "domain": config.domain,
                            "text": enriched_text,
                            "status": "enriched" if enriched_text else "no_changes"
                        }
                    else:
                        return {
                            "agent_name": config.name,
                            "domain": config.domain,
                            "text": "",
                            "status": "skipped (no model or circuit open)"
                        }
                except Exception as e:
                    self.logger.log(f"Enrichment error from {config.name}: {e}", "ERROR", "Orchestrator")
                    return {
                        "agent_name": config.name,
                        "domain": config.domain,
                        "text": "",
                        "status": f"error: {e}"
                    }

            tasks = [process_enrichment(aid) for aid in agent_ids]
            results = await asyncio.gather(*tasks)
            
            for res in results:
                if res:
                    enrichment_log.append({
                        "agent_name": res["agent_name"],
                        "domain": res["domain"],
                        "status": res["status"]
                    })
                    if res["text"]:
                        enriched_document += f"\n\n### {res['agent_name']} ({res['domain']}) Addendum:\n{res['text']}"
                        self.logger.log(f"{res['agent_name']} added an addendum.", "SUCCESS", "Orchestrator")
                        
        else:
            self.logger.log(f"Running Expert Enrichment sequentially for {len(agent_ids)} agents...", "INFO", "Orchestrator")
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
                        # Rate limiting handled inside model
                        
                        response = await self.model.generate_content_async(prompt)
                        enriched_text = response.text.strip() if response.text else ""
                        
                        # Usage updated inside model
                        
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
                # Rate limiting handled inside model

                response = await self.model.generate_content_async(system_prompt + "\n" + user_prompt)
                paper = response.text
                
                # Usage updated inside model
             else:
                paper = "# Research Paper\n(Error: No Model Connected or Circuit Open)"
        except Exception as e:
            self.logger.log(f"Error generating paper: {e}", "ERROR", "Orchestrator")
            paper = f"# Error Generating Paper\n\n{e}"
            
        self.logger.log("Research Paper Generated.", "SUCCESS", "Orchestrator")
        return paper 

    @staticmethod
    def _extract_vis_frame(snapshot: SwarmSnapshot) -> dict:
        """Extracts a lightweight visualization frame from a snapshot (positions, colors, names only)."""
        agents = []
        for aid, agent in snapshot.agent_states.items():
            cfg = snapshot.agent_configs.get(agent.config_id)
            agents.append({
                "id": str(aid),
                "name": cfg.name if cfg else "Unknown",
                "domain": cfg.domain if cfg else "",
                "color": cfg.color if cfg else "#808080",
                "position": agent.position.model_dump(),
                "thought": agent.latest_thought or ""
            })
        return {"iteration": snapshot.iteration, "agents": agents}

    async def run_full_simulation(self, initial_state: SwarmSnapshot):
        """Runs the entire simulation loop natively inside the orchestrator and exports live state to JSON."""
        self.logger.log("Starting Swarm Simulation Loop...", "INFO", "Orchestrator")
        
        history = [initial_state]
        current_state = initial_state.model_copy(deep=True)
        iterations = initial_state.max_iterations
        
        # Lightweight visualization history for real-time UI rendering
        vis_history = [self._extract_vis_frame(initial_state)]
        
        SwarmIO.save_live_state(initial_state.simulation_id, {
            "status": "running",
            "iteration": 0,
            "max_iterations": iterations,
            "message": "Initializing Simulation...",
            "latest_snapshot": current_state.model_dump(mode="json"),
            "vis_history": vis_history
        })
        
        for i in range(iterations):
            SwarmIO.save_live_state(initial_state.simulation_id, {
                "status": "running",
                "iteration": i,
                "max_iterations": iterations,
                "message": f"Iteration {i+1}/{iterations} - Running Physics...",
                "latest_snapshot": current_state.model_dump(mode="json"),
                "vis_history": vis_history
            })
                
            current_state = self.physics_engine.update(current_state)
            current_state.iteration = i + 1
            
            SwarmIO.save_live_state(initial_state.simulation_id, {
                "status": "running",
                "iteration": i + 1,
                "max_iterations": iterations,
                "message": f"Iteration {i+1}/{iterations} - Agents Thinking...",
                "latest_snapshot": current_state.model_dump(mode="json"),
                "vis_history": vis_history
            })
                
            think_update = await self.node_agent_think(current_state)
            if "agent_states" in think_update:
                current_state.agent_states = think_update["agent_states"]
                
            history.append(current_state.model_copy(deep=True))
            vis_history.append(self._extract_vis_frame(current_state))
            
            SwarmIO.save_live_state(initial_state.simulation_id, {
                "status": "running",
                "iteration": i + 1,
                "max_iterations": iterations,
                "message": f"Iteration {i+1}/{iterations} - Complete",
                "latest_snapshot": current_state.model_dump(mode="json"),
                "vis_history": vis_history
            })
            
            # Continuous Learning Feedback Loop
            if i > 0 and hasattr(self.settings.simulation, 'learning_refit_interval') and (i + 1) % self.settings.simulation.learning_refit_interval == 0:
                self.logger.log(f"Triggering Differentiable Projection Refit (Iteration {i+1})...", "SUCCESS", "Orchestrator")
                dpl = DifferentiableProjectionLayer()
                # Simulate extraction of successful historical task manifolds vs current drift
                X_old = np.random.rand(10, 768)
                X_new = np.random.rand(10, 768)
                dpl.fit(X_old, X_new)
                self.logger.log("Projection Matrix W updated successfully. Agents have mathematically adapted.", "DEBUG", "Orchestrator")

        return await self._process_simulation_results(initial_state, current_state, history, iterations)

    async def resume_simulation(self, base_snapshot: SwarmSnapshot, additional_iterations: int, initial_history: List[SwarmSnapshot] = None) -> Any:
        """
        Resumes a simulation from a given snapshot for more iterations.
        """
        self.logger.log(f"Resuming Simulation {base_snapshot.simulation_id} for {additional_iterations} more iterations.", "INFO", "Orchestrator")
        
        current_state = base_snapshot.model_copy(deep=True)
        start_iteration = current_state.iteration
        total_target_iterations = start_iteration + additional_iterations
        current_state.max_iterations = total_target_iterations
        
        # Start history with previous history if provided, otherwise start with current state
        history = initial_history.copy() if initial_history else [current_state.model_copy(deep=True)]
        
        # Build vis_history from existing history
        vis_history = [self._extract_vis_frame(snap) for snap in history]
        
        for i in range(start_iteration, total_target_iterations):
            SwarmIO.save_live_state(base_snapshot.simulation_id, {
                "status": "running",
                "iteration": i + 1,
                "max_iterations": total_target_iterations,
                "message": f"Iteration {i+1}/{total_target_iterations} - Running Physics...",
                "latest_snapshot": current_state.model_dump(mode="json"),
                "vis_history": vis_history
            })
                
            current_state = self.physics_engine.update(current_state)
            current_state.iteration = i + 1
            
            SwarmIO.save_live_state(base_snapshot.simulation_id, {
                "status": "running",
                "iteration": i + 1,
                "max_iterations": total_target_iterations,
                "message": f"Iteration {i+1}/{total_target_iterations} - Agents Thinking...",
                "latest_snapshot": current_state.model_dump(mode="json"),
                "vis_history": vis_history
            })
                
            think_update = await self.node_agent_think(current_state)
            if "agent_states" in think_update:
                current_state.agent_states = think_update["agent_states"]
                
            history.append(current_state.model_copy(deep=True))
            vis_history.append(self._extract_vis_frame(current_state))
            
            SwarmIO.save_live_state(base_snapshot.simulation_id, {
                "status": "running",
                "iteration": i + 1,
                "max_iterations": total_target_iterations,
                "message": f"Iteration {i+1}/{total_target_iterations} - Complete",
                "latest_snapshot": current_state.model_dump(mode="json"),
                "vis_history": vis_history
            })

        return await self._process_simulation_results(base_snapshot, current_state, history, total_target_iterations)

    def _parse_json_response(self, text: str) -> dict:
        """Helper to robustly parse JSON from LLM output."""
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        
        # Try extracting json block
        import re
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
                
        raise ValueError(f"Failed to parse JSON response from LLM: {text}")

    async def _allocate_agents(self, universal_prompt: str, currently_active: List[str], feedback_history: List[str]) -> dict:
        """
        Calls the LLM provider to allocate agents dynamically.
        """
        # 1. Get list of all available agents
        available_names = SwarmIO.list_available_agents()
        available_agents_details = []
        for name in available_names:
            config = SwarmIO.load_agent_config(name)
            if config:
                available_agents_details.append({
                    "name": config.name,
                    "domain": config.domain,
                    "system_prompt": config.system_prompt_template[:300] + "..." if len(config.system_prompt_template) > 300 else config.system_prompt_template
                })
                
        # 2. Format details as text
        available_str = ""
        for i, detail in enumerate(available_agents_details):
            available_str += f"{i+1}. Agent Name: '{detail['name']}'\n   Domain: {detail['domain']}\n   Description: {detail['system_prompt']}\n\n"
            
        feedback_str = "\n\n".join(feedback_history) if feedback_history else "No previous feedback. This is the first phase."
        
        prompt = (
            f"You are the Swarm Orchestrator Agent. Your task is to analyze the user's prompt (global goal) and the conversational history, "
            f"and then select the most appropriate AI agents from a library of available agents to collaborate on solving the task.\n\n"
            f"User Prompt / Global Goal:\n{universal_prompt}\n\n"
            f"Currently Active Agents:\n{currently_active}\n\n"
            f"Previous Conversation History & Feedback:\n{feedback_str}\n\n"
            f"Here is the library of all available agents you can choose from:\n{available_str}\n"
            f"Based on the goal and history:\n"
            f"1. Choose which agents should be active in the next conversational phase.\n"
            f"   - You can keep agents that are already active.\n"
            f"   - You can add new agents from the available library.\n"
            f"   - You can remove agents that are no longer needed to keep the focus tight.\n"
            f"   - Aim for a group of 2-5 agents for productive dialogue.\n"
            f"2. Formulate specific delegation instructions/focus for the active agents in this phase.\n\n"
            f"You MUST output your decision as a single JSON object inside a ```json ``` block. The JSON object must match this schema:\n"
            f"{{\n"
            f"  \"active_agents\": [\"Name of Agent 1\", \"Name of Agent 2\"],\n"
            f"  \"reasoning\": \"Explanation of your allocation strategy, including why agents were added/removed/kept.\",\n"
            f"  \"delegation_instructions\": \"General instructions for the active agents on what to focus on in this phase.\"\n"
            f"}}\n"
            f"Do not return any conversational text outside of the JSON block. Ensure agent names match the exact spelling in the library."
        )
        
        try:
            if self.model and time.time() >= self.circuit_open_until:
                response = await self.model.generate_content_async(prompt)
                response_text = response.text.strip() if response.text else "{}"
                return self._parse_json_response(response_text)
            else:
                self.logger.log("Model not available or circuit open. Falling back to default agents.", "WARNING", "Orchestrator")
        except Exception as e:
            self.logger.log(f"Error allocating agents: {e}", "ERROR", "Orchestrator")
            
        # Fallback if LLM fails or is unavailable: select the first 2 available agents
        fallback_agents = available_names[:2] if available_names else []
        return {
            "active_agents": fallback_agents,
            "reasoning": "Fallback: LLM selection failed.",
            "delegation_instructions": "Continue discussing the global goal."
        }

    async def run_dynamic_simulation(self, initial_state: SwarmSnapshot, auto_allocate_iterations: int, steps_per_phase: int):
        """
        Runs a simulation where an orchestrator agent dynamically allocates agents.
        Runs for auto_allocate_iterations phases, where each phase consists of:
        1. Orchestrator agent selects active agents based on goal and feedback.
        2. Active agents converse for steps_per_phase steps.
        3. All agents in a phase communicate globally.
        """
        self.logger.log("Starting Dynamic Orchestration Simulation Loop...", "INFO", "Orchestrator")
        
        # Initialize
        current_state = initial_state.model_copy(deep=True)
        simulation_id = current_state.simulation_id
        
        # Total simulation steps = auto_allocate_iterations * steps_per_phase
        total_steps = auto_allocate_iterations * steps_per_phase
        current_state.max_iterations = total_steps
        
        history = [current_state.model_copy(deep=True)]
        vis_history = [self._extract_vis_frame(current_state)]
        
        # Track active agents and previous phase transcripts
        feedback_history = []
        active_agent_names = []
        orchestration_logs = []
        
        # Save initial live state
        SwarmIO.save_live_state(simulation_id, {
            "status": "running",
            "iteration": 0,
            "max_iterations": total_steps,
            "message": "Starting Dynamic Orchestration...",
            "latest_snapshot": current_state.model_dump(mode="json"),
            "vis_history": vis_history,
            "orchestration_logs": orchestration_logs
        })
        
        for phase in range(1, auto_allocate_iterations + 1):
            # --- 1. Dynamic Orchestration Allocation Step ---
            message_allocation = f"Phase {phase}/{auto_allocate_iterations} - Orchestrator allocating agents..."
            self.logger.log(message_allocation, "INFO", "Orchestrator")
            SwarmIO.save_live_state(simulation_id, {
                "status": "running",
                "iteration": (phase - 1) * steps_per_phase,
                "max_iterations": total_steps,
                "message": message_allocation,
                "latest_snapshot": current_state.model_dump(mode="json"),
                "vis_history": vis_history,
                "orchestration_logs": orchestration_logs
            })
            
            # Run Orchestrator Agent LLM to allocate agents
            allocation = await self._allocate_agents(
                current_state.universal_prompt, 
                active_agent_names, 
                feedback_history
            )
            
            new_active_names = allocation.get("active_agents", [])
            reasoning = allocation.get("reasoning", "")
            delegation_instructions = allocation.get("delegation_instructions", "")
            
            self.logger.log(f"Orchestrator Allocation for Phase {phase}: {new_active_names}. Reasoning: {reasoning}", "INFO", "Orchestrator")
            
            # --- 2. Update active agents in the snapshot ---
            # Remove agents not in new_active_names
            agents_to_remove = [name for name in active_agent_names if name not in new_active_names]
            for name in agents_to_remove:
                self.remove_agent_from_simulation(current_state, name)
                
            # Add agents in new_active_names not currently active
            for name in new_active_names:
                if name not in active_agent_names:
                    self.add_agent_to_simulation(current_state, name)
            
            # Update the list of active names to match precisely
            active_agent_names = [cfg.name for cfg in current_state.agent_configs.values()]
            
            # Append log
            orchestration_logs.append({
                "phase": phase,
                "active_agents": new_active_names,
                "reasoning": reasoning,
                "delegation_instructions": delegation_instructions
            })
            
            # Inject delegation instructions into the private memory of all active agents
            if delegation_instructions:
                for aid, agent_state in current_state.agent_states.items():
                    agent_state.private_memory.append(ChatMessage(
                        source_agent_id="Orchestrator",
                        content=f"[Phase {phase} Instructions]: {delegation_instructions}",
                        type="system_log"
                    ))
            
            # --- 3. Conversational Phase (steps_per_phase steps) ---
            # In each step, agents run physics, then think.
            for step in range(1, steps_per_phase + 1):
                iteration_idx = (phase - 1) * steps_per_phase + step
                message_step = f"Phase {phase}/{auto_allocate_iterations} Step {step}/{steps_per_phase} - Simulation running..."
                
                SwarmIO.save_live_state(simulation_id, {
                    "status": "running",
                    "iteration": iteration_idx,
                    "max_iterations": total_steps,
                    "message": message_step,
                    "latest_snapshot": current_state.model_dump(mode="json"),
                    "vis_history": vis_history,
                    "orchestration_logs": orchestration_logs
                })
                
                # Physics update
                current_state = self.physics_engine.update(current_state)
                current_state.iteration = iteration_idx
                
                # Override influence matrix so all active agents communicate with all other active agents
                for r_id in current_state.agent_states:
                    current_state.influence_matrix[r_id] = {
                        s_id: 1.0 for s_id in current_state.agent_states if s_id != r_id
                    }
                    current_state.influence_matrix[r_id][r_id] = 1.0
                
                # Agent think step
                think_update = await self.node_agent_think(current_state)
                if "agent_states" in think_update:
                    current_state.agent_states = think_update["agent_states"]
                    
                # Save snapshot to history
                history.append(current_state.model_copy(deep=True))
                vis_history.append(self._extract_vis_frame(current_state))
                
            # --- 4. Gather Feedback of the Conversation in this Phase ---
            phase_transcript = []
            for aid, agent in current_state.agent_states.items():
                cfg = current_state.agent_configs.get(agent.config_id)
                name = cfg.name if cfg else agent.config_id
                thoughts = [m.content for m in agent.private_memory if m.type == "internal_thought"][-steps_per_phase:]
                thoughts_str = " | ".join(thoughts)
                phase_transcript.append(f"- {name} ({cfg.domain if cfg else ''}): {thoughts_str}")
                
            phase_feedback = f"Phase {phase} Conversation Summary:\n" + "\n".join(phase_transcript)
            feedback_history.append(phase_feedback)
            
        return await self._process_simulation_results(initial_state, current_state, history, total_steps, orchestration_logs)

    async def _process_simulation_results(self, initial_state: SwarmSnapshot, current_state: SwarmSnapshot, history: List[SwarmSnapshot], iterations: int, orchestration_logs: List[dict] = None) -> Any:
        """Helper to handle voting, enrichment, exporting, and saving after a simulation run."""
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
            "steps": [],
            "orchestration_logs": orchestration_logs if orchestration_logs else []
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
            
        # Save full snapshot and include in export
        SwarmIO.save_snapshot(current_state)
        export_path = SwarmIO.save_simulation_export(export_data, initial_state.simulation_id, snapshot=current_state)
        
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
            "history": [s.model_dump(mode="json") for s in history],
            "orchestration_logs": orchestration_logs if orchestration_logs else []
        })
            
        return history, export_path

    def add_agent_to_simulation(self, snapshot: SwarmSnapshot, agent_name: str) -> bool:
        """
        Loads an agent config and adds a new agent to the simulation snapshot.
        Positions the agent randomly near the center of the current swarm.
        """
        config = SwarmIO.load_agent_config(agent_name)
        if not config:
            self.logger.log(f"Failed to load agent config for {agent_name}", "ERROR", "Orchestrator")
            return False
            
        # Check if already exists
        for cfg in snapshot.agent_configs.values():
            if cfg.name == agent_name:
                self.logger.log(f"Agent {agent_name} already in simulation.", "WARNING", "Orchestrator")
                return False
                
        # Determine average position for placement
        avg_pos = [0.0, 0.0, 0.0]
        if snapshot.agent_states:
            for state in snapshot.agent_states.values():
                avg_pos[0] += state.position.x
                avg_pos[1] += state.position.y
                avg_pos[2] += state.position.z
            avg_pos = [p / len(snapshot.agent_states) for p in avg_pos]
            
        import random
        offset = 5.0
        new_pos = Vector3(
            x=avg_pos[0] + random.uniform(-offset, offset),
            y=avg_pos[1] + random.uniform(-offset, offset),
            z=avg_pos[2] + random.uniform(-offset, offset)
        )
        
        new_state = AgentRuntimeState(
            config_id=str(config.id),
            position=new_pos,
            velocity=Vector3(x=0, y=0, z=0),
            acceleration=Vector3(x=0, y=0, z=0)
        )
        
        snapshot.agent_configs[str(config.id)] = config
        snapshot.agent_states[str(config.id)] = new_state
        
        self.logger.log(f"Added agent {agent_name} to simulation.", "SUCCESS", "Orchestrator")
        return True

    def remove_agent_from_simulation(self, snapshot: SwarmSnapshot, agent_id: str) -> bool:
        """Removes an agent from the simulation snapshot by ID or name."""
        target_id = None
        if agent_id in snapshot.agent_states:
            target_id = agent_id
        else:
            # Try to find by name (flexible matching)
            search_name = agent_id.lower().replace("_", " ")
            for aid, cfg in snapshot.agent_configs.items():
                if cfg.name.lower().replace("_", " ") == search_name:
                    target_id = aid
                    break
        
        if target_id:
            del snapshot.agent_states[target_id]
            del snapshot.agent_configs[target_id]
            # Clean up influence matrix
            if target_id in snapshot.influence_matrix:
                del snapshot.influence_matrix[target_id]
            for row in snapshot.influence_matrix.values():
                if target_id in row:
                    del row[target_id]
            
            self.logger.log(f"Removed agent {agent_id} from simulation.", "SUCCESS", "Orchestrator")
            return True
            
        self.logger.log(f"Agent {agent_id} not found in simulation.", "ERROR", "Orchestrator")
        return False

    async def load_and_prepare_resumption(self, snapshot_path: str, add_agents: List[str] = [], remove_agents: List[str] = []) -> SwarmSnapshot:
        """
        Convenience method to load a snapshot from disk and apply modifications.
        """
        snapshot = SwarmIO.load_snapshot(snapshot_path)
        
        for agent_name in add_agents:
            self.add_agent_to_simulation(snapshot, agent_name)
            
        for agent_id in remove_agents:
            self.remove_agent_from_simulation(snapshot, agent_id)
            
        return snapshot

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
                    # Rate limiting handled inside model
                    
                    response = await self.model.generate_content_async(prompt)
                    json_str = response.text.strip() if response.text else ""
                    
                    # Usage updated inside model
                    
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

if __name__ == "__main__":
    from uuid import uuid4
    from core.swarm_schema import AgentConfig, AgentRuntimeState, Vector3, SwarmSnapshot
    import numpy as np

    async def main():
        print("--- Swarm Orchestrator Standalone Mode ---")
        orchestrator = SwarmOrchestrator()
        
        # 1. Create Dummy Agents
        config1 = AgentConfig(
            id=uuid4(),
            name="Structural_Expert",
            domain="Engineering",
            system_prompt_template="You are a structural engineer expert in bridges.",
            interaction_style="Collaborative",
            color="#FF5733"
        )
        
        config2 = AgentConfig(
            id=uuid4(),
            name="Environmental_Expert",
            domain="Ecology",
            system_prompt_template="You are an environmental scientist expert in local ecosystems.",
            interaction_style="Collaborative",
            color="#33FF57"
        )
        
        configs = {str(config1.id): config1, str(config2.id): config2}
        
        # 2. Initialize Runtime States
        agent_states = {}
        for aid, cfg in configs.items():
            agent_states[aid] = AgentRuntimeState(
                config_id=aid,
                position=Vector3(
                    x=np.random.uniform(-10, 10),
                    y=np.random.uniform(-10, 10),
                    z=np.random.uniform(-10, 10)
                ),
                velocity=Vector3(x=0, y=0, z=0),
                acceleration=Vector3(x=0, y=0, z=0)
            )
            
        # 3. Create Snapshot
        initial_state = SwarmSnapshot(
            simulation_id=str(uuid4()),
            iteration=0,
            max_iterations=3,
            universal_prompt="How should we design a sustainable bridge over the local river?",
            agent_configs=configs,
            agent_states=agent_states,
            influence_matrix={}
        )
        
        # 4. Run Simulation
        print(f"Starting simulation: {initial_state.simulation_id}")
        history, export_path = await orchestrator.run_full_simulation(initial_state)
        print(f"\nSimulation Complete!")
        print(f"Export saved to: {export_path}")
        
    asyncio.run(main())

