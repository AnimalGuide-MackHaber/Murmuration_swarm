import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import asyncio
import os
import traceback
from uuid import uuid4
from datetime import datetime
import time
import numpy as np

from swarm_schema import AgentConfig, SwarmSnapshot, AgentRuntimeState, Vector3
from swarm_io import SwarmIO
from swarm_orchestrator import SwarmOrchestrator
from settings_manager import SettingsManager, GlobalSettings, PhysicsSettings, PhysicsWeights
from knowledge_graph import KnowledgeArchitect
from swarm_logger import SwarmLogger
from DOAB_client import DOABClient

# Page Config
st.set_page_config(page_title="Swarm Intelligence Control Tower", layout="wide", page_icon="🐝")

# Initialize Directories
SwarmIO.ensure_directories()

# Initialize Session State for DOAB Pipeline
if "doab_pipeline_state" not in st.session_state:
    st.session_state.doab_pipeline_state = "IDLE"
if "doab_progress" not in st.session_state:
    st.session_state.doab_progress = 0
if "doab_status" not in st.session_state:
    st.session_state.doab_status = ""
if "doab_error" not in st.session_state:
    st.session_state.doab_error = ""
if "doab_result" not in st.session_state:
    st.session_state.doab_result = ""

# Initialize Managers with Caching
@st.cache_resource
def get_core_services():
    """Initializes core services once and caches them."""
    _settings_manager = SettingsManager()
    _orchestrator = SwarmOrchestrator()
    _architect = KnowledgeArchitect()
    _logger = SwarmLogger()
    return _settings_manager, _orchestrator, _architect, _logger

settings_manager, orchestrator, architect, logger = get_core_services()

# Check for API Key (provider-specific)
_init_provider = settings_manager.get_settings().llm_provider.lower()
if _init_provider == "gemini" and not settings_manager.get_api_key():
    st.error("⚠️ Google API Key is missing! Please set `GOOGLE_API_KEY` environment variable or add `google_api_key` to `swarm_settings.json`.")
elif _init_provider == "nvidia" and not settings_manager.get_nvidia_api_key():
    st.error("⚠️ Nvidia API Key is missing! Please set `NVIDIA_API_KEY` environment variable or add it to `swarm_settings.json`.")


# --- Sidebar: Physics Tuner ---
st.sidebar.header("Physics Engine Tuner")

current_settings = settings_manager.get_settings()
physics = current_settings.physics
weights = physics.weights

# Sliders
new_sep = st.sidebar.slider("Separation Weight", 0.0, 5.0, weights.separation, 0.1)
new_ali = st.sidebar.slider("Alignment Weight", 0.0, 5.0, weights.alignment, 0.1)
new_coh = st.sidebar.slider("Cohesion Weight", 0.0, 5.0, weights.cohesion, 0.1)
new_goal = st.sidebar.slider("Goal Seeking Weight", 0.0, 5.0, weights.goal_seeking, 0.1)
new_vis = st.sidebar.slider("Visual Range", 1.0, 100.0, physics.visual_range, 1.0)
new_speed = st.sidebar.slider("Max Speed", 1.0, 20.0, physics.max_speed, 0.5)

# Update Settings
if (new_sep != weights.separation or new_ali != weights.alignment or 
    new_coh != weights.cohesion or new_goal != weights.goal_seeking or
    new_vis != physics.visual_range or new_speed != physics.max_speed):
    
    new_weights = weights.model_copy(update={
        "separation": new_sep,
        "alignment": new_ali,
        "cohesion": new_coh,
        "goal_seeking": new_goal
    })
    
    new_physics = physics.model_copy(update={
        "visual_range": new_vis,
        "max_speed": new_speed,
        "weights": new_weights
    })
    
    updated_settings = current_settings.model_copy(update={
        "physics": new_physics
    })
    
    settings_manager.update_settings(updated_settings)
    # Persist? 
    # settings_manager.save_settings() # Optional: auto-save or explicit save button.

# --- Sidebar: Model Provider Selector ---
st.sidebar.divider()
st.sidebar.header("🤖 Model Provider")

PROVIDER_OPTIONS = ["ollama", "nvidia", "gemini"]
PROVIDER_LABELS = {"ollama": "Ollama (Local)", "nvidia": "Nvidia NIM", "gemini": "Gemini API"}

current_provider = current_settings.llm_provider.lower()
current_model = current_settings.model_name

# Default model suggestions per provider
DEFAULT_MODELS = {
    "ollama": "gemma4:e2b",
    "nvidia": "moonshotai/kimi-k2.5",
    "gemini": "gemini-2.5-flash"
}

provider_index = PROVIDER_OPTIONS.index(current_provider) if current_provider in PROVIDER_OPTIONS else 0

selected_provider = st.sidebar.selectbox(
    "LLM Provider",
    PROVIDER_OPTIONS,
    index=provider_index,
    format_func=lambda x: PROVIDER_LABELS.get(x, x)
)

# If provider changed, suggest default model for that provider
if selected_provider != current_provider:
    suggested_model = DEFAULT_MODELS.get(selected_provider, current_model)
else:
    suggested_model = current_model

selected_model = st.sidebar.text_input("Model Name", value=suggested_model)

# Show rate limits for selected provider
rl = settings_manager.get_rate_limiting_for_provider(selected_provider)
st.sidebar.caption(f"**Rate Limits ({PROVIDER_LABELS.get(selected_provider, selected_provider)}):**")
st.sidebar.caption(f"  RPM: {rl.requests_per_minute:,}  |  TPM: {rl.tokens_per_minute:,}  |  Delay: {rl.request_delay_seconds}s")

# Apply changes if provider or model changed
if selected_provider != current_provider or selected_model != current_model:
    if st.sidebar.button("⚡ Apply Model Change", use_container_width=True):
        updated_settings = current_settings.model_copy(update={
            "llm_provider": selected_provider,
            "model_name": selected_model
        })
        settings_manager.update_settings(updated_settings)
        settings_manager.save_settings()
        
        # Hotswap the orchestrator's provider
        orchestrator.switch_provider(selected_provider, selected_model)
        
        st.sidebar.success(f"Switched to {PROVIDER_LABELS.get(selected_provider, selected_provider)}: {selected_model}")
        st.cache_resource.clear()
        st.rerun()


# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Agent Forge", "Swarm Simulation", "Inspector", "Knowledge Graph Viewer"])

# --- Tab 1: Agent Forge ---
with tab1:
    st.header("Agent Forge")
    
    forge_tab1, forge_tab2 = st.tabs(["Knowledge Lab", "Agent Workshop"])
    
    # --- Sub-Tab 1: Knowledge Lab ---
    with forge_tab1:
        st.subheader("Knowledge Graph Architect")
        st.write("Upload documents or search DOAB to generate domain-specific Knowledge Graphs.")
        
        kg_domain = st.text_input("Domain Name for KG", placeholder="e.g. Neuroscience, QuantumPhysics")
        
        source_mode = st.radio("Knowledge Source", ["File Upload", "DOAB Search"], horizontal=True)
        
        if source_mode == "File Upload":
            uploaded_files = st.file_uploader("Upload Knowledge Source", type=["pdf", "md", "txt"], accept_multiple_files=True)
            
            if uploaded_files and st.button("Build Knowledge Graph"):
                if not kg_domain:
                    st.error("Please specify a Domain Name.")
                else:
                    with st.spinner("Architecting Knowledge Graph..."):
                        temp_paths = []
                        for uploaded_file in uploaded_files:
                            temp_path = os.path.join("swarm_data", uploaded_file.name)
                            with open(temp_path, "wb") as f:
                                f.write(uploaded_file.getbuffer())
                            temp_paths.append(temp_path)
                            
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        def update_progress(current, total):
                            percent = int((current / total) * 100)
                            progress_bar.progress(percent)
                            status_text.text(f"Processing chunk {current}/{total}...")
                            
                        try:
                            status_text.text(f"Ingesting {len(temp_paths)} files and building graph...")
                            kg_filename = architect.build_graph(temp_paths, kg_domain.replace(" ", "_"))
                            progress_bar.progress(100)
                            status_text.text("Done!")
                            st.success(f"Knowledge Graph created successfully: `{kg_filename}`")
                        except Exception as e:
                            st.error(f"Error creating Knowledge Graph: {e}")
                        finally:
                            for temp_path in temp_paths:
                                if os.path.exists(temp_path):
                                    os.remove(temp_path)
                                    
        elif source_mode == "DOAB Search":
            search_prompt = st.text_area("Describe what you want to learn about", placeholder="e.g. I want to learn about the effects of climate change on marine biology.")
            max_pdfs = st.number_input("Max PDFs to download", min_value=1, max_value=10, value=3)
            
            def run_doab_pipeline_thread(prompt, domain, limit):
                import threading
                import asyncio
                try:
                    from streamlit.runtime.scriptrunner import add_script_run_ctx
                except ImportError:
                    add_script_run_ctx = None

                def background_worker():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        st.session_state.doab_pipeline_state = "RUNNING"
                        st.session_state.doab_progress = 0
                        st.session_state.doab_status = "Generating search terms..."
                        st.session_state.doab_error = ""
                        
                        # 1. Generate search terms
                        logger.log(f"Starting DOAB Search Pipeline for prompt: '{prompt}'", "INFO", "App")
                        prompt_for_terms = f"You are an expert researcher. Extract 2 to 4 distinct, concise search keywords or short phrases from this prompt to be used individually in database queries: '{prompt}'. Return ONLY a comma-separated list of the terms. Do not include any conversational filler, quotes, or markdown."
                        
                        response = loop.run_until_complete(architect.model.generate_content_async(prompt_for_terms))
                        raw_response = response.text.strip().replace('"', '')
                        term_list = [t.strip() for t in raw_response.split(',') if t.strip()]
                        logger.log(f"Parsed Search Terms: {term_list}", "INFO", "App")
                        
                        # 2. Download from DOAB
                        st.session_state.doab_progress = 20
                        st.session_state.doab_status = "Searching DOAB..."
                        client = DOABClient()
                        downloaded_files = []
                        
                        for i, term in enumerate(term_list):
                            if len(downloaded_files) >= limit:
                                break
                            
                            remaining_limit = limit - len(downloaded_files)
                            st.session_state.doab_status = f"Searching and downloading for: '{term}'..."
                            files = client.download_pdfs(term, limit=remaining_limit)
                            
                            for f in files:
                                if f not in downloaded_files:
                                    downloaded_files.append(f)
                            
                            # Increment progress
                            st.session_state.doab_progress = 20 + int(30 * ((i+1) / len(term_list)))

                        if not downloaded_files:
                            st.session_state.doab_pipeline_state = "ERROR"
                            st.session_state.doab_error = "No PDFs found or downloaded from DOAB for these terms."
                            return

                        # 3. Build Graph
                        st.session_state.doab_progress = 50
                        st.session_state.doab_status = f"Ingesting {len(downloaded_files)} files and building graph..."
                        
                        def progress_cb(curr, total):
                            pct = 50 + int(50 * (curr / total))
                            st.session_state.doab_progress = pct
                            st.session_state.doab_status = f"Building Knowledge Graph (chunk {curr}/{total})..."

                        kg_filename = loop.run_until_complete(architect.build_graph_async(downloaded_files, domain.replace(" ", "_"), progress_callback=progress_cb))
                        
                        st.session_state.doab_pipeline_state = "SUCCESS"
                        st.session_state.doab_result = kg_filename
                        st.session_state.doab_status = "Done!"
                        st.session_state.doab_progress = 100
                        logger.log(f"DOAB Pipeline completed. KG: {kg_filename}", "SUCCESS", "App")

                    except Exception as e:
                        error_trace = traceback.format_exc()
                        logger.log(f"DOAB Pipeline Error: {e}\n{error_trace}", "ERROR", "App")
                        st.session_state.doab_pipeline_state = "ERROR"
                        st.session_state.doab_error = str(e)
                    finally:
                        loop.close()

                t = threading.Thread(target=background_worker)
                if add_script_run_ctx:
                    add_script_run_ctx(t)
                t.start()

            # UI for Triggering and Progress
            if st.session_state.doab_pipeline_state == "RUNNING":
                st.button("Search DOAB & Build Knowledge Graph", disabled=True)
                st.info(f"⏳ {st.session_state.doab_status}")
                st.progress(st.session_state.doab_progress / 100)
                # Auto-refresh to show progress
                time.sleep(1)
                st.rerun()
            else:
                if st.button("Search DOAB & Build Knowledge Graph"):
                    if not kg_domain:
                        st.error("Please specify a Domain Name.")
                    elif not search_prompt:
                        st.error("Please describe what you want to learn about.")
                    else:
                        run_doab_pipeline_thread(search_prompt, kg_domain, max_pdfs)
                        st.rerun()

            # Result/Error Display
            if st.session_state.doab_pipeline_state == "SUCCESS":
                st.success(f"Knowledge Graph created successfully: `{st.session_state.doab_result}`")
                if st.button("Clear Finished Task"):
                    st.session_state.doab_pipeline_state = "IDLE"
                    st.rerun()
            elif st.session_state.doab_pipeline_state == "ERROR":
                st.error(f"Error creating Knowledge Graph: {st.session_state.doab_error}")
                if st.button("Retry"):
                    st.session_state.doab_pipeline_state = "IDLE"
                    st.rerun()
                            
        # List existing KGs
        st.divider()
        st.subheader("Available Knowledge Graphs")
        kgs = SwarmIO.list_available_kgs()
        if kgs:
            st.code("\n".join(kgs), language="text")
        else:
            st.info("No Knowledge Graphs found.")

    # --- Sub-Tab 2: Agent Workshop ---
    with forge_tab2:
        st.subheader("Agent Workshop")
        
        mode = st.radio("Mode", ["Create New Agent", "Edit Existing Agent"], horizontal=True)
        
        # Initialize default values
        default_name = "Agent 001"
        default_domain = "Generalist"
        default_color = "#00CCFF"
        default_style = "Collaborative"
        default_prompt = "You are an AI agent that is a researcher and domain expert, and highly motivated to share your knowledge and to listen to and synthesize knowledge from other AI agents"
        default_kg_index = 0
        
        selected_agent_config = None
        current_agent_selection = None
        
        # Load existing agent if in edit mode
        if mode == "Edit Existing Agent":
            agents_list = SwarmIO.list_available_agents()
            if not agents_list:
                st.warning("No agents available to edit.")
                selected_agent_name = None
            else:
                selected_agent_name = st.selectbox("Select Agent to Edit", agents_list)
                current_agent_selection = selected_agent_name
                
            if selected_agent_name:
                selected_agent_config = SwarmIO.load_agent_config(selected_agent_name)
                if selected_agent_config:
                    default_name = selected_agent_config.name
                    default_domain = selected_agent_config.domain
                    default_color = selected_agent_config.color
                    default_style = selected_agent_config.interaction_style
                    default_prompt = selected_agent_config.system_prompt_template
                    # KG handling logic below

        # Update session state on selection change
        if st.session_state.get("forge_last_selection") != current_agent_selection:
            st.session_state.forge_sys_prompt = default_prompt
            st.session_state.sys_prompt_area = default_prompt
            st.session_state.prompt_generation_state = "IDLE"
            st.session_state.forge_last_selection = current_agent_selection
        
        col1, col2 = st.columns(2)
        
        with col1:
            # If editing, name might effectively change ID if we generated ID from name, but we use UUID.
            # However, filename is based on name. Renaming agent -> new file.
            name_val = st.text_input("Agent Name", value=default_name)
            domain_val = st.text_input("Domain Expertise", value=default_domain)
            color_val = st.color_picker("Agent Color", value=default_color)
            style_val = st.selectbox("Interaction Style", 
                                     ["Collaborative", "Debater", "Socratic", "Critic"], 
                                     index=["Collaborative", "Debater", "Socratic", "Critic"].index(default_style) if default_style in ["Collaborative", "Debater", "Socratic", "Critic"] else 0)
            
            # KG Selection
            available_kgs = SwarmIO.list_available_kgs()
            available_kgs_opts = ["None"] + available_kgs
            
            # Determine current KG index
            current_kg_opt = "None"
            if selected_agent_config and selected_agent_config.knowledge_graph_path:
                 if selected_agent_config.knowledge_graph_path in available_kgs:
                     current_kg_opt = selected_agent_config.knowledge_graph_path
            
            kg_index = available_kgs_opts.index(current_kg_opt) if current_kg_opt in available_kgs_opts else 0
            
            selected_kg = st.selectbox("Assign Knowledge Graph", available_kgs_opts, index=kg_index)

        with col2:
            if "forge_sys_prompt" not in st.session_state:
                st.session_state.forge_sys_prompt = default_prompt
            if "sys_prompt_area" not in st.session_state:
                st.session_state.sys_prompt_area = default_prompt
            if "prompt_generation_state" not in st.session_state:
                st.session_state.prompt_generation_state = "IDLE"

            # 1. APPLY phase: Safely update widget states before they are instantiated
            if st.session_state.prompt_generation_state == "APPLY":
                st.session_state.forge_sys_prompt = st.session_state.pending_enhanced_prompt
                st.session_state.sys_prompt_area = st.session_state.pending_enhanced_prompt
                st.session_state.prompt_generation_state = "IDLE"
                st.session_state.llm_lock = False

            def sync_prompt():
                if st.session_state.prompt_generation_state == "IDLE":
                    st.session_state.forge_sys_prompt = st.session_state.sys_prompt_area
            
            def trigger_generation():
                st.session_state.prompt_generation_state = "PREPARE"
                
            # 2. UI Definition
            is_generating = st.session_state.prompt_generation_state in ["PREPARE", "GENERATING"]
            
            st.text_area("System Prompt Template", key="sys_prompt_area", on_change=sync_prompt, disabled=is_generating)
            
            if selected_kg != "None":
                st.button("✨ Auto-Enhance Prompt", on_click=trigger_generation, disabled=is_generating)
            else:
                 st.info("Select a Knowledge Graph to enable Auto-Enhance.")

            # 3. Execution phases
            if st.session_state.prompt_generation_state == "PREPARE":
                # UI is now disabled, rerun to trigger generation
                st.session_state.prompt_generation_state = "GENERATING"
                st.rerun()
            elif st.session_state.prompt_generation_state == "GENERATING":
                if not st.session_state.get("llm_lock"):
                    st.session_state.llm_lock = True
                    
                    import threading
                    try:
                        from streamlit.runtime.scriptrunner import add_script_run_ctx
                    except ImportError:
                        add_script_run_ctx = None
                        
                    def run_llm_background(prompt, kg):
                        try:
                            res = architect.enrich_system_prompt(prompt, kg)
                            if res and res.strip():
                                st.session_state.pending_enhanced_prompt = res
                                st.session_state.prompt_generation_state = "APPLY"
                            else:
                                st.session_state.prompt_generation_state = "IDLE"
                        except Exception as e:
                            logger.log(f"Prompt generation failed: {e}", "ERROR", "App")
                            st.session_state.prompt_generation_state = "IDLE"
                        finally:
                            st.session_state.llm_lock = False
                            
                    t = threading.Thread(target=run_llm_background, args=(st.session_state.forge_sys_prompt, selected_kg))
                    if add_script_run_ctx:
                        add_script_run_ctx(t)
                    t.start()
                    st.rerun() # Immediately rerun to drop into the spinner block
                else:
                    with st.spinner("Refining persona using AI..."):
                        st.info("Generation running in the background. The UI will automatically update when complete.")

        # Save Button
        btn_label = "Update Agent" if mode == "Edit Existing Agent" else "Forge Agent"
        if st.button(btn_label):
            new_kg_path = selected_kg if selected_kg != "None" else ""
            
            # Reuse ID if editing, else new UUID
            agent_id = selected_agent_config.id if selected_agent_config else uuid4()
            
            new_config = AgentConfig(
                id=agent_id,
                name=name_val,
                domain=domain_val,
                knowledge_graph_path=new_kg_path,
                system_prompt_template=st.session_state.forge_sys_prompt,
                interaction_style=style_val,
                color=color_val
            )
            
            SwarmIO.save_agent_config(new_config)
            st.success(f"Agent '{name_val}' saved successfully!")
            time.sleep(1) # Give time to read
            st.rerun()


# --- Tab 2: Swarm Simulation ---
with tab2:
    st.header("Swarm Simulation")
    
    # Inputs
    available_agents = SwarmIO.list_available_agents()
    selected_agent_names = st.multiselect("Select Agents", available_agents)
    universal_prompt = st.text_area("Universal Prompt (Global Goal)", "CUse your expertise to determine how to develop an evolving swarm of multi agernt AI agents to continously improve and adapt to new knowledge and information")
    iterations = st.number_input("Iterations", min_value=1, max_value=50, value=10)
    
    # --- Import Simulation ---
    uploaded_sim = st.file_uploader("Import Simulation Export", type=["json"], key="sim_import")
    if uploaded_sim is not None:
        try:
            # Check if this file is already loaded to prevent reset loop
            file_id = f"{uploaded_sim.name}_{uploaded_sim.size}"
            if "last_imported_sim" not in st.session_state or st.session_state.last_imported_sim != file_id:
                
                import json
                import copy
                
                # 1. Parse JSON
                export_data = json.load(uploaded_sim)
                
                # 2. Validation
                if "steps" not in export_data or not isinstance(export_data["steps"], list):
                    st.error("Invalid simulation file format.")
                else:
                    # 3. Reconstruct History
                    reconstructed_history = []
                    
                    # We need to reconstruct configs. Since export only has ID/Name/Position, 
                    # we try to load full config from SwarmIO. If not found, use defaults.
                    # We do this per step to be safe, but really configs should be constant.
                    # Let's build a cache of configs found in the import to avoid repeated IO.
                    config_cache = {} 
                    
                    with st.spinner("Reconstructing Simulation..."):
                        for step_data in export_data["steps"]:
                            iteration = step_data.get("iteration", 0)
                            current_agent_states = {}
                            current_configs = {}
                            
                            for agent_data in step_data.get("agents", []):
                                a_id = agent_data.get("id")
                                a_name = agent_data.get("name", "Unknown Agent")
                                
                                # Try to load config if not in cache
                                if a_id not in config_cache:
                                    loaded_cfg = SwarmIO.load_agent_config(a_name) # Filename based on name usually
                                    if loaded_cfg and str(loaded_cfg.id) == a_id:
                                         config_cache[a_id] = loaded_cfg
                                    else:
                                        # Fallback: Create a dummy config
                                        config_cache[a_id] = AgentConfig(
                                            id=a_id,
                                            name=a_name,
                                            domain="Imported",
                                            system_prompt_template="",
                                            color="#808080" # Gray for imported/unknown
                                        )
                                
                                # Use cached config
                                cfg = config_cache[a_id]
                                current_configs[a_id] = cfg
                                
                                # Reconstruct State
                                pos_data = agent_data.get("position", {"x": 0, "y": 0, "z": 0})
                                vel_data = agent_data.get("velocity", {"x": 0, "y": 0, "z": 0})
                                
                                current_agent_states[a_id] = AgentRuntimeState(
                                    config_id=a_id,
                                    position=Vector3(**pos_data),
                                    velocity=Vector3(**vel_data),
                                    acceleration=Vector3(x=0, y=0, z=0), # Not in export, assume 0
                                    latest_thought=agent_data.get("thought", "")
                                )
                            
                            snapshot = SwarmSnapshot(
                                iteration=iteration,
                                max_iterations=len(export_data["steps"]),
                                universal_prompt="Imported Simulation", # Metdata lost in export, placeholder
                                agent_configs=current_configs,
                                agent_states=current_agent_states,
                                influence_matrix={} # Not in export
                            )
                            reconstructed_history.append(snapshot)
                    
                    # 4. Update State
                    if len(reconstructed_history) > 0:
                        st.session_state.history = reconstructed_history
                        st.session_state.last_imported_sim = file_id # Mark as loaded
                        st.success(f"Successfully imported {len(reconstructed_history)} steps!")
                        
                        # Optional: Force slider reset
                        st.session_state.step_slider = 0
                        
        except Exception as e:
            st.error(f"Error importing simulation: {e}")
    
    if st.button("Initialize & Run Swarm"):
        if not selected_agent_names:
            st.error("Please select at least one agent.")
        else:
            # Setup Simulation
            st.session_state.history = []
            
            # Load Configs
            configs = {}
            for name in selected_agent_names:
                cfg = SwarmIO.load_agent_config(name)
                if cfg:
                    configs[str(cfg.id)] = cfg
            
            # Initialize States (Random positions in 50x50x50 cube)
            agent_states = {}
            for aid, cfg in configs.items():
                agent_states[aid] = AgentRuntimeState(
                    config_id=str(cfg.id),
                    position=Vector3(
                        x=np.random.uniform(-25, 25),
                        y=np.random.uniform(-25, 25),
                        z=np.random.uniform(-25, 25)
                    ),
                    velocity=Vector3(x=0, y=0, z=0),
                    acceleration=Vector3(x=0, y=0, z=0)
                )
            
            initial_state = SwarmSnapshot(
                iteration=0,
                max_iterations=iterations,
                universal_prompt=universal_prompt,
                agent_configs=configs,
                agent_states=agent_states,
                influence_matrix={}
            )
            
            st.session_state.history = [initial_state]
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Real-time Simulation Console
            st.divider()
            st.subheader("📡 Real-time Agent Comms & Physics")
            sim_console = st.empty()
            
            current_state = initial_state
            
            with st.spinner("Running Simulation..."):
                import threading
                def run_sim_thread(state):
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        loop.run_until_complete(orchestrator.run_full_simulation(state))
                    except Exception as e:
                        import traceback
                        logger.log(f"Simulation Error: {e}", "ERROR", "App")
                        SwarmIO.save_live_state(state.simulation_id, {
                            "status": "error",
                            "message": str(e),
                            "traceback": traceback.format_exc()
                        })
                        
                t = threading.Thread(target=run_sim_thread, args=(initial_state,), daemon=True)
                try:
                    from streamlit.runtime.scriptrunner import add_script_run_ctx
                    add_script_run_ctx(t)
                except ImportError:
                    pass
                t.start()
                
                status_text = st.empty()
                progress_bar = st.progress(0)
                sim_console_container = st.empty()
                
                sim_id = initial_state.simulation_id
                
                last_iteration = -1
                last_message = ""
                
                while True:
                    time.sleep(0.5)
                    live_state = SwarmIO.load_live_state(sim_id)
                    if not live_state:
                        continue
                        
                    status = live_state.get("status", "running")
                    iteration = live_state.get("iteration", 0)
                    max_iter = live_state.get("max_iterations", 1)
                    msg = live_state.get("message", "Running...")
                    
                    if iteration != last_iteration or msg != last_message:
                        status_text.text(msg)
                        progress_bar.progress(iteration / max_iter if max_iter > 0 else 0)
                        
                        snap_data = live_state.get("latest_snapshot")
                        if snap_data:
                            live_data = []
                            for aid, agent in snap_data.get("agent_states", {}).items():
                                cfg = snap_data.get("agent_configs", {}).get(agent.get("config_id"))
                                if not cfg: continue
                                pos = agent.get('position', {'x':0, 'y':0, 'z':0})
                                live_data.append({
                                    "Agent": cfg.get("name", "Unknown Agent"),
                                    "Position": f"({pos.get('x', 0):.1f}, {pos.get('y', 0):.1f}, {pos.get('z', 0):.1f})",
                                    "Thought": agent.get("latest_thought", "...")
                                })
                                
                            with sim_console_container.container():
                                for item in live_data:
                                    col_a, col_b = st.columns([1, 3])
                                    with col_a:
                                        st.markdown(f"**{item['Agent']}**")
                                        st.caption(f"Pos: {item['Position']}")
                                    with col_b:
                                        st.info(item['Thought'])
                                        
                        last_iteration = iteration
                        last_message = msg
                        
                    if status == "complete":
                        st.session_state.history = []
                        for snap_dict in live_state.get("history", []):
                            try:
                                snap = SwarmSnapshot.model_validate(snap_dict)
                                st.session_state.history.append(snap)
                            except Exception as e:
                                pass
                        
                        st.success(f"Simulation Complete! Data exported to: {live_state.get('export_path')}")
                        break
                    elif status == "error":
                        st.error(f"Error during simulation: {live_state.get('message')}")
                        st.code(live_state.get('traceback', ''))
                        break
                
                # --- Synthesis Step ---
                st.divider()
                st.subheader("📝 Consensus Synthesis")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("Generate Research Paper", use_container_width=True):
                        with st.status("Synthesizing Research Paper...", expanded=True) as status:
                            st.write("🧠 Aggregating agent thoughts...")
                            st.write("📝 Drafting content with LLM...")
                            try:
                                synth_loop = asyncio.new_event_loop()
                                asyncio.set_event_loop(synth_loop)
                                paper_content = synth_loop.run_until_complete(orchestrator.synthesize_paper(
                                    st.session_state.history, 
                                    universal_prompt
                                ))
                            finally:
                                synth_loop.close()
                            
                            st.write("💾 Saving to disk...")
                            paper_path = SwarmIO.save_research_paper(paper_content, initial_state.simulation_id)
                            abs_path = os.path.abspath(paper_path)
                            status.update(label="Research Paper Generated!", state="complete", expanded=False)
                            
                        st.success(f"Paper saved to: `{abs_path}`")
                        with st.expander("📄 View Research Paper"):
                            st.markdown(paper_content)
                
                with col2:
                    kg_domain_input = st.text_input("New KG Domain Name", value="Collaborative_KG")
                    if st.button("Generate Collaborative KG", use_container_width=True):
                        # Extract conclusion from export
                        conclusion_text = ""
                        export_path = live_state.get('export_path')
                        if export_path and os.path.exists(export_path):
                            with open(export_path, 'r') as f:
                                export_data = json.load(f)
                                conclusion_text = export_data.get("consensus", {}).get("enriched_conclusion", "")
                        
                        if not conclusion_text:
                            st.error("No consensus found. Please try running the simulation again.")
                        else:
                            with st.status("Extracting Collaborative Knowledge Graph...", expanded=True) as status:
                                st.write("🧠 Parsing simulation consensus...")
                                
                                progress_bar = st.progress(0.0)
                                progress_text = st.empty()
                                
                                def tk_progress_callback(msg, pct):
                                    progress_text.text(msg)
                                    progress_bar.progress(pct)
                                
                                try:
                                    synth_loop = asyncio.new_event_loop()
                                    asyncio.set_event_loop(synth_loop)
                                    kg_filename = synth_loop.run_until_complete(orchestrator.generate_collaborative_kg(
                                        st.session_state.history, 
                                        conclusion_text,
                                        kg_domain_input,
                                        progress_callback=tk_progress_callback
                                    ))
                                finally:
                                    synth_loop.close()
                                
                                status.update(label="Collaborative Knowledge Graph Generated!", state="complete", expanded=False)
                                
                            if "Error" in kg_filename:
                                st.error(kg_filename)
                            else:
                                st.success(f"Knowledge Graph saved successfully: `{kg_filename}`")
                                st.info("You can now assign this Knowledge Graph to an agent in the Agent Forge tab.")

    # Visualization
    if "history" in st.session_state and st.session_state.history:
        history = st.session_state.history
        max_step = len(history) - 1
        
        # --- Visualization Controls ---
        # Calculate global ranges for stability
        all_x, all_y, all_z = [], [], []
        for snap in history:
            for agent in snap.agent_states.values():
                all_x.append(agent.position.x)
                all_y.append(agent.position.y)
                all_z.append(agent.position.z)
        
        if all_x:
            padding = 5
            x_range = [min(all_x)-padding, max(all_x)+padding]
            y_range = [min(all_y)-padding, max(all_y)+padding]
            z_range = [min(all_z)-padding, max(all_z)+padding]
        else:
            x_range, y_range, z_range = [-30, 30], [-30, 30], [-30, 30]

        def get_static_figure(snap, xr, yr, zr):
            """Generates a static 3D figure for a single snapshot."""
            # 1. Agents
            x, y, z, colors, texts = [], [], [], [], []
            for aid, agent in snap.agent_states.items():
                x.append(agent.position.x)
                y.append(agent.position.y)
                z.append(agent.position.z)
                config = snap.agent_configs[agent.config_id]
                colors.append(config.color)
                texts.append(f"{config.name}<br>{config.domain}")
                
            data = []
            data.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=8, color=colors, opacity=0.8),
                text=texts,
                name="Agents"
            ))
            
            # 2. Goal
            data.append(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=10, color='gold', symbol='diamond'),
                name="Goal"
            ))
            
            # 3. Connections
            ix, iy, iz = [], [], []
            threshold = 0.1
            for r_id, row in snap.influence_matrix.items():
                if r_id in snap.agent_states:
                    r_pos = snap.agent_states[r_id].position
                    for s_id, score in row.items():
                        if r_id != s_id and score > threshold and s_id in snap.agent_states:
                            s_pos = snap.agent_states[s_id].position
                            ix.extend([r_pos.x, s_pos.x, None])
                            iy.extend([r_pos.y, s_pos.y, None])
                            iz.extend([r_pos.z, s_pos.z, None])
            
            if ix:
                data.append(go.Scatter3d(
                    x=ix, y=iy, z=iz,
                    mode='lines',
                    line=dict(color='rgba(255,255,255,0.2)', width=3),
                    showlegend=False,
                    name="Influence"
                ))
            
            fig = go.Figure(
                data=data,
                layout=go.Layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    scene=dict(
                        xaxis=dict(range=xr),
                        yaxis=dict(range=yr),
                        zaxis=dict(range=zr),
                        aspectmode='cube'
                    ),
                    margin=dict(l=0, r=0, b=0, t=0),
                    font=dict(color="white")
                )
            )
            return fig

        def render_thoughts(snap):
            """Renders the active thoughts dataframe."""
            data = []
            for aid, agent in snap.agent_states.items():
                cfg = snap.agent_configs[agent.config_id]
                data.append({
                    "Agent": cfg.name,
                    "Domain": cfg.domain,
                    "Active Thought": agent.latest_thought or "..."
                })
            return pd.DataFrame(data)

        # Controls
        col_ctrl1, col_ctrl2 = st.columns([1, 4])
        with col_ctrl1:
            play_btn = st.button("▶ Play Simulation", use_container_width=True)
        with col_ctrl2:
            if max_step > 0:
                step_idx = st.slider("Timeline", 0, max_step, max_step, key="sim_slider")
            else:
                st.slider("Timeline", 0, 1, 0, disabled=True, key="sim_slider_disabled")
                step_idx = 0

        # Placeholders
        chart_placeholder = st.empty()
        st.caption("Active Thoughts")
        thoughts_placeholder = st.empty()

        # Playback Logic
        if play_btn:
            for i in range(len(history)):
                snap = history[i]
                fig = get_static_figure(snap, x_range, y_range, z_range)
                chart_placeholder.plotly_chart(fig, width='stretch', key=f"chart_{i}")
                
                df = render_thoughts(snap)
                thoughts_placeholder.dataframe(
                    df, 
                    width='stretch', 
                    hide_index=True,
                    column_config={
                        "Active Thought": st.column_config.TextColumn(width="large")
                    }
                )
                time.sleep(0.1) # Speed of playback
            # Final state update to match slider? 
            # Ideally we'd update the slider too but we can't easily.
        else:
            # Static View based on Slider
            current_snap = history[step_idx]
            fig = get_static_figure(current_snap, x_range, y_range, z_range)
            chart_placeholder.plotly_chart(fig, width='stretch')
            
            df = render_thoughts(current_snap)
            thoughts_placeholder.dataframe(
                df, 
                width='stretch', 
                hide_index=True,
                column_config={
                    "Active Thought": st.column_config.TextColumn(width="large")
                }
            )
        
        # Metrics - Show latest state by default
        snapshot = history[-1]
        
        st.markdown(f"**Latest Iteration:** {snapshot.iteration}")
        st.markdown("**Active Thoughts:**")
        cols = st.columns(2)
        for i, (aid, agent) in enumerate(snapshot.agent_states.items()):
            config = snapshot.agent_configs[agent.config_id]
            with cols[i % 2]:
                with st.expander(f"{config.name} ({config.domain})", expanded=True):
                    if agent.latest_thought:
                        st.write(agent.latest_thought)
                    else:
                        st.write("*Thinking...*")

# --- Tab 3: Inspector ---
with tab3:
    st.header("Inspector")
    
    if "history" in st.session_state and st.session_state.history:
        max_insp_step = len(st.session_state.history)-1
        
        if max_insp_step > 0:
            step_idx = st.slider("Step Index", 0, max_insp_step, max_insp_step, key="inspector_slider")
        else:
            step_idx = 0
            st.write(f"Step Index: {step_idx}")

        snap = st.session_state.history[step_idx]
        
        agent_names = [cfg.name for cfg in snap.agent_configs.values()]
        selected_agent_name = st.selectbox("Inspect Agent", agent_names)
        
        # Find ID
        target_id = None
        for aid, cfg in snap.agent_configs.items():
            if cfg.name == selected_agent_name:
                target_id = str(cfg.id) 
                break
        
        if target_id and target_id in snap.agent_states:
            agent_state = snap.agent_states[target_id]
            st.json(agent_state.model_dump())

@st.dialog("Node Details")
def show_node_details(label, summary, connected_labels=None):
    st.markdown(f"**Label:** {label}")
    st.markdown(f"**Summary:** {summary}")
    if connected_labels:
        st.markdown("**Connected Nodes:**")
        for cl in connected_labels:
            st.markdown(f"- {cl}")

# --- Tab 4: Knowledge Graph Viewer ---
with tab4:
    st.header("Knowledge Graph Viewer")
    kgs = SwarmIO.list_available_kgs()
    if not kgs:
         st.info("No Knowledge Graphs found.")
    else:
         selected_kg = st.selectbox("Select a Knowledge Graph to View", kgs, key="kg_viewer_select")
         if selected_kg:
             kg_path = os.path.join("swarm_data", "knowledge_graphs", selected_kg)
             try:
                 import json
                 import numpy as np
                 with open(kg_path, 'r') as f:
                     kg_data = json.load(f)
                 
                 nodes = kg_data.get("nodes", {})
                 node_id_list = list(nodes.keys())
                 
                 if not nodes:
                     st.warning("Empty Knowledge Graph")
                 else:
                     # Fibonacci Sphere 3D Layout (No Jitter)
                     points = {}
                     n_nodes = len(node_id_list)
                     offset = 2.0 / max(1, n_nodes)
                     increment = np.pi * (3.0 - np.sqrt(5.0)) # golden angle

                     for i, nid in enumerate(node_id_list):
                         y = ((i * offset) - 1) + (offset / 2)
                         r = np.sqrt(1 - pow(y, 2))
                         phi = ((i + 1) % n_nodes) * increment

                         r_scale = 15
                         x = np.cos(phi) * r * r_scale
                         z = np.sin(phi) * r * r_scale
                         y_scaled = y * r_scale
                         
                         points[nid] = (x, y_scaled, z)
                     
                     x_vals, y_vals, z_vals, texts = [], [], [], []
                     for nid in node_id_list:
                         px, py, pz = points[nid]
                         x_vals.append(px)
                         y_vals.append(py)
                         z_vals.append(pz)
                         texts.append(nodes[nid].get("label", nid))
                     
                     # Edges
                     edge_x, edge_y, edge_z = [], [], []
                     for nid, node_data in nodes.items():
                         children = node_data.get("children", [])
                         for child_id in children:
                             if child_id in points:
                                 edge_x.extend([points[nid][0], points[child_id][0], None])
                                 edge_y.extend([points[nid][1], points[child_id][1], None])
                                 edge_z.extend([points[nid][2], points[child_id][2], None])
                     
                     data = []
                     if edge_x:
                         data.append(go.Scatter3d(
                             x=edge_x, y=edge_y, z=edge_z,
                             mode='lines',
                             line=dict(color='rgba(255,255,255,0.4)', width=2),
                             hoverinfo='none',
                             showlegend=False
                         ))
                     
                     data.append(go.Scatter3d(
                         x=x_vals, y=y_vals, z=z_vals,
                         mode='markers',
                         marker=dict(size=8, color='#00CCFF', opacity=0.8),
                         text=texts,
                         hoverinfo='text',
                         name="Nodes"
                     ))
                     
                     fig = go.Figure(data=data, layout=go.Layout(
                         template="plotly_dark",
                         paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)',
                         scene=dict(aspectmode='cube'),
                         margin=dict(l=0, r=0, b=0, t=0)
                     ))
                     
                     from streamlit_plotly_events import plotly_events
                     selection = plotly_events(fig, click_event=True, hover_event=False, key=f"kg_plot_{selected_kg}")
                     
                     if selection and len(selection) > 0:
                         st.session_state["pause_autorefresh"] = True
                         point_idx = selection[0].get("pointNumber", selection[0].get("pointIndex", 0))
                         selected_node_id = node_id_list[min(point_idx, len(node_id_list)-1)]
                         
                         connected_ids = set()
                         for child in nodes.get(selected_node_id, {}).get("children", []):
                             connected_ids.add(child)
                         for nid, ndata in nodes.items():
                             if selected_node_id in ndata.get("children", []):
                                 connected_ids.add(nid)
                                 
                         connected_labels = [nodes.get(cid, {}).get("label", cid) for cid in connected_ids]
                         
                         show_node_details(nodes[selected_node_id].get("label", selected_node_id), nodes[selected_node_id].get("summary", "No summary available."), connected_labels)
                     else:
                         st.session_state["pause_autorefresh"] = False
             except Exception as e:
                 st.error(f"Error loading Knowledge Graph: {e}")

# --- Developer Console ---
st.divider()
with st.expander("👩‍💻 Developer Console", expanded=True):
    logs = logger.get_logs()
    
    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("Clear Console", key="clear_logs"):
            logger.clear()
            st.rerun()
            
    if logs:
        # Reverse to show newest first
        df_logs = pd.DataFrame(logs).iloc[::-1]
        
        # Color coding
        def color_log(val):
            color = 'black'
            if val == 'ERROR': color = 'red'
            elif val == 'WARNING': color = 'orange'
            elif val == 'SUCCESS': color = 'green'
            return f'color: {color}'
            
        # Custom Log Rendering
        for idx, row in df_logs.iterrows():
            with st.container():
                # Color coding based on level
                level = row['level']
                msg = f"**[{row['timestamp']}] [{row['source']}]** {row['message']}"
                
                if level == 'ERROR':
                    st.error(msg, icon="🚨")
                elif level == 'WARNING':
                    st.warning(msg, icon="⚠️")
                elif level == 'SUCCESS':
                    st.success(msg, icon="✅")
                else:
                    st.info(msg, icon="ℹ️")
    else:
        st.info("System Ready. Waiting for events...")

    # Poll for updates
    try:
        from streamlit_autorefresh import st_autorefresh
        if not st.session_state.get("pause_autorefresh", False):
            st_autorefresh(interval=2000, key="console_refresh")
    except ImportError:
        st.warning("Install `streamlit-autorefresh` for real-time log updates.")
