import streamlit as st
import json
import copy
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

from core.swarm_schema import AgentConfig, SwarmSnapshot, AgentRuntimeState, Vector3
from core.swarm_io import SwarmIO
from core.swarm_orchestrator import SwarmOrchestrator
from core.settings_manager import SettingsManager, GlobalSettings, PhysicsSettings, PhysicsWeights, KnowledgeGraphSettings
from core.knowledge_graph import KnowledgeArchitect
from core.swarm_logger import SwarmLogger
from core.DOAB_client import DOABClient
from core.swarm_gemini import GeminiLLM

# Page Config
st.set_page_config(page_title="Swarm Intelligence Control Tower", layout="wide", page_icon="🐝")

# Custom UI Styles injection
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Apply premium typography font */
    html, body, [class*="css"], [class*="st-"] {
        font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Sleek gradient background for main container */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a, #0b0f19) !important;
    }

    /* Stylish Cards */
    .custom-card {
        background: rgba(30, 41, 59, 0.4);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .custom-card:hover {
        border-color: rgba(0, 242, 254, 0.3);
    }

    /* Sidebar container restyling */
    section[data-testid="stSidebar"] {
        background-color: #080a0f !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Streamlit Tab Custom Styling */
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: rgba(255, 255, 255, 0.55) !important;
        background-color: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 10px 16px !important;
        transition: all 0.3s ease !important;
    }
    button[data-baseweb="tab"]:hover {
        color: #00f2fe !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f2fe !important;
        border-bottom: 2px solid #00f2fe !important;
        text-shadow: 0 0 8px rgba(0, 242, 254, 0.4);
    }
    
    /* Horizontal Dividers */
    hr {
        margin: 1.5rem 0 !important;
        border-color: rgba(255, 255, 255, 0.08) !important;
    }

    /* Expanders styling */
    div[data-testid="stExpander"] {
        background: rgba(30, 41, 59, 0.25) !important;
        border: 1px solid rgba(255, 255, 255, 0.06) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 10px rgba(0,0,0,0.15) !important;
    }

    /* Custom scrollable terminal console styling */
    .terminal-console {
        background-color: #05070a !important;
        border: 1px solid rgba(0, 242, 254, 0.15) !important;
        border-radius: 8px !important;
        font-family: 'Courier New', Courier, monospace !important;
        padding: 15px !important;
        height: 550px !important;
        overflow-y: auto !important;
        color: #e5e7eb !important;
        box-shadow: inset 0 2px 10px rgba(0,0,0,0.9) !important;
    }
    .terminal-line {
        padding: 6px 0 !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.03) !important;
        display: flex !important;
        align-items: flex-start !important;
        font-size: 0.85rem !important;
    }
    .terminal-time {
        color: #6b7280 !important;
        margin-right: 10px !important;
        flex-shrink: 0 !important;
    }
    .terminal-badge {
        font-weight: bold !important;
        padding: 1px 6px !important;
        border-radius: 4px !important;
        font-size: 0.75rem !important;
        text-transform: uppercase !important;
        margin-right: 10px !important;
        flex-shrink: 0 !important;
    }
    .badge-error { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-warning { background-color: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-success { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-info { background-color: rgba(59, 130, 246, 0.15); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.3); }
    
    .terminal-source {
        color: #a78bfa !important;
        margin-right: 10px !important;
        font-weight: 600 !important;
        flex-shrink: 0 !important;
    }
    .terminal-msg {
        white-space: pre-wrap !important;
        word-break: break-all !important;
    }

    /* Tag cards */
    .kg-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 8px;
        padding: 12px 16px;
        margin: 6px 0;
        transition: all 0.3s ease;
    }
    .kg-card:hover {
        background: rgba(0, 242, 254, 0.04);
        border-color: rgba(0, 242, 254, 0.25);
    }
</style>
""", unsafe_allow_html=True)

# Page Config (already set at the top of the file)

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

# Caching helpers for disk listing operations to prevent clunky UI and JSON parsing overhead
@st.cache_data
def cached_list_agents():
    return SwarmIO.list_available_agents()

@st.cache_data
def cached_list_kgs():
    return SwarmIO.list_available_kgs()

@st.cache_data
def cached_load_all_agents():
    names = SwarmIO.list_available_agents()
    configs = []
    for name in names:
        cfg = SwarmIO.load_agent_config(name)
        if cfg:
            configs.append(cfg)
    return configs

# --- Global UI Helpers and Fragments ---

def render_thoughts(snap):
    data = []
    for aid, agent in snap.agent_states.items():
        cfg = snap.agent_configs[agent.config_id]
        data.append({
            "Agent": cfg.name,
            "Domain": cfg.domain,
            "Active Thought": agent.latest_thought or "..."
        })
    return pd.DataFrame(data)

def build_animated_figure(history, xr, yr, zr, frame_duration_ms=100, num_interp=5):
    if not history:
        return go.Figure()
    
    first_snap = history[0]
    agent_order = list(first_snap.agent_states.keys())
    colors = []
    texts = []
    for aid in agent_order:
        agent = first_snap.agent_states[aid]
        cfg = first_snap.agent_configs[agent.config_id]
        colors.append(cfg.color)
        texts.append(f"{cfg.name}<br>{cfg.domain}")
    
    all_frames = []
    frame_labels = []
    
    for step_idx, snap in enumerate(history):
        x, y, z = [], [], []
        for aid in agent_order:
            if aid in snap.agent_states:
                a = snap.agent_states[aid]
                x.append(a.position.x)
                y.append(a.position.y)
                z.append(a.position.z)
            else:
                x.append(0); y.append(0); z.append(0)
        
        if step_idx < len(history) - 1:
            next_snap = history[step_idx + 1]
            nx, ny, nz = [], [], []
            for aid in agent_order:
                if aid in next_snap.agent_states:
                    a = next_snap.agent_states[aid]
                    nx.append(a.position.x)
                    ny.append(a.position.y)
                    nz.append(a.position.z)
                else:
                    nx.append(0); ny.append(0); nz.append(0)
            
            for t in range(num_interp + 1):
                alpha = t / (num_interp + 1)
                ix = [x[j] + (nx[j] - x[j]) * alpha for j in range(len(agent_order))]
                iy = [y[j] + (ny[j] - y[j]) * alpha for j in range(len(agent_order))]
                iz = [z[j] + (nz[j] - z[j]) * alpha for j in range(len(agent_order))]
                
                label = f"{step_idx}" if t == 0 else f"{step_idx}.{t}"
                frame_labels.append(label)
                all_frames.append(go.Frame(
                    data=[
                        go.Scatter3d(x=ix, y=iy, z=iz,
                            mode='markers',
                            marker=dict(size=8, color=colors, opacity=0.8),
                            text=texts, name="Agents"),
                        go.Scatter3d(x=[0], y=[0], z=[0],
                            mode='markers',
                            marker=dict(size=10, color='gold', symbol='diamond'),
                            name="Goal")
                    ],
                    name=label
                ))
        else:
            label = f"{step_idx}"
            frame_labels.append(label)
            all_frames.append(go.Frame(
                data=[
                    go.Scatter3d(x=x, y=y, z=z,
                        mode='markers',
                        marker=dict(size=8, color=colors, opacity=0.8),
                        text=texts, name="Agents"),
                    go.Scatter3d(x=[0], y=[0], z=[0],
                        mode='markers',
                        marker=dict(size=10, color='gold', symbol='diamond'),
                        name="Goal")
                ],
                name=label
            ))
    
    init_x = [first_snap.agent_states[aid].position.x for aid in agent_order]
    init_y = [first_snap.agent_states[aid].position.y for aid in agent_order]
    init_z = [first_snap.agent_states[aid].position.z for aid in agent_order]
    
    slider_steps = []
    for i in range(len(history)):
        label = f"{i}"
        slider_steps.append(dict(
            args=[[label], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate", "transition": {"duration": 0}}],
            label=str(i),
            method="animate"
        ))
    
    transition_ms = max(10, frame_duration_ms // 2)
    
    fig = go.Figure(
        data=[
            go.Scatter3d(x=init_x, y=init_y, z=init_z,
                mode='markers',
                marker=dict(size=8, color=colors, opacity=0.8),
                text=texts, name="Agents"),
            go.Scatter3d(x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=10, color='gold', symbol='diamond'),
                name="Goal")
        ],
        frames=all_frames,
        layout=go.Layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(range=xr), yaxis=dict(range=yr), zaxis=dict(range=zr),
                aspectmode='cube'
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            font=dict(color="white"),
            height=500,
            updatemenus=[dict(
                type="buttons",
                showactive=False,
                x=0.0, y=1.08,
                xanchor="left", yanchor="top",
                buttons=[
                    dict(label="▶ Play",
                        method="animate",
                        args=[None, {
                            "frame": {"duration": frame_duration_ms, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": transition_ms}
                        }]),
                    dict(label="⏸ Pause",
                        method="animate",
                        args=[[None], {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }])
                ]
            )],
            sliders=[dict(
                active=0,
                steps=slider_steps,
                x=0.0, y=0.0, len=1.0,
                xanchor="left",
                currentvalue=dict(prefix="Step: ", visible=True, xanchor="center"),
                transition=dict(duration=transition_ms)
            )]
        )
    )
    return fig

@st.fragment(run_every="1s")
def sim_status_fragment():
    sim_id = st.session_state.sim_id
    live_state = SwarmIO.load_live_state(sim_id)
    if not live_state:
        st.info("Initializing...")
        return
    
    status = live_state.get("status", "running")
    iteration = live_state.get("iteration", 0)
    max_iter = live_state.get("max_iterations", 1)
    msg = live_state.get("message", "Running...")
    
    st.text(msg)
    st.progress(iteration / max_iter if max_iter > 0 else 0)
    
    # Real-time 3D Visualization
    vis_history = live_state.get("vis_history", [])
    if vis_history:
        latest_frame = vis_history[-1]
        agents = latest_frame.get("agents", [])
        if agents:
            x = [a["position"]["x"] for a in agents]
            y = [a["position"]["y"] for a in agents]
            z = [a["position"]["z"] for a in agents]
            colors = [a.get("color", "#808080") for a in agents]
            texts = [f"{a['name']}<br>{a.get('domain','')}" for a in agents]
            
            padding = 5
            all_x, all_y, all_z = [], [], []
            for frame in vis_history:
                for a in frame.get("agents", []):
                    all_x.append(a["position"]["x"])
                    all_y.append(a["position"]["y"])
                    all_z.append(a["position"]["z"])
            xr = [min(all_x)-padding, max(all_x)+padding] if all_x else [-30, 30]
            yr = [min(all_y)-padding, max(all_y)+padding] if all_y else [-30, 30]
            zr = [min(all_z)-padding, max(all_z)+padding] if all_z else [-30, 30]
            
            data = []
            if len(vis_history) > 1:
                agent_ids = [a["id"] for a in agents]
                for idx, aid in enumerate(agent_ids):
                    trail_x, trail_y, trail_z = [], [], []
                    for frame in vis_history:
                        for a in frame.get("agents", []):
                            if a["id"] == aid:
                                trail_x.append(a["position"]["x"])
                                trail_y.append(a["position"]["y"])
                                trail_z.append(a["position"]["z"])
                    data.append(go.Scatter3d(
                        x=trail_x, y=trail_y, z=trail_z,
                        mode='lines',
                        line=dict(color=colors[idx], width=2),
                        opacity=0.3,
                        showlegend=False
                    ))
            
            data.append(go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                marker=dict(size=8, color=colors, opacity=0.9),
                text=texts, name="Agents"
            ))
            data.append(go.Scatter3d(
                x=[0], y=[0], z=[0],
                mode='markers',
                marker=dict(size=10, color='gold', symbol='diamond'),
                name="Goal"
            ))
            
            fig = go.Figure(data=data, layout=go.Layout(
                template="plotly_dark",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                scene=dict(
                    xaxis=dict(range=xr), yaxis=dict(range=yr), zaxis=dict(range=zr),
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, b=0, t=0),
                font=dict(color="white"),
                height=450
            ))
            st.plotly_chart(fig, width='stretch', key="live_simulation_chart")
            
            # Realtime tabs for details
            vis_tab1, vis_tab2 = st.tabs(["💭 Agent Thoughts", "🤖 Orchestration Logs"])
            with vis_tab1:
                thought_data = [{"Agent": a["name"], "Thought": a.get("thought", "...")} for a in agents]
                st.dataframe(pd.DataFrame(thought_data), width='stretch', height=180, hide_index=True)
            with vis_tab2:
                orchestration_logs = live_state.get("orchestration_logs", [])
                if orchestration_logs:
                    for log in orchestration_logs:
                        st.markdown(f"**Phase {log['phase']}: Allocating {', '.join(log['active_agents'])}**")
                        st.markdown(f"*Reasoning:* {log['reasoning']}")
                        st.markdown(f"*Instructions:* {log['delegation_instructions']}")
                        st.divider()
                else:
                    st.info("No orchestration logs generated yet.")
    
    if status == "complete":
        st.session_state.history = []
        for snap_dict in live_state.get("history", []):
            try:
                snap = SwarmSnapshot.model_validate(snap_dict)
                st.session_state.history.append(snap)
            except Exception as e:
                pass
        
        st.success(f"Simulation Complete! Data exported to: {live_state.get('export_path')}")
        st.session_state.sim_active = False
        st.session_state.sim_finished = True
        st.session_state.sim_final_state = live_state
        st.rerun()
    elif status == "error":
        st.error(f"Error during simulation: {live_state.get('message')}")
        st.code(live_state.get('traceback', ''))
        st.session_state.sim_active = False
        st.rerun()

@st.fragment
def render_playback_panel(history, x_range, y_range, z_range, max_step):
    st.write("### 🎬 Simulation Playback")
    col_speed, col_interp, col_metric = st.columns([2, 2, 2])
    with col_speed:
        speed = st.slider("⚡ Playback Speed", min_value=0.25, max_value=4.0, value=1.0, step=0.25, key="playback_speed")
    with col_interp:
        num_interp = st.slider("🔀 Smoothness", min_value=1, max_value=10, value=5, key="interp_frames")
    with col_metric:
        st.metric("Total Steps", max_step + 1)
        st.caption(f"Interpolated: {(max_step) * (num_interp + 1) + 1}")

    base_duration_ms = 100
    frame_duration_ms = int(base_duration_ms / speed)
    
    animated_fig = build_animated_figure(history, x_range, y_range, z_range, frame_duration_ms, num_interp)
    st.plotly_chart(animated_fig, width='stretch', key="animated_sim_chart")
    
    st.divider()
    
    # Interactive details tabs below chart
    hist_tab1, hist_tab2, hist_tab3 = st.tabs(["💭 Thought Inspector", "🤖 Final Orchestration Log", "📊 Stats"])
    with hist_tab1:
        st.caption("💭 Thought Inspector (Drag to select a step to inspect agent thoughts)")
        if max_step > 0:
            thought_step = st.slider("Inspect Step", 0, max_step, max_step, key="thought_slider")
        else:
            thought_step = 0
        
        current_snap = history[thought_step]
        df = render_thoughts(current_snap)
        st.dataframe(
            df, 
            width='stretch',
            height=240,
            hide_index=True,
            column_config={
                "Active Thought": st.column_config.TextColumn(width="large")
            }
        )
    with hist_tab2:
        live_state = st.session_state.get("sim_final_state", {})
        orchestration_logs = live_state.get("orchestration_logs", [])
        if orchestration_logs:
            for log in orchestration_logs:
                st.markdown(f"**Phase {log['phase']}: Allocating {', '.join(log['active_agents'])}**")
                st.markdown(f"**Reasoning:** {log['reasoning']}")
                st.markdown(f"**Instructions:** {log['delegation_instructions']}")
                st.divider()
        else:
            st.info("No orchestration logs available for this simulation.")
    with hist_tab3:
        snapshot = history[-1]
        st.markdown(f"**Latest Iteration:** {snapshot.iteration}")
        st.markdown(f"**Active Agents Count:** {len(snapshot.agent_states)}")
        st.markdown(f"**Simulation ID:** `{st.session_state.get('sim_id', 'N/A')}`")

# Check for API Key (provider-specific)
_init_provider = settings_manager.get_settings().llm_provider.lower()
if _init_provider == "gemini" and not settings_manager.get_api_key():
    st.error("⚠️ Google API Key is missing! Please set `GOOGLE_API_KEY` environment variable or add `google_api_key` to `swarm_settings.json`.")
elif _init_provider == "nvidia" and not settings_manager.get_nvidia_api_key():
    st.error("⚠️ Nvidia API Key is missing! Please set `NVIDIA_API_KEY` environment variable or add it to `swarm_settings.json`.")


# --- Sidebar: Collapsible Configuration Panel ---
current_settings = settings_manager.get_settings()
physics = current_settings.physics
weights = physics.weights

# 1. Model & Memory Settings
with st.sidebar.expander("🧠 Model & Memory Settings", expanded=True):
    current_ltm_setting = current_settings.knowledge_graph.search_long_term_memory if current_settings.knowledge_graph else True
    new_ltm_setting = st.checkbox("Search Long Term Memory", value=current_ltm_setting)
    
    if new_ltm_setting != current_ltm_setting:
        updated_kg_settings = current_settings.knowledge_graph.model_copy(update={
            "search_long_term_memory": new_ltm_setting
        }) if current_settings.knowledge_graph else KnowledgeGraphSettings(search_long_term_memory=new_ltm_setting)
        
        updated_settings = current_settings.model_copy(update={
            "knowledge_graph": updated_kg_settings
        })
        settings_manager.update_settings(updated_settings)
        settings_manager.save_settings()
        st.success("Memory settings updated!")
        st.rerun()
        
    st.divider()
    
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
    
    selected_provider = st.selectbox(
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
        
    if selected_provider == "gemini":
        # Populate Gemini models via list_models()
        api_key = settings_manager.get_api_key()
        available_gemini_models = GeminiLLM.list_models(api_key=api_key)
        
        # Ensure current model is in the list to avoid ValueError in index()
        if suggested_model not in available_gemini_models:
            available_gemini_models = [suggested_model] + available_gemini_models
        
        selected_model = st.selectbox(
            "Model Name", 
            options=available_gemini_models, 
            index=available_gemini_models.index(suggested_model)
        )
    else:
        selected_model = st.text_input("Model Name", value=suggested_model)
        
    # Show rate limits for selected provider
    rl = settings_manager.get_rate_limiting_for_provider(selected_provider)
    st.caption(f"**Rate Limits ({PROVIDER_LABELS.get(selected_provider, selected_provider)}):**")
    st.caption(f"  RPM: {rl.requests_per_minute:,}  |  TPM: {rl.tokens_per_minute:,}  |  Delay: {rl.request_delay_seconds}s")
    
    # Apply changes if provider or model changed
    if selected_provider != current_provider or selected_model != current_model:
        if st.button("⚡ Apply Model Change", width='stretch'):
            updated_settings = current_settings.model_copy(update={
                "llm_provider": selected_provider,
                "model_name": selected_model
            })
            settings_manager.update_settings(updated_settings)
            settings_manager.save_settings()
            
            # Hotswap the orchestrator's provider
            orchestrator.switch_provider(selected_provider, selected_model)
            
            st.success(f"Switched to {PROVIDER_LABELS.get(selected_provider, selected_provider)}: {selected_model}")
            st.cache_resource.clear()
            st.rerun()

# 2. Physics Engine Tuner Settings
with st.sidebar.expander("⚙️ Physics Engine Tuner", expanded=False):
    st.caption("Adjust the flocking rules (Boids algorithm) guiding agent movement in 3D space.")
    new_sep = st.slider("Separation Weight", 0.0, 5.0, weights.separation, 0.1)
    new_ali = st.slider("Alignment Weight", 0.0, 5.0, weights.alignment, 0.1)
    new_coh = st.slider("Cohesion Weight", 0.0, 5.0, weights.cohesion, 0.1)
    new_goal = st.slider("Goal Seeking Weight", 0.0, 5.0, weights.goal_seeking, 0.1)
    new_vis = st.slider("Visual Range", 1.0, 100.0, physics.visual_range, 1.0)
    new_speed = st.slider("Max Speed", 1.0, 20.0, physics.max_speed, 0.5)

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


# --- Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Agent Forge", "Swarm Simulation", "Inspector", "Knowledge Graph Viewer", "Developer Console"])

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
                            cached_list_kgs.clear()
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
                        cached_list_kgs.clear()
                        
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
                
                @st.fragment(run_every="1s")
                def doab_status_fragment():
                    if st.session_state.doab_pipeline_state == "RUNNING":
                        st.info(f"⏳ {st.session_state.doab_status}")
                        st.progress(st.session_state.doab_progress / 100)
                    else:
                        st.rerun()
                
                doab_status_fragment()
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
        st.subheader("📊 Available Knowledge Graphs")
        kgs = cached_list_kgs()
        if kgs:
            cols = st.columns(3)
            for idx, kg in enumerate(kgs):
                col = cols[idx % 3]
                display_name = kg.replace("_KG.json", " KG").replace("_kg.json", " KG").replace(".json", "")
                display_name = display_name.replace("_", " ")
                col.markdown(f"""
                <div class="kg-card">
                    <div style="font-weight: 600; color: #00f2fe; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
                        <span>📊</span> {display_name}
                    </div>
                    <div style="font-size: 0.75rem; color: rgba(255,255,255,0.45); margin-top: 6px; font-family: monospace; word-break: break-all;">
                        {kg}
                    </div>
                </div>
                """, unsafe_allow_html=True)
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
            agents_list = cached_list_agents()
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
            available_kgs = cached_list_kgs()
            
            # Determine current KG selection
            current_kg_opts = []
            if selected_agent_config and hasattr(selected_agent_config, 'knowledge_graph_paths'):
                current_kg_opts = [kg for kg in selected_agent_config.knowledge_graph_paths if kg in available_kgs]
            elif selected_agent_config and hasattr(selected_agent_config, 'knowledge_graph_path') and selected_agent_config.knowledge_graph_path in available_kgs:
                current_kg_opts = [selected_agent_config.knowledge_graph_path]
            
            selected_kgs = st.multiselect("Assign Knowledge Graphs", available_kgs, default=current_kg_opts)

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
            
            st.text_area("System Prompt Template", key="sys_prompt_area", on_change=sync_prompt, disabled=False)
            
            if selected_kgs:
                st.button("✨ Auto-Enhance Prompt", on_click=trigger_generation, disabled=is_generating)
            else:
                 st.info("Select at least one Knowledge Graph to enable Auto-Enhance.")

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
                        
                    def run_llm_background(prompt, kgs):
                        try:
                            res = architect.enrich_system_prompt(prompt, kgs[0])
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
                            
                    t = threading.Thread(target=run_llm_background, args=(st.session_state.forge_sys_prompt, selected_kgs))
                    if add_script_run_ctx:
                        add_script_run_ctx(t)
                    t.start()
                    st.rerun() # Immediately rerun to drop into the spinner block
                else:
                    @st.fragment(run_every="1s")
                    def prompt_status_fragment():
                        if st.session_state.prompt_generation_state == "GENERATING":
                            with st.spinner("Refining persona using AI..."):
                                st.info("Generation running in the background. The UI will automatically update when complete.")
                        else:
                            st.rerun()
                    prompt_status_fragment()

        # Save Button
        btn_label = "Update Agent" if mode == "Edit Existing Agent" else "Forge Agent"
        if st.button(btn_label):
            # Reuse ID if editing, else new UUID
            agent_id = selected_agent_config.id if selected_agent_config else uuid4()
            
            new_config = AgentConfig(
                id=agent_id,
                name=name_val,
                domain=domain_val,
                knowledge_graph_paths=selected_kgs,
                system_prompt_template=st.session_state.forge_sys_prompt,
                interaction_style=style_val,
                color=color_val
            )
            
            SwarmIO.save_agent_config(new_config)
            cached_list_agents.clear()
            cached_load_all_agents.clear()
            st.success(f"Agent '{name_val}' saved successfully!")
            time.sleep(1) # Give time to read
            st.rerun()

        if mode == "Edit Existing Agent" and current_agent_selection:
            st.divider()
            st.subheader("🧠 Long Term Memory")
            if st.button("Synthesize Thoughts", width='stretch'):
                thoughts_file = SwarmIO.get_agent_thoughts_file(current_agent_selection)
                if os.path.exists(thoughts_file):
                    with st.status(f"Synthesizing Long Term Memory for {current_agent_selection}...", expanded=True) as status:
                        synth_loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(synth_loop)
                        try:
                            safe_name = "".join([c for c in current_agent_selection if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "_")
                            kg_domain = f"agent_{safe_name}_ltm_kg"
                            output_path = SwarmIO.get_agent_ltm_kg_path(current_agent_selection)
                            
                            print(f"\n[LTM SYNTHESIS] Starting synthesis for agent: {current_agent_selection}")
                            print(f"[LTM SYNTHESIS] Reading thoughts from: {thoughts_file}")
                            print(f"[LTM SYNTHESIS] Generating Knowledge Graph...")
                            
                            kg_filename = synth_loop.run_until_complete(
                                architect.build_graph_async([thoughts_file], kg_domain, output_filepath=output_path)
                            )
                            
                            print(f"[LTM SYNTHESIS] Successfully saved to: {output_path}\n")
                            
                            st.write(f"✅ Generated {kg_filename}")
                            status.update(label="Long Term Memory Synthesized!", state="complete", expanded=False)
                        finally:
                            synth_loop.close()
                else:
                    st.warning(f"No thoughts file found for {current_agent_selection}.")

        if "global_synth_state" not in st.session_state:
            st.session_state.global_synth_state = "IDLE"
            st.session_state.global_synth_status = ""
            st.session_state.global_synth_progress = 0
            st.session_state.global_synth_results = []

        st.divider()
        st.subheader("🌐 Global Swarm Memory")
        
        def run_global_synth_thread(agents_to_process):
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
                    st.session_state.global_synth_state = "RUNNING"
                    st.session_state.global_synth_progress = 0
                    st.session_state.global_synth_results = []
                    
                    total = len(agents_to_process)
                    st.session_state.global_synth_status = f"Processing {total} agents in parallel..."
                    
                    async def process_agent(agent_name):
                        thoughts_file = SwarmIO.get_agent_thoughts_file(agent_name)
                        if not os.path.exists(thoughts_file):
                            st.session_state.global_synth_results.append(f"⏭️ No thoughts file for {agent_name}, skipping.")
                        else:
                            safe_name = "".join([c for c in agent_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "_")
                            kg_domain = f"agent_{safe_name}_ltm_kg"
                            output_path = SwarmIO.get_agent_ltm_kg_path(agent_name)
                            
                            logger.log(f"Starting LTM synthesis for: {agent_name}", "INFO", "App")
                            
                            try:
                                kg_filename = await architect.build_graph_async([thoughts_file], kg_domain, output_filepath=output_path)
                                st.session_state.global_synth_results.append(f"✅ Generated {kg_filename} for {agent_name}")
                            except Exception as e:
                                st.session_state.global_synth_results.append(f"❌ Failed for {agent_name}: {str(e)}")
                                logger.log(f"Failed LTM synthesis for {agent_name}: {e}", "ERROR", "App")
                                
                        # Safely increment progress
                        st.session_state.global_synth_progress = min(100, st.session_state.global_synth_progress + int(100 / total))

                    async def run_all():
                        tasks = [process_agent(name) for name in agents_to_process]
                        await asyncio.gather(*tasks)

                    loop.run_until_complete(run_all())
                    
                    st.session_state.global_synth_progress = 100
                    st.session_state.global_synth_state = "SUCCESS"
                    st.session_state.global_synth_status = "Done!"
                except Exception as e:
                    import traceback
                    logger.log(f"Global Synth Error: {e}\n{traceback.format_exc()}", "ERROR", "App")
                    st.session_state.global_synth_state = "ERROR"
                    st.session_state.global_synth_status = str(e)
                finally:
                    loop.close()

            t = threading.Thread(target=background_worker)
            if add_script_run_ctx:
                add_script_run_ctx(t)
            t.start()

        if st.session_state.global_synth_state == "RUNNING":
            st.button("Synthesize All Agents' Thoughts", disabled=True, width='stretch')
            
            @st.fragment(run_every="1s")
            def global_synth_status_fragment():
                if st.session_state.global_synth_state == "RUNNING":
                    st.info(f"⏳ {st.session_state.global_synth_status}")
                    st.progress(st.session_state.global_synth_progress / 100)
                else:
                    st.rerun()
            global_synth_status_fragment()
        else:
            if st.button("Synthesize All Agents' Thoughts", width='stretch'):
                all_agents = cached_list_agents()
                if not all_agents:
                    st.warning("No agents available.")
                else:
                    run_global_synth_thread(all_agents)
                    st.rerun()

        if st.session_state.global_synth_state == "SUCCESS":
            st.success("All Long Term Memories Synthesized!")
            with st.expander("View Results", expanded=True):
                for res in st.session_state.global_synth_results:
                    st.write(res)
            if st.button("Clear Finished Synthesis Task"):
                st.session_state.global_synth_state = "IDLE"
                st.rerun()
        elif st.session_state.global_synth_state == "ERROR":
            st.error(f"Error during synthesis: {st.session_state.global_synth_status}")
            if st.button("Retry Synthesis"):
                st.session_state.global_synth_sta# --- Tab 2: Swarm Simulation ---
with tab2:
    st.header("Swarm Simulation")
    
    col_setup, col_viz = st.columns([1, 2])
    
    with col_setup:
        st.subheader("🎛️ Swarm Control Panel")
        
        # Inputs
        available_agents = cached_list_agents()
        universal_prompt = st.text_area(
            "Universal Prompt (Global Goal)", 
            "CUse your expertise to determine how to develop an evolving swarm of multi agernt AI agents to continously improve and adapt to new knowledge and information",
            height=120
        )
        
        st.write("### 🧠 Dynamic Orchestration Settings")
        dynamic_orchestration_enabled = st.checkbox(
            "Enable Dynamic Orchestration",
            value=st.session_state.get("dynamic_orchestration_enabled", False),
            help="When enabled, an orchestrator agent analyzes the prompt and dynamically selects, adds, or removes agents from the swarm depending on the context of the conversation."
        )
        st.session_state.dynamic_orchestration_enabled = dynamic_orchestration_enabled
        
        if dynamic_orchestration_enabled:
            col_o1, col_o2 = st.columns(2)
            with col_o1:
                auto_allocate_iterations = st.number_input(
                    "Auto-Allocate Phases",
                    min_value=1,
                    max_value=20,
                    value=st.session_state.get("auto_allocate_iterations", 3),
                    help="The number of times the orchestrator will analyze conversation progress and re-allocate agents."
                )
                st.session_state.auto_allocate_iterations = auto_allocate_iterations
            with col_o2:
                steps_per_phase = st.number_input(
                    "Steps per Phase",
                    min_value=1,
                    max_value=10,
                    value=st.session_state.get("steps_per_phase", 2),
                    help="The number of conversational steps the active agents converse for in each phase before giving feedback back to the orchestrator."
                )
                st.session_state.steps_per_phase = steps_per_phase
            
            # Set values for the backend initialization
            iterations = auto_allocate_iterations * steps_per_phase
            selected_agent_names = []
        else:
            if "selected_agents_list" not in st.session_state:
                st.session_state.selected_agents_list = []

            # Filter selected agents to make sure they still exist in available configs
            available_agent_configs = cached_load_all_agents()
            available_names_set = {cfg.name for cfg in available_agent_configs}
            st.session_state.selected_agents_list = [name for name in st.session_state.selected_agents_list if name in available_names_set]

            # Split configs into available and selected
            avail_configs = [cfg for cfg in available_agent_configs if cfg.name not in st.session_state.selected_agents_list]
            selected_configs = [cfg for cfg in available_agent_configs if cfg.name in st.session_state.selected_agents_list]

            st.write("##### 👥 Swarm Composition")
            pane_avail, pane_selected = st.columns(2)
            
            with pane_avail:
                st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: rgba(255,255,255,0.6); margin-bottom: 8px;'>Available Agents</p>", unsafe_allow_html=True)
                if not avail_configs:
                    st.caption("No available agents")
                else:
                    for cfg in avail_configs:
                        # Render agent card
                        card_col, btn_col = st.columns([7, 3])
                        with card_col:
                            st.markdown(f"""
                            <div style="border-left: 3px solid {cfg.color}; padding-left: 6px; margin-bottom: 12px; line-height: 1.2;">
                                <div style="font-weight: 600; color: #ffffff; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{cfg.name}</div>
                                <div style="font-size: 0.7rem; color: rgba(255,255,255,0.45); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{cfg.domain}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with btn_col:
                            if st.button("➕ Add", key=f"add_pane_{cfg.name}", use_container_width=True):
                                st.session_state.selected_agents_list.append(cfg.name)
                                st.rerun()
            
            with pane_selected:
                st.markdown("<p style='font-size: 0.85rem; font-weight: 600; color: #00f2fe; margin-bottom: 8px;'>Selected Swarm</p>", unsafe_allow_html=True)
                if not selected_configs:
                    st.caption("No agents selected")
                else:
                    for cfg in selected_configs:
                        card_col, btn_col = st.columns([7, 3])
                        with card_col:
                            st.markdown(f"""
                            <div style="border-left: 3px solid {cfg.color}; padding-left: 6px; margin-bottom: 12px; line-height: 1.2;">
                                <div style="font-weight: 600; color: #ffffff; font-size: 0.85rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{cfg.name}</div>
                                <div style="font-size: 0.7rem; color: rgba(255,255,255,0.45); white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{cfg.domain}</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with btn_col:
                            if st.button("➖ Remove", key=f"remove_pane_{cfg.name}", use_container_width=True):
                                st.session_state.selected_agents_list.remove(cfg.name)
                                st.rerun()

            selected_agent_names = st.session_state.selected_agents_list
            iterations = st.number_input("Iterations", min_value=1, max_value=100000000, value=10)
        
        # Expandable Import Simulation
        with st.expander("📥 Import Prior Simulation", expanded=False):
            uploaded_sim = st.file_uploader("Import Simulation Export", type=["json"], key="sim_import")
            if uploaded_sim is not None:
                try:
                    # Check if this file is already loaded to prevent reset loop
                    file_id = f"{uploaded_sim.name}_{uploaded_sim.size}"
                    if "last_imported_sim" not in st.session_state or st.session_state.last_imported_sim != file_id:
                        # Parse JSON
                        export_data = json.load(uploaded_sim)
                        
                        # Validation
                        if "steps" not in export_data or not isinstance(export_data["steps"], list):
                            st.error("Invalid simulation file format.")
                        else:
                            # Reconstruct History
                            reconstructed_history = []
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
                                            loaded_cfg = SwarmIO.load_agent_config(a_name)
                                            if loaded_cfg and str(loaded_cfg.id) == a_id:
                                                 config_cache[a_id] = loaded_cfg
                                            else:
                                                # Fallback: Create a dummy config
                                                config_cache[a_id] = AgentConfig(
                                                    id=a_id,
                                                    name=a_name,
                                                    domain="Imported",
                                                    system_prompt_template="",
                                                    color="#808080"
                                                )
                                        
                                        cfg = config_cache[a_id]
                                        current_configs[a_id] = cfg
                                        
                                        # Reconstruct State
                                        pos_data = agent_data.get("position", {"x": 0, "y": 0, "z": 0})
                                        vel_data = agent_data.get("velocity", {"x": 0, "y": 0, "z": 0})
                                        
                                        current_agent_states[a_id] = AgentRuntimeState(
                                            config_id=a_id,
                                            position=Vector3(**pos_data),
                                            velocity=Vector3(**vel_data),
                                            acceleration=Vector3(x=0, y=0, z=0),
                                            latest_thought=agent_data.get("thought", "")
                                        )
                                    
                                    snapshot = SwarmSnapshot(
                                        iteration=iteration,
                                        max_iterations=len(export_data["steps"]),
                                        universal_prompt="Imported Simulation",
                                        agent_configs=current_configs,
                                        agent_states=current_agent_states,
                                        influence_matrix={}
                                    )
                                    reconstructed_history.append(snapshot)
                            
                            # Update State
                            if len(reconstructed_history) > 0:
                                st.session_state.history = reconstructed_history
                                st.session_state.last_imported_sim = file_id
                                st.success(f"Successfully imported {len(reconstructed_history)} steps!")
                                st.session_state.step_slider = 0
                except Exception as e:
                    st.error(f"Error importing simulation: {e}")
        
        # Initialize & Run Button
        if st.button("Initialize & Run Swarm", width='stretch', type="primary"):
            if not dynamic_orchestration_enabled and not selected_agent_names:
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
                
                # Initialize States
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
                st.session_state.sim_id = initial_state.simulation_id
                st.session_state.sim_active = True
                st.session_state.sim_finished = False
                
                import threading
                is_dynamic = st.session_state.get("dynamic_orchestration_enabled", False)
                auto_alloc_iters = st.session_state.get("auto_allocate_iterations", 3)
                phase_steps = st.session_state.get("steps_per_phase", 2)
                
                def run_sim_thread(state, is_dyn, alloc_iters, steps_ph):
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        if is_dyn:
                            loop.run_until_complete(orchestrator.run_dynamic_simulation(
                                state,
                                alloc_iters,
                                steps_ph
                            ))
                        else:
                            loop.run_until_complete(orchestrator.run_full_simulation(state))
                    except Exception as e:
                        import traceback
                        logger.log(f"Simulation Error: {e}", "ERROR", "App")
                        SwarmIO.save_live_state(state.simulation_id, {
                            "status": "error",
                            "message": str(e),
                            "traceback": traceback.format_exc()
                        })
                        
                t = threading.Thread(target=run_sim_thread, args=(initial_state, is_dynamic, auto_alloc_iters, phase_steps), daemon=True)
                try:
                    from streamlit.runtime.scriptrunner import add_script_run_ctx
                    add_script_run_ctx(t)
                except ImportError:
                    pass
                t.start()
                
        # Consensus Synthesis when finished
        if st.session_state.get("sim_finished", False):
            live_state = st.session_state.get("sim_final_state", {})
            st.divider()
            st.subheader("📝 Consensus Synthesis")
            
            paper_col, kg_col = st.columns(2)
            with paper_col:
                if st.button("Generate Research Paper", width='stretch'):
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
            
            with kg_col:
                kg_domain_input = st.text_input("New KG Domain Name", value="Collaborative_KG")
                if st.button("Generate Collaborative KG", width='stretch'):
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
                                cached_list_kgs.clear()
                            finally:
                                synth_loop.close()
                            
                            status.update(label="Collaborative Knowledge Graph Generated!", state="complete", expanded=False)
                            
                        if "Error" in kg_filename:
                            st.error(kg_filename)
                        else:
                            st.success(f"Knowledge Graph saved successfully: `{kg_filename}`")
                            st.info("You can now assign this Knowledge Graph to an agent in the Agent Forge tab.")

    with col_viz:
        st.subheader("📡 Swarm Visualization Tower")
        
        # Case 1: Active simulation running in background
        if st.session_state.get("sim_active", False):
            sim_status_fragment()
            
        # Case 2: Finished or imported simulation data exists
        elif "history" in st.session_state and st.session_state.history:
            history = st.session_state.history
            max_step = len(history) - 1
            
            # Playback data calculations
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

            render_playback_panel(history, x_range, y_range, z_range, max_step)

        # Case 3: Idle status
        else:
            st.markdown("""
            <div style="background: rgba(30, 41, 59, 0.25); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 60px; text-align: center; margin-top: 40px; box-shadow: inset 0 2px 10px rgba(0,0,0,0.2);">
                <h3 style="color: #00f2fe; margin-bottom: 12px; font-weight: 600;">🌌 Control Tower Active</h3>
                <p style="color: rgba(255,255,255,0.65); max-width: 500px; margin: 0 auto; line-height: 1.6; font-size: 0.95rem;">
                    Configure your swarm parameters on the left and click <b>Initialize & Run Swarm</b> to watch the collective intelligence murmuration in 3D space.
                </p>
                <div style="margin-top: 30px; display: flex; justify-content: center; gap: 20px;">
                    <div style="text-align: center; background: rgba(255,255,255,0.02); padding: 12px 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 2rem;">🐝</span>
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.4); margin-top: 5px;">Cognitive Swarm</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.02); padding: 12px 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 2rem;">📐</span>
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.4); margin-top: 5px;">Boids Flocking</div>
                    </div>
                    <div style="text-align: center; background: rgba(255,255,255,0.02); padding: 12px 20px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.04);">
                        <span style="font-size: 2rem;">📊</span>
                        <div style="font-size: 0.75rem; color: rgba(255,255,255,0.4); margin-top: 5px;">Semantic RAG</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

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
    
    kg_source_type = st.radio("Knowledge Graph Source", ["Global", "Agent Long Term Memory"], horizontal=True)
    kg_path = None
    
    if kg_source_type == "Global":
        kgs = cached_list_kgs()
        if not kgs:
             st.info("No Global Knowledge Graphs found.")
        else:
             selected_kg = st.selectbox("Select a Knowledge Graph to View", kgs, key="kg_viewer_select")
             if selected_kg:
                 kg_path = os.path.join("swarm_data", "knowledge_graphs", selected_kg)
    else:
        agents = cached_list_agents()
        agents_with_kgs = []
        for a in agents:
            p = SwarmIO.get_agent_ltm_kg_path(a)
            if os.path.exists(p):
                agents_with_kgs.append(a)
                
        if not agents_with_kgs:
            st.info("No Agent Long Term Memory Knowledge Graphs found.")
        else:
            selected_agent_kg = st.selectbox("Select an Agent to View their LTM", agents_with_kgs, key="agent_kg_viewer_select")
            if selected_agent_kg:
                kg_path = SwarmIO.get_agent_ltm_kg_path(selected_agent_kg)

    if kg_path:
        try:
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
                     # Set fixed height for KG plot stability
                     fig.update_layout(height=700)
                     selection = plotly_events(fig, click_event=True, hover_event=False, key=f"kg_plot_{selected_kg}", override_height=700)
                     
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

# --- Tab 5: Developer Console ---
with tab5:
    st.header("👩‍💻 Developer Console")
    
    @st.fragment(run_every="2s")
    def console_fragment():
        logs = logger.get_logs()
        
        col1, col2 = st.columns([8, 2])
        with col1:
            st.write("Real-time system events and LLM interaction logs.")
        with col2:
            if st.button("Clear Console", key="clear_logs"):
                logger.clear()
                st.rerun()
                
        if logs:
            df_logs = pd.DataFrame(logs).iloc[::-1]
            
            log_lines_html = []
            for idx, row in df_logs.iterrows():
                level = row['level'].upper()
                badge_class = "badge-info"
                if level == 'ERROR':
                    badge_class = "badge-error"
                elif level == 'WARNING':
                    badge_class = "badge-warning"
                elif level == 'SUCCESS':
                    badge_class = "badge-success"
                
                escaped_msg = row['message'].replace('<', '&lt;').replace('>', '&gt;')
                log_lines_html.append(f"""
                <div class="terminal-line">
                    <span class="terminal-time">[{row['timestamp']}]</span>
                    <span class="terminal-badge {badge_class}">{level}</span>
                    <span class="terminal-source">[{row['source']}]</span>
                    <span class="terminal-msg">{escaped_msg}</span>
                </div>
                """)
            
            console_html = f"""
            <div class="terminal-console">
                {"".join(log_lines_html)}
            </div>
            """
            st.markdown(console_html, unsafe_allow_html=True)
        else:
            st.info("System Ready. Waiting for events...")
            
    console_fragment()
