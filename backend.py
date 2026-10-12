import os
import json
import asyncio
import threading
import traceback
from uuid import uuid4
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.swarm_schema import AgentConfig, SwarmSnapshot, AgentRuntimeState, Vector3
from core.swarm_io import SwarmIO
from core.swarm_orchestrator import SwarmOrchestrator
from core.settings_manager import SettingsManager, GlobalSettings, PhysicsSettings, PhysicsWeights, KnowledgeGraphSettings
from core.knowledge_graph import KnowledgeArchitect
from core.swarm_logger import SwarmLogger
from core.DOAB_client import DOABClient
from core.swarm_gemini import GeminiLLM

# Initialize core services
settings_manager = SettingsManager()
orchestrator = SwarmOrchestrator()
architect = KnowledgeArchitect()
logger = SwarmLogger()

app = FastAPI(title="Swarm Intelligence Control Tower API")

# Configure CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory status stores
doab_status_store = {
    "state": "IDLE",
    "progress": 0,
    "status": "",
    "error": "",
    "result": ""
}

global_synth_status_store = {
    "state": "IDLE",
    "progress": 0,
    "status": "",
    "results": []
}

# --- Settings API ---
@app.get("/api/settings")
def get_settings():
    return settings_manager.get_settings().model_dump()

@app.post("/api/settings")
def update_settings(settings: dict):
    try:
        current_settings = settings_manager.get_settings()
        updated = current_settings.model_copy(update=settings)
        
        # If llm_provider or model_name changed, switch orchestrator
        prov_changed = settings.get("llm_provider") and settings.get("llm_provider") != current_settings.llm_provider
        model_changed = settings.get("model_name") and settings.get("model_name") != current_settings.model_name
        
        settings_manager.update_settings(updated)
        settings_manager.save_settings()
        
        if prov_changed or model_changed:
            orchestrator.switch_provider(updated.llm_provider, updated.model_name)
            
        return {"status": "success", "settings": updated.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings/physics")
def update_physics(weights: dict):
    try:
        current_settings = settings_manager.get_settings()
        physics = current_settings.physics
        current_weights = physics.weights
        
        new_weights = current_weights.model_copy(update={
            "separation": weights.get("separation", current_weights.separation),
            "alignment": weights.get("alignment", current_weights.alignment),
            "cohesion": weights.get("cohesion", current_weights.cohesion),
            "goal_seeking": weights.get("goal_seeking", current_weights.goal_seeking)
        })
        
        new_physics = physics.model_copy(update={
            "visual_range": weights.get("visual_range", physics.visual_range),
            "max_speed": weights.get("max_speed", physics.max_speed),
            "weights": new_weights
        })
        
        updated_settings = current_settings.model_copy(update={
            "physics": new_physics
        })
        
        settings_manager.update_settings(updated_settings)
        # Note: Do not save to disk on every slide to avoid excessive wear, just keep in memory.
        # Front-end can call save explicitly or we can save on shutdown/apply.
        return {"status": "success", "physics": new_physics.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/settings/save")
def save_settings_to_disk():
    try:
        settings_manager.save_settings()
        return {"status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Agents API ---
@app.get("/api/agents")
def list_agents():
    # Return configs list directly
    names = SwarmIO.list_available_agents()
    configs = []
    for name in names:
        cfg = SwarmIO.load_agent_config(name)
        if cfg:
            configs.append(cfg.model_dump())
    return configs

@app.post("/api/agents")
def save_agent(agent_data: dict):
    try:
        # Generate new UUID if not provided
        agent_id = agent_data.get("id")
        if not agent_id:
            agent_id = str(uuid4())
            
        config = AgentConfig(
            id=agent_id,
            name=agent_data["name"],
            domain=agent_data["domain"],
            knowledge_graph_paths=agent_data.get("knowledge_graph_paths", []),
            system_prompt_template=agent_data["system_prompt_template"],
            interaction_style=agent_data.get("interaction_style", "Collaborative"),
            color=agent_data.get("color", "#00CCFF")
        )
        SwarmIO.save_agent_config(config)
        return {"status": "success", "agent": config.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/agents/enrich-prompt")
async def enrich_system_prompt(data: dict):
    try:
        prompt = data["system_prompt_template"]
        kg_path = data["kg_path"]
        enriched = architect.enrich_system_prompt(prompt, kg_path)
        return {"status": "success", "enriched_prompt": enriched}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- KGs API ---
@app.get("/api/kgs")
def list_kgs():
    return SwarmIO.list_available_kgs()

@app.post("/api/kgs/build")
async def build_kg_from_files(files: List[UploadFile] = File(...), domain: str = Form(...)):
    SwarmIO.ensure_directories()
    temp_paths = []
    try:
        for file in files:
            temp_path = os.path.join("swarm_data", file.filename)
            content = await file.read()
            with open(temp_path, "wb") as f:
                f.write(content)
            temp_paths.append(temp_path)
            
        # Run synchronous build graph
        kg_filename = architect.build_graph(temp_paths, domain.replace(" ", "_"))
        return {"status": "success", "filename": kg_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

@app.get("/api/kgs/view/{filename}")
def view_kg_data(filename: str):
    kg_path = os.path.join("swarm_data", "knowledge_graphs", filename)
    if not os.path.exists(kg_path):
        # Fallback to agent LTM KG if global not found
        # Check if filename is actually agent LTM path
        for agent_name in SwarmIO.list_available_agents():
            p = SwarmIO.get_agent_ltm_kg_path(agent_name)
            if os.path.basename(p) == filename:
                kg_path = p
                break
                
    if not os.path.exists(kg_path):
        raise HTTPException(status_code=404, detail="Knowledge Graph not found.")
        
    try:
        with open(kg_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- DOAB Pipeline Async Background Tasks ---
def doab_pipeline_background_worker(prompt: str, domain: str, limit: int):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        global doab_status_store
        doab_status_store["state"] = "RUNNING"
        doab_status_store["progress"] = 5
        doab_status_store["status"] = "Generating search terms..."
        doab_status_store["error"] = ""
        doab_status_store["result"] = ""
        
        logger.log(f"Starting DOAB Search Pipeline for prompt: '{prompt}'", "INFO", "Backend")
        prompt_for_terms = (
            f"You are an expert researcher. Extract 2 to 4 distinct, concise search keywords or short phrases "
            f"from this prompt to be used individually in database queries: '{prompt}'. "
            f"Return ONLY a comma-separated list of the terms. Do not include any conversational filler, quotes, or markdown."
        )
        
        response = loop.run_until_complete(architect.model.generate_content_async(prompt_for_terms))
        raw_response = response.text.strip().replace('"', '')
        term_list = [t.strip() for t in raw_response.split(',') if t.strip()]
        logger.log(f"Parsed Search Terms: {term_list}", "INFO", "Backend")
        
        # Download PDFs
        doab_status_store["progress"] = 20
        doab_status_store["status"] = "Searching and downloading from DOAB..."
        client = DOABClient()
        downloaded_files = []
        
        for i, term in enumerate(term_list):
            if len(downloaded_files) >= limit:
                break
            remaining_limit = limit - len(downloaded_files)
            doab_status_store["status"] = f"Searching and downloading for: '{term}'..."
            files = client.download_pdfs(term, limit=remaining_limit)
            
            for f in files:
                if f not in downloaded_files:
                    downloaded_files.append(f)
            doab_status_store["progress"] = 20 + int(30 * ((i + 1) / len(term_list)))
            
        if not downloaded_files:
            doab_status_store["state"] = "ERROR"
            doab_status_store["error"] = "No PDFs found or downloaded from DOAB for these terms."
            return
            
        # Ingest and Build Graph
        doab_status_store["progress"] = 50
        doab_status_store["status"] = "Ingesting files and building Knowledge Graph..."
        
        def progress_cb(curr, total):
            pct = 50 + int(50 * (curr / total))
            doab_status_store["progress"] = pct
            doab_status_store["status"] = f"Building Knowledge Graph (chunk {curr}/{total})..."
            
        kg_filename = loop.run_until_complete(
            architect.build_graph_async(downloaded_files, domain.replace(" ", "_"), progress_callback=progress_cb)
        )
        
        doab_status_store["state"] = "SUCCESS"
        doab_status_store["result"] = kg_filename
        doab_status_store["status"] = "Done!"
        doab_status_store["progress"] = 100
        logger.log(f"DOAB Pipeline completed. KG: {kg_filename}", "SUCCESS", "Backend")
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.log(f"DOAB Pipeline Error: {e}\n{error_trace}", "ERROR", "Backend")
        doab_status_store["state"] = "ERROR"
        doab_status_store["error"] = str(e)
    finally:
        loop.close()

@app.post("/api/kgs/doab")
def start_doab_pipeline(data: dict, background_tasks: BackgroundTasks):
    prompt = data["prompt"]
    domain = data["domain"]
    limit = data.get("limit", 3)
    
    if doab_status_store["state"] == "RUNNING":
        raise HTTPException(status_code=400, detail="A DOAB search is already running.")
        
    background_tasks.add_task(doab_pipeline_background_worker, prompt, domain, limit)
    return {"status": "started"}

@app.get("/api/kgs/doab/status")
def get_doab_status():
    return doab_status_store

@app.post("/api/kgs/doab/clear")
def clear_doab_status():
    global doab_status_store
    doab_status_store = {
        "state": "IDLE",
        "progress": 0,
        "status": "",
        "error": "",
        "result": ""
    }
    return {"status": "success"}

# --- Long Term Memory APIs ---
@app.post("/api/agents/{agent_name}/ltm/synthesize")
def synthesize_agent_ltm(agent_name: str):
    thoughts_file = SwarmIO.get_agent_thoughts_file(agent_name)
    if not os.path.exists(thoughts_file):
        raise HTTPException(status_code=404, detail=f"No thoughts file found for {agent_name}.")
        
    try:
        synth_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(synth_loop)
        
        safe_name = "".join([c for c in agent_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "_")
        kg_domain = f"agent_{safe_name}_ltm_kg"
        output_path = SwarmIO.get_agent_ltm_kg_path(agent_name)
        
        kg_filename = synth_loop.run_until_complete(
            architect.build_graph_async([thoughts_file], kg_domain, output_filepath=output_path)
        )
        return {"status": "success", "filename": kg_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        synth_loop.close()

def global_ltm_background_worker(agents: List[str]):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        global global_synth_status_store
        global_synth_status_store["state"] = "RUNNING"
        global_synth_status_store["progress"] = 0
        global_synth_status_store["results"] = []
        
        total = len(agents)
        global_synth_status_store["status"] = f"Processing {total} agents in parallel..."
        
        async def process_agent(agent_name):
            thoughts_file = SwarmIO.get_agent_thoughts_file(agent_name)
            if not os.path.exists(thoughts_file):
                global_synth_status_store["results"].append(f"⏭️ No thoughts file for {agent_name}, skipping.")
            else:
                safe_name = "".join([c for c in agent_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "_")
                kg_domain = f"agent_{safe_name}_ltm_kg"
                output_path = SwarmIO.get_agent_ltm_kg_path(agent_name)
                logger.log(f"Starting LTM synthesis for: {agent_name}", "INFO", "Backend")
                try:
                    kg_filename = await architect.build_graph_async([thoughts_file], kg_domain, output_filepath=output_path)
                    global_synth_status_store["results"].append(f"✅ Generated {kg_filename} for {agent_name}")
                except Exception as e:
                    global_synth_status_store["results"].append(f"❌ Failed for {agent_name}: {str(e)}")
                    logger.log(f"Failed LTM synthesis for {agent_name}: {e}", "ERROR", "Backend")
            global_synth_status_store["progress"] = min(100, global_synth_status_store["progress"] + int(100 / total))

        async def run_all():
            tasks = [process_agent(name) for name in agents]
            await asyncio.gather(*tasks)

        loop.run_until_complete(run_all())
        global_synth_status_store["progress"] = 100
        global_synth_status_store["state"] = "SUCCESS"
        global_synth_status_store["status"] = "Done!"
    except Exception as e:
        logger.log(f"Global LTM Synthesis Error: {e}", "ERROR", "Backend")
        global_synth_status_store["state"] = "ERROR"
        global_synth_status_store["status"] = str(e)
    finally:
        loop.close()

@app.post("/api/agents/ltm/synthesize-all")
def synthesize_all_agents_ltm(background_tasks: BackgroundTasks):
    agents = SwarmIO.list_available_agents()
    if not agents:
        raise HTTPException(status_code=400, detail="No agents available.")
        
    if global_synth_status_store["state"] == "RUNNING":
        raise HTTPException(status_code=400, detail="LTM synthesis is already running.")
        
    background_tasks.add_task(global_ltm_background_worker, agents)
    return {"status": "started"}

@app.get("/api/agents/ltm/synthesize-all/status")
def get_global_synth_status():
    return global_synth_status_store

@app.post("/api/agents/ltm/synthesize-all/clear")
def clear_global_synth_status():
    global global_synth_status_store
    global_synth_status_store = {
        "state": "IDLE",
        "progress": 0,
        "status": "",
        "results": []
    }
    return {"status": "success"}

# --- Simulation Endpoints ---
def run_simulation_thread(state: SwarmSnapshot, is_dyn: bool, alloc_iters: int, steps_ph: int):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        if is_dyn:
            loop.run_until_complete(orchestrator.run_dynamic_simulation(state, alloc_iters, steps_ph))
        else:
            loop.run_until_complete(orchestrator.run_full_simulation(state))
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.log(f"Simulation Error: {e}", "ERROR", "Backend")
        SwarmIO.save_live_state(state.simulation_id, {
            "status": "error",
            "message": str(e),
            "traceback": error_trace
        })
    finally:
        loop.close()

@app.post("/api/simulation/run")
def run_simulation(data: dict):
    try:
        universal_prompt = data.get("universal_prompt", "Solve the goal collectively")
        dynamic_orchestration_enabled = data.get("dynamic_orchestration_enabled", False)
        selected_agent_names = data.get("selected_agent_names", [])
        iterations = data.get("iterations", 10)
        
        auto_allocate_iterations = data.get("auto_allocate_iterations", 3)
        steps_per_phase = data.get("steps_per_phase", 2)
        
        if dynamic_orchestration_enabled:
            iterations = auto_allocate_iterations * steps_per_phase
            selected_agent_names = [] # Done dynamically
            
        configs = {}
        if not dynamic_orchestration_enabled:
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
        
        # Start daemon thread
        t = threading.Thread(
            target=run_simulation_thread,
            args=(initial_state, dynamic_orchestration_enabled, auto_allocate_iterations, steps_per_phase),
            daemon=True
        )
        t.start()
        
        return {
            "status": "success",
            "simulation_id": initial_state.simulation_id,
            "max_iterations": iterations
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/simulation/status/{sim_id}")
def get_simulation_status(sim_id: str):
    live_state = SwarmIO.load_live_state(sim_id)
    if not live_state:
        return {"status": "initializing", "message": "Starting simulation runner..."}
    return live_state

@app.post("/api/simulation/import")
async def import_simulation(file: UploadFile = File(...)):
    try:
        content = await file.read()
        export_data = json.loads(content)
        
        if "steps" not in export_data or not isinstance(export_data["steps"], list):
            raise HTTPException(status_code=400, detail="Invalid simulation file format.")
            
        # Reconstruct History
        reconstructed_history = []
        config_cache = {} 
        
        for step_data in export_data["steps"]:
            iteration = step_data.get("iteration", 0)
            current_agent_states = {}
            current_configs = {}
            
            for agent_data in step_data.get("agents", []):
                a_id = agent_data.get("id")
                a_name = agent_data.get("name", "Unknown Agent")
                
                if a_id not in config_cache:
                    loaded_cfg = SwarmIO.load_agent_config(a_name)
                    if loaded_cfg and str(loaded_cfg.id) == a_id:
                         config_cache[a_id] = loaded_cfg
                    else:
                        config_cache[a_id] = AgentConfig(
                            id=a_id,
                            name=a_name,
                            domain="Imported",
                            system_prompt_template="",
                            color="#808080"
                        )
                
                cfg = config_cache[a_id]
                current_configs[a_id] = cfg
                
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
            reconstructed_history.append(snapshot.model_dump(mode="json"))
            
        # Create a mock live state representing complete imported simulation
        sim_id = str(uuid4())
        live_state = {
            "status": "complete",
            "iteration": len(reconstructed_history) - 1,
            "max_iterations": len(reconstructed_history) - 1,
            "message": "Imported Complete",
            "export_path": "",
            "history": reconstructed_history,
            "orchestration_logs": export_data.get("orchestration_logs", []),
            "sim_final_state": export_data
        }
        SwarmIO.save_live_state(sim_id, live_state)
        return {"status": "success", "simulation_id": sim_id, "live_state": live_state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulation/{sim_id}/generate-paper")
async def generate_paper(sim_id: str, data: dict):
    live_state = SwarmIO.load_live_state(sim_id)
    if not live_state:
        raise HTTPException(status_code=404, detail="Simulation not found.")
        
    history = [SwarmSnapshot.model_validate(s) for s in live_state.get("history", [])]
    universal_prompt = data.get("universal_prompt", "Simulation Goal")
    
    try:
        paper_content = await orchestrator.synthesize_paper(history, universal_prompt)
        paper_path = SwarmIO.save_research_paper(paper_content, sim_id)
        return {
            "status": "success",
            "paper_content": paper_content,
            "saved_path": os.path.abspath(paper_path)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/simulation/{sim_id}/generate-kg")
async def generate_collaborative_kg_endpoint(sim_id: str, data: dict):
    live_state = SwarmIO.load_live_state(sim_id)
    if not live_state:
        raise HTTPException(status_code=404, detail="Simulation not found.")
        
    history = [SwarmSnapshot.model_validate(s) for s in live_state.get("history", [])]
    conclusion_text = data.get("conclusion_text", "")
    kg_domain = data.get("kg_domain", "Collaborative_KG")
    
    if not conclusion_text:
        raise HTTPException(status_code=400, detail="Conclusion text is required.")
        
    try:
        kg_filename = await orchestrator.generate_collaborative_kg(
            history,
            conclusion_text,
            kg_domain,
            progress_callback=lambda msg, pct: None
        )
        return {"status": "success", "filename": kg_filename}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Logger API ---
@app.get("/api/logs")
def get_logs():
    return logger.get_logs()

@app.post("/api/logs/clear")
def clear_logs():
    logger.clear()
    return {"status": "success"}

# Serve React static assets
frontend_dist_path = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(frontend_dist_path):
    app.mount("/", StaticFiles(directory=frontend_dist_path, html=True), name="frontend")
else:
    # Development fallback
    @app.get("/")
    def read_root():
        return {"info": "FastAPI is running. Build the frontend to see the UI or run 'npm run dev' in frontend directory."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend:app", host="0.0.0.0", port=8501, reload=True)
