import os
import json
from typing import List, Optional
from swarm_schema import AgentConfig, SwarmSnapshot

class SwarmIO:
    BASE_DIR = "swarm_data"
    AGENTS_DIR = os.path.join(BASE_DIR, "agents")
    SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")
    KG_DIR = os.path.join(BASE_DIR, "knowledge_graphs")
    LIVE_STATES_DIR = "live_states"
    
    @classmethod
    def ensure_directories(cls):
        """Ensures that the necessary directories exist."""
        os.makedirs(cls.AGENTS_DIR, exist_ok=True)
        os.makedirs(cls.SNAPSHOTS_DIR, exist_ok=True)
        os.makedirs(cls.KG_DIR, exist_ok=True)
        os.makedirs(cls.LIVE_STATES_DIR, exist_ok=True)

    @classmethod
    def save_agent_config(cls, config: AgentConfig):
        """Saves an AgentConfig to disk."""
        cls.ensure_directories()
        # Filename: agent_{name}.json. Sanitize name?
        # Using ID might be safer, but spec says "agent_{name}.json" in SPEC_01 overview,
        # but SPEC_02 says "load_agent_config(agent_id: str)".
        # Let's use ID for storage to avoid collision and name changes, but maybe name in filename for readability?
        # Let's stick to using ID as the primary key for lookup.
        # However, list_available_agents() usually implies listing names or IDs.
        # Let's check SPEC_01 3.1: "Stored as agent_{name}.json"
        if not os.path.exists(cls.AGENTS_DIR):
            cls.ensure_directories()
            
        safe_name = "".join([c for c in config.name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "_")
        filepath = os.path.join(cls.AGENTS_DIR, f"agent_{safe_name}.json")
        
        with open(filepath, 'w') as f:
            f.write(config.model_dump_json(indent=4))
            
        # Log success (using print for now as we don't have logger instance injected, 
        # but SwarmLogger is singleton so we can instantiate it)
        from swarm_logger import SwarmLogger
        SwarmLogger().log(f"Saved Agent Config to {filepath}", "SUCCESS", "SwarmIO")

    @classmethod
    def load_agent_config(cls, agent_name: str) -> Optional[AgentConfig]:
        """Loads an AgentConfig by name (filename base)."""
        # Note: Spec said agent_id, but stored as agent_{name}. Loading by ID would require scanning all files.
        # Assuming the argument here is effectively the identifier used to find the file.
        # If I pass "agent_Structural_Engineer", it loads "agent_Structural_Engineer.json".
        
        # If the arg is the raw name "Structural Engineer", I need to converting it to safe name.
        safe_name = "".join([c for c in agent_name if c.isalnum() or c in (' ', '_', '-')]).strip().replace(" ", "_")
        filename = f"agent_{safe_name}.json"
        filepath = os.path.join(cls.AGENTS_DIR, filename)
        
        if not os.path.exists(filepath):
            # Try checking if the user passed the full filename or just the name
            if os.path.exists(os.path.join(cls.AGENTS_DIR, agent_name)):
                 filepath = os.path.join(cls.AGENTS_DIR, agent_name)
            else:
                 return None
        
        with open(filepath, 'r') as f:
            json_str = f.read()
            return AgentConfig.model_validate_json(json_str)

    @classmethod
    def save_snapshot(cls, snapshot: SwarmSnapshot):
        """Saves a SwarmSnapshot to disk."""
        cls.ensure_directories()
        # Filename: snapshot_{sim_id}_{iteration}.json
        filename = f"snapshot_{snapshot.simulation_id}_{snapshot.iteration}.json"
        filepath = os.path.join(cls.SNAPSHOTS_DIR, filename)
        
        with open(filepath, 'w') as f:
            f.write(snapshot.model_dump_json(indent=4))

    @classmethod
    def load_snapshot(cls, filepath: str) -> SwarmSnapshot:
        """Loads a SwarmSnapshot from a specific filepath."""
        if not os.path.isabs(filepath):
            # Assume it's in snapshots dir if just a filename is passed
            filepath = os.path.join(cls.SNAPSHOTS_DIR, filepath)
            
        with open(filepath, 'r') as f:
            return SwarmSnapshot.model_validate_json(f.read())
            
    @classmethod
    def list_available_agents(cls) -> List[str]:
        """Returns a list of available agent names found in the agents directory."""
        cls.ensure_directories()
        agents = []
        for filename in os.listdir(cls.AGENTS_DIR):
            if filename.startswith("agent_") and filename.endswith(".json"):
                # Clean up: remove "agent_" prefix and ".json" suffix
                name_part = filename[6:-5]
                # We could also load the file and get the real name, but that's slower.
                # Let's return the "file stem" or the actual Agent name?
                # Spec says "list_available_agents() -> List[str]"
                # Let's load them to be accurate.
                try:
                    with open(os.path.join(cls.AGENTS_DIR, filename), 'r') as f:
                        config = AgentConfig.model_validate_json(f.read())
                        agents.append(config.name)
                except:
                    continue
        return agents

    @classmethod
    def list_available_kgs(cls) -> List[str]:
        """Returns a list of available Knowledge Graph filenames."""
        cls.ensure_directories()
        kgs = []
        if os.path.exists(cls.KG_DIR):
            for filename in os.listdir(cls.KG_DIR):
                if filename.endswith(".json"):
                    kgs.append(filename)
        return kgs

    @classmethod
    def save_simulation_export(cls, data: dict, simulation_id: str):
        """Saves a custom simulation export (comms + positions) to disk."""
        cls.ensure_directories()
        # Create an 'exports' directory inside swarm_data
        exports_dir = os.path.join(cls.BASE_DIR, "exports")
        os.makedirs(exports_dir, exist_ok=True)
        
        filename = f"sim_export_{simulation_id}.json"
        filepath = os.path.join(exports_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Exported simulation data to {filepath}")
        return filepath

    @classmethod
    def save_conclusion_markdown(cls, content: str, timestamp_str: str):
        """Saves a standalone markdown file containing the consensus conclusion."""
        cls.ensure_directories()
        conclusions_dir = os.path.join(cls.BASE_DIR, "conclusions")
        os.makedirs(conclusions_dir, exist_ok=True)
        
        filename = f"conclusion_{timestamp_str}.md"
        filepath = os.path.join(conclusions_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
            
        print(f"Saved conclusion to {filepath}")
        return filepath

    @classmethod
    def save_research_paper(cls, content: str, simulation_id: str):
        """Saves the generated research paper to disk."""
        cls.ensure_directories()
        papers_dir = os.path.join(cls.BASE_DIR, "papers")
        os.makedirs(papers_dir, exist_ok=True)
        
        filename = f"research_paper_{simulation_id}.md"
        filepath = os.path.join(papers_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
            
        print(f"Propagated research paper to {filepath}")
        return filepath

    @classmethod
    def save_live_state(cls, simulation_id: str, data: dict):
        """Saves real-time simulation state to disk for the UI to read."""
        cls.ensure_directories()
        filepath = os.path.join(cls.LIVE_STATES_DIR, f"live_state_{simulation_id}.json")
        tmp_path = filepath + ".tmp"
        with open(tmp_path, 'w') as f:
            json.dump(data, f)
        os.replace(tmp_path, filepath)

    @classmethod
    def load_live_state(cls, simulation_id: str) -> Optional[dict]:
        """Loads the real-time simulation state from disk."""
        filepath = os.path.join(cls.LIVE_STATES_DIR, f"live_state_{simulation_id}.json")
        if not os.path.exists(filepath):
            return None
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except json.JSONDecodeError:
            return None
