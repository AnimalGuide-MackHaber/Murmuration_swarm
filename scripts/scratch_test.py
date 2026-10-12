from core.swarm_schema import AgentConfig

# Test 1: V1 style (kwargs)
cfg1 = AgentConfig(name="test", domain="test", knowledge_graph_path="old.json", system_prompt_template="test", color="#000")
print(f"Test 1 paths: {cfg1.knowledge_graph_paths}")

# Test 2: V2 style (kwargs)
cfg2 = AgentConfig(name="test", domain="test", knowledge_graph_paths=["new1.json", "new2.json"], system_prompt_template="test", color="#000")
print(f"Test 2 paths: {cfg2.knowledge_graph_paths}")

# Test 3: Loading from dict/json simulating old file format
data = {
    "name": "test",
    "domain": "test",
    "knowledge_graph_path": "old.json",
    "system_prompt_template": "test",
    "color": "#000"
}
cfg3 = AgentConfig.model_validate(data)
print(f"Test 3 paths: {cfg3.knowledge_graph_paths}")
