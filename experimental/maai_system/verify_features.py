import asyncio
import numpy as np
import uuid
from database import client, CORE_KNOWLEDGE, XOH_SANDBOX, init_db
from agents import run_iconoclast, process_producer_consumer
from governor import control_logic, inject_xoh_nodes
from config import config

async def test_vse_masking():
    print("\n--- Testing VSE Masking ---")
    # 1. Insert a node with high visitation
    node_id = str(uuid.uuid4())
    embedding = [0.1] * 3072
    client.insert(collection_name=CORE_KNOWLEDGE, data=[{
        "id": node_id,
        "embedding": embedding,
        "content": "Saturated knowledge node",
        "confidence_score": 0.9,
        "visitation_count": 100
    }])
    
    # 2. Set threshold lower
    config.set_visitation_threshold(50)
    
    # 3. Try to retrieve context
    from agents import retrieve_context
    results = await retrieve_context("test query")
    
    found = any(node['id'] == node_id for node in results)
    if not found:
        print("Success: Saturated node was masked.")
    else:
        print("Failure: Saturated node was NOT masked.")

async def test_iconoclast():
    print("\n--- Testing Iconoclast Pruning ---")
    # 1. Insert two identical nodes
    id1 = str(uuid.uuid4())
    id2 = str(uuid.uuid4())
    emb = [0.5] * 3072
    client.insert(collection_name=CORE_KNOWLEDGE, data=[
        {"id": id1, "embedding": emb, "content": "Node A", "confidence_score": 0.5, "visitation_count": 0},
        {"id": id2, "embedding": emb, "content": "Node B", "confidence_score": 0.5, "visitation_count": 0}
    ])
    
    # 2. Run iconoclast
    run_iconoclast(0.99)
    
    # 3. Check if one is deleted
    res = client.get(collection_name=CORE_KNOWLEDGE, ids=[id1, id2])
    if len(res) == 1:
        print(f"Success: One redundant node pruned. Remaining: {res[0]['id']}")
    else:
        print(f"Failure: Expected 1 node remaining, found {len(res)}")

async def test_governor_stagnation():
    print("\n--- Testing Governor Stagnation ---")
    # 1. Populate XOH_SANDBOX
    client.insert(collection_name=XOH_SANDBOX, data=[{
        "id": str(uuid.uuid4()),
        "embedding": [0.9] * 3072,
        "content": "Adversarial high-entropy data",
        "confidence_score": 0.1,
        "visitation_count": 0
    }])
    
    # 2. Trigger stagnation (requires < 3 nodes for simple telemetry or specific embeddings)
    # We'll just call the action directly or mock telemetry if needed.
    # But let's see if control_logic handles it.
    status = control_logic()
    print(f"Governor status: {status}")

async def main():
    init_db()
    await test_vse_masking()
    await test_iconoclast()
    await test_governor_stagnation()

if __name__ == "__main__":
    asyncio.run(main())
