import os
from pymilvus import MilvusClient, DataType

# Use local file-based Milvus for easy setup
DB_FILE = "./milvus_maai.db"
client = MilvusClient(DB_FILE)

CORE_KNOWLEDGE = "core_knowledge"
XOH_SANDBOX = "xoh_sandbox"

def init_db():
    schema = MilvusClient.create_schema(
        auto_id=False,
        enable_dynamic_field=True,
    )
    # id field matches EpistemicNode.id
    schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=36, is_primary=True)
    # 3072 is the embedding dimension observed from models/gemini-embedding-001 in this environment
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=3072)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
    schema.add_field(field_name="confidence_score", datatype=DataType.FLOAT)
    schema.add_field(field_name="visitation_count", datatype=DataType.INT64)

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="embedding",
        index_type="AUTOINDEX",
        metric_type="COSINE"
    )

    for collection_name in [CORE_KNOWLEDGE, XOH_SANDBOX]:
        if not client.has_collection(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                index_params=index_params
            )
            print(f"Created collection: {collection_name}")

if __name__ == "__main__":
    init_db()

def increment_visitation(collection_name: str, ids: list[str]):
    """Increments visitation_count for specified IDs."""
    if not ids:
        return
    
    # Milvus doesn't have a direct "increment" update easily in local file mode without UPSERT
    # We fetch current values, then upsert
    res = client.get(
        collection_name=collection_name,
        ids=ids,
        output_fields=["id", "embedding", "content", "confidence_score", "visitation_count"]
    )
    
    updates = []
    for item in res:
        item["visitation_count"] = item.get("visitation_count", 0) + 1
        updates.append(item)
    
    if updates:
        client.upsert(collection_name=collection_name, data=updates)

def get_random_nodes(collection_name: str, count: int = 10) -> list[dict]:
    """Fetches random nodes from a collection."""
    # Simplified random fetch: query with no filter and limit
    # Milvus doesn't have a built-in "RANDOM" sort in local file mode easily
    try:
        results = client.query(
            collection_name=collection_name,
            filter="id != ''",
            output_fields=["embedding", "content", "confidence_score", "visitation_count"],
            limit=count
        )
        return results
    except Exception as e:
        print(f"Error fetching random nodes: {e}")
        return []
