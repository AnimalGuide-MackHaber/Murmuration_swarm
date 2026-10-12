import asyncio
import uuid
import json
from google import genai
from google.genai import types
from pydantic import ValidationError

from models import EpistemicNode, EpistemicNetwork
from database import client, CORE_KNOWLEDGE
from config import config
from rate_limiter import AsyncRateLimiter

# Initialize rate limiter using config
limits = config.get_rate_limits(config.llm_provider)
rate_limiter = AsyncRateLimiter(limits["requests_per_minute"], limits["tokens_per_minute"])

# Assumes GEMINI_API_KEY is in environment if not in config
genai_client = genai.Client(api_key=config.google_api_key if config.google_api_key else None) 

async def get_embedding(text: str) -> list[float]:
    estimated_tokens = max(10, len(text) // 4)
    await rate_limiter.wait_if_needed(estimated_tokens)
    
    response = await genai_client.aio.models.embed_content(
        model=config.embedding_model_name,
        contents=text,
    )
    emb = response.embeddings[0].values
    return emb

async def run_producer(seed_context: str) -> str:
    """Generates varied heuristics."""
    prompt = f"Generate 5 novel hypotheses based on: {seed_context}"
    # Rough estimation: input tokens + expected output tokens (e.g., 500)
    estimated_tokens = (len(prompt) // 4) + 500
    await rate_limiter.wait_if_needed(estimated_tokens)
    
    response = await genai_client.aio.models.generate_content(
        model=config.model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=1.2,
            top_p=0.95,
        )
    )
    return response.text

async def run_consumer(raw_output: str) -> list[EpistemicNode]:
    """Parses raw output into structured assertions."""
    prompt = f"Parse the following into structured assertions: {raw_output}"
    # Estimation: input tokens + expected JSON structure overhead
    estimated_tokens = (len(prompt) // 4) + 200
    await rate_limiter.wait_if_needed(estimated_tokens)
    
    response = await genai_client.aio.models.generate_content(
        model=config.model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=EpistemicNetwork,
        )
    )
    
    try:
        # Pydantic validation
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        
        parsed_network = EpistemicNetwork.model_validate_json(text)
        
        # Ensure IDs are set for all nodes
        for node in parsed_network.nodes:
            if not node.id:
                node.id = str(uuid.uuid4())
            
        return parsed_network.nodes
    except ValidationError as e:
        print(f"Consumer validation error: {e}")
        return []

async def retrieve_context(query_text: str, limit: int = 5) -> list[dict]:
    """Retrieves context from CORE_KNOWLEDGE respecting visitation_threshold (VSE)."""
    embedding = await get_embedding(query_text)
    
    threshold = config.visitation_threshold
    search_expr = f"visitation_count < {threshold}"
    
    try:
        results = client.search(
            collection_name=CORE_KNOWLEDGE,
            data=[embedding],
            filter=search_expr,
            limit=limit,
            output_fields=["id", "content"]
        )
        # client.search returns a list of lists (one for each input vector)
        return results[0] if results else []
    except Exception as e:
        print(f"Error retrieving context: {e}")
        return []

async def process_producer_consumer(seed_context: str):
    """Orchestrates Producer -> Consumer -> Database loop with VSE."""
    if config.producers_paused:
        print("Producers are currently paused by the Governor.")
        return

    # 1. Retrieve context based on seed (VSE)
    context_nodes = await retrieve_context(seed_context)
    context_text = "\n".join([node['entity']['content'] for node in context_nodes])
    
    if context_text:
        augmented_seed = f"{seed_context}\n\nAdditional Context:\n{context_text}"
    else:
        augmented_seed = seed_context

    # 2. Run Producer
    raw_output = await run_producer(augmented_seed)
    
    # 3. Run Consumer
    nodes = await run_consumer(raw_output)
    
    # 4. Insert into DB
    if nodes:
        data_to_insert = []
        for node in nodes:
            embedding = await get_embedding(node.content)
            data_to_insert.append({
                "id": node.id,
                "embedding": embedding,
                "content": node.content,
                "confidence_score": node.confidence_score,
                "visitation_count": 0,
            })
        
        if data_to_insert:
            client.insert(collection_name=CORE_KNOWLEDGE, data=data_to_insert)
            print(f"Inserted {len(data_to_insert)} nodes into {CORE_KNOWLEDGE}")

    # 5. Increment visitation for context used (VSE)
    if context_nodes:
        from database import increment_visitation
        node_ids = [node['id'] for node in context_nodes]
        increment_visitation(CORE_KNOWLEDGE, node_ids)
        print(f"Incremented visitation for {len(node_ids)} context nodes.")

def run_synthesizer(large_document_corpus: str) -> str:
    """Aggregates nodes using context caching."""
    # Note: Requires Gemini 2.5 Pro for caching, and it is billed.
    cache = genai_client.caches.create(
        model="gemini-2.5-pro",
        contents=[large_document_corpus],
        config=types.CreateCacheConfig(
            display_name="core_knowledge_cache",
            ttl="1h" # Time to live
        )
    )
    
    response = genai_client.models.generate_content(
        model="gemini-2.5-pro",
        contents="Synthesize these nodes into a unified structural theory. Identify contradictions.",
        config=types.GenerateContentConfig(
            cached_content=cache.name,
            temperature=0.4
        )
    )
    return response.text

def run_iconoclast(similarity_threshold: float = 0.95):
    """Programmatic pruning of redundant nodes."""
    print(f"Running Iconoclast with threshold {similarity_threshold}...")
    
    try:
        results = client.query(
            collection_name=CORE_KNOWLEDGE,
            filter="id != ''",
            output_fields=["id", "embedding"],
            limit=1000
        )
        if len(results) < 2:
            return

        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        embeddings = np.array([res['embedding'] for res in results])
        ids = [res['id'] for res in results]
        
        sim_matrix = cosine_similarity(embeddings)
        
        to_delete = set()
        for i in range(len(ids)):
            if ids[i] in to_delete:
                continue
            for j in range(i + 1, len(ids)):
                if sim_matrix[i, j] > similarity_threshold:
                    to_delete.add(ids[j])
                    print(f"Iconoclast: Flagging redundant node {ids[j]} (similarity {sim_matrix[i,j]:.4f})")
        
        if to_delete:
            client.delete(collection_name=CORE_KNOWLEDGE, ids=list(to_delete))
            print(f"Iconoclast: Deleted {len(to_delete)} redundant nodes.")
        else:
            print("Iconoclast: No redundant nodes found.")
            
    except Exception as e:
        print(f"Iconoclast Error: {e}")
