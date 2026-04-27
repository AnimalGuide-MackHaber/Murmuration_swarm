import os
import json
import time
import asyncio
from markitdown import MarkItDown
from typing import Dict, Any, List
from swarm_io import SwarmIO

# Using GeminiCLIWrapper instead of google.generativeai
# Using OllamaWrapper instead of GeminiCLIWrapper
from swarm_ollama import OllamaWrapper
from swarm_nvidia import NvidiaLLM
from swarm_gemini import GeminiLLM
from settings_manager import SettingsManager
from swarm_logger import SwarmLogger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class KnowledgeArchitect:
    def __init__(self):
        self.settings_manager = SettingsManager()
        self.logger = SwarmLogger()
        self.settings = self.settings_manager.get_settings()
        
        # Initialize LLM Provider
        self.model = None
        provider = self.settings.llm_provider.lower()
        
        try:
            if provider == "nvidia":
                self.logger.log("Initializing Nvidia LLM...", "INFO", "KnowledgeArchitect")
                api_key = self.settings_manager.get_nvidia_api_key()
                model_name = self.settings.model_name
                
                self.model = NvidiaLLM(
                    model_name=model_name,
                    base_url=self.settings.nvidia_api.base_url,
                    api_key=api_key
                )
                self.model.configure()
                self.logger.log(f"Nvidia LLM initialized with model {model_name}", "INFO", "KnowledgeArchitect")
                
            elif provider == "ollama":
                # Default to deepseek if model name not provided or just use global model name
                # The original code hardcoded "deepseek-r1:1.5b", but better to use settings
                model_name = self.settings.model_name if self.settings.model_name else "deepseek-r1:1.5b"
                self.model = OllamaWrapper(model_name=model_name)
                self.logger.log(f"Ollama Wrapper initialized with model: {model_name}", "INFO", "KnowledgeArchitect")
            elif provider == "gemini":
                self.logger.log("Initializing Gemini LLM...", "INFO", "KnowledgeArchitect")
                api_key = self.settings_manager.get_api_key()
                model_name = self.settings.model_name
                
                self.model = GeminiLLM(
                    model_name=model_name,
                    api_key=api_key
                )
                self.model.configure()
                self.logger.log(f"Gemini LLM initialized with model {model_name}", "INFO", "KnowledgeArchitect")
            
            else:
                 self.logger.log(f"Unknown LLM provider: {provider}", "ERROR", "KnowledgeArchitect")

        except Exception as e:
            self.model = None
            self.logger.log(f"Failed to initialize model ({provider}): {e}", "ERROR", "KnowledgeArchitect")

    def ingest_file(self, file_path: str) -> str:
        """Ingests a file (PDF or Markdown) and returns its text content."""
        self.logger.log(f"Ingesting file: {file_path}", "INFO", "KnowledgeArchitect")
        if not os.path.exists(file_path):
             self.logger.log(f"File not found: {file_path}", "ERROR", "KnowledgeArchitect")
             raise FileNotFoundError(f"File not found at {file_path}")
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext == ".pdf":
            return self.ingest_pdf(file_path)
        elif ext in [".md", ".markdown", ".txt"]: 
            return self.ingest_markdown(file_path)
        else:
            self.logger.log(f"Unsupported file type: {ext}", "ERROR", "KnowledgeArchitect")
            raise ValueError(f"Unsupported file type: {ext}")

    def ingest_markdown(self, md_path: str) -> str:
        """Extracts text from a Markdown file."""
        with open(md_path, 'r') as f:
            text = f.read()
        return self._clean_text(text)

    def ingest_pdf(self, pdf_path: str) -> str:
        """Extracts text from a PDF file using MarkItDown."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found at {pdf_path}")
        
        md = MarkItDown()
        result = md.convert(pdf_path)
        return result.text_content

    def _clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        return " ".join(text.split())

    def _chunk_text(self, text: str, chunk_size: int = 15000) -> List[str]:
        """Splits text into chunks of approximately chunk_size characters."""
        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + chunk_size
            if end >= text_len:
                chunks.append(text[start:])
                break
            
            # Try to find a paragraph break or sentence break
            last_newline = text.rfind('\n', start, end)
            last_period = text.rfind('. ', start, end)
            
            if last_newline != -1 and last_newline > start + chunk_size // 2:
                end = last_newline + 1
            elif last_period != -1 and last_period > start + chunk_size // 2:
                end = last_period + 1
            
            chunks.append(text[start:end])
            start = end
        
        return chunks

    def _merge_graphs(self, master_graph: Dict[str, Any], partial_graph: Dict[str, Any]):
        """Merges a partial graph into the master graph."""
        if not partial_graph or "nodes" not in partial_graph:
            return

        master_nodes = master_graph.setdefault("nodes", {})
        partial_nodes = partial_graph.get("nodes", {})

        for node_id, node_data in partial_nodes.items():
            if node_id not in master_nodes:
                master_nodes[node_id] = node_data
            else:
                # Merge children if node exists
                existing_children = set(master_nodes[node_id].get("children", []))
                new_children = node_data.get("children", [])
                for child in new_children:
                    existing_children.add(child)
                master_nodes[node_id]["children"] = list(existing_children)


    async def build_graph_async(self, file_paths: List[str], domain: str, progress_callback=None) -> str:
        """
        Async version of build_graph.
        """
        self.logger.log(f"Starting Knowledge Graph build (async) for domain: {domain}", "INFO", "KnowledgeArchitect")
        
        try:
            text = ""
            for fp in file_paths:
                text += self.ingest_file(fp) + "\n\n"
        except Exception as e:
            self.logger.log(f"Error ingesting files {file_paths}: {e}", "ERROR", "KnowledgeArchitect")
            raise

        if not self.model:
             self.logger.log("Cannot build graph: No Gemini Model loaded.", "ERROR", "KnowledgeArchitect")
             return self._create_dummy_graph(domain)

        # Get settings
        settings = self.settings_manager.get_settings()
        delay = settings.rate_limiting.request_delay_seconds if settings.rate_limiting else 2.0
        chunk_size = settings.knowledge_graph.chunk_size if settings.knowledge_graph else 15000

        chunks = self._chunk_text(text, chunk_size=chunk_size)
        self.logger.log(f"Split text into {len(chunks)} chunks.", "INFO", "KnowledgeArchitect")
        
        master_graph = {"nodes": {}}

        system_prompt = (
            "You are a Knowledge Architect. Convert this text into a hierarchical JSON graph. "
            "Be sure to add as much detail as possible. "
            "It is CRITICAL that you include any and all technical details, definitions, and explanations. "
            "The generated graph will be used to inform an AI agent about the domain, so it needs to be comprehensive and accurate. "
            "Structure: Root Domain -> Core Concepts -> Sub-concepts -> Atomic Facts (with page refs). "
            "Output strictly valid JSON with the following schema:\n"
            "{\n"
            "  'nodes': {\n"
            "    'node_id': {\n"
            "      'label': 'String',\n"
            "      'summary': 'String',\n"
            "      'type': 'concept|fact',\n"
            "      'children': ['child_id_1', 'child_id_2']\n"
            "    }\n"
            "  }\n"
            "}\n"
            "Do not use markdown code blocks. Just the raw JSON string."
        )

        for i, chunk in enumerate(chunks):
            try:
                if progress_callback:
                    progress_callback(i, len(chunks))
                
                self.logger.log(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)...", "INFO", "KnowledgeArchitect")
                
                full_prompt = f"{system_prompt}\n\nContent:\n{chunk}"
                response = await self.model.generate_content_async(full_prompt)
                
                json_str = response.text
                
                if json_str.startswith("```json"):
                    json_str = json_str[7:]
                if json_str.endswith("```"):
                    json_str = json_str[:-3]
                    
                json_str = json_str.strip()
                
                partial_data = json.loads(json_str)
                self._merge_graphs(master_graph, partial_data)
                
                if i < len(chunks) - 1:
                    await asyncio.sleep(delay)

            except Exception as e:
                self.logger.log(f"Error processing chunk {i+1}: {e}", "ERROR", "KnowledgeArchitect")
        
        if progress_callback:
            progress_callback(len(chunks), len(chunks))

        try:
            if not master_graph["nodes"]:
                 self.logger.log(f"No nodes extracted from any chunk.", "WARNING", "KnowledgeArchitect")
                 return self._create_dummy_graph(domain)

            filename = f"{domain}.json"
            filepath = os.path.join(SwarmIO.KG_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(master_graph, f, indent=2)
                
            self.logger.log(f"Knowledge Graph saved to {filename}", "SUCCESS", "KnowledgeArchitect")
            return filename
            
        except Exception as e:
            self.logger.log(f"Error saving graph: {e}", "ERROR", "KnowledgeArchitect")
            return self._create_dummy_graph(domain)

    def build_graph(self, file_paths: List[str], domain: str, progress_callback=None) -> str:
        """
        Synchronous wrapper for build_graph_async.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            # If the loop is already running, we can't use run_until_complete easily if we are in the same thread.
            # However, in a background thread, we usually create a new loop anyway.
            # For safety, if it's running, we use a separate thread or just return a dummy if we can't wait.
            # But usually, this is called from a sync context that expects a result.
            self.logger.log("Warning: build_graph called while event loop is running.", "WARNING", "KnowledgeArchitect")
            # Fallback to a new loop in a temporary thread or just raise?
            # Let's try to run it in the existing loop if possible, but that's complex for a sync return.
            # Realistically, we should avoid calling this from a running loop.
            return asyncio.run(self.build_graph_async(file_paths, domain, progress_callback))
        else:
            return loop.run_until_complete(self.build_graph_async(file_paths, domain, progress_callback))
        
        if progress_callback:
            progress_callback(len(chunks), len(chunks))

        
        # Save merged graph to disk
        try:
            if not master_graph["nodes"]:
                 self.logger.log(f"No nodes extracted from any chunk.", "WARNING", "KnowledgeArchitect")
                 return self._create_dummy_graph(domain)

            filename = f"{domain}.json"
            filepath = os.path.join(SwarmIO.KG_DIR, filename)
            
            with open(filepath, 'w') as f:
                json.dump(master_graph, f, indent=2)
                
            self.logger.log(f"Knowledge Graph saved to {filename}", "SUCCESS", "KnowledgeArchitect")
            return filename
            
        except Exception as e:
            self.logger.log(f"Error saving graph: {e}", "ERROR", "KnowledgeArchitect")
            return self._create_dummy_graph(domain)

    async def generate_base_kg_from_text(self, text: str) -> Dict[str, Any]:
        """
        Takes raw text (e.g., a consensus conclusion) and returns an initial KG JSON structure.
        """
        self.logger.log("Generating base Knowledge Graph from text...", "INFO", "KnowledgeArchitect")
        if not self.model:
            self.logger.log("Cannot build graph: No LLM loaded.", "ERROR", "KnowledgeArchitect")
            return {"nodes": {}}

        system_prompt = (
            "You are a Knowledge Architect. Convert this text into a hierarchical JSON graph. "
            "Structure: Root Domain -> Core Concepts -> Sub-concepts. "
            "Output strictly valid JSON with the following schema:\n"
            "{\n"
            "  'nodes': {\n"
            "    'node_id': {\n"
            "      'label': 'String',\n"
            "      'summary': 'String',\n"
            "      'type': 'concept|fact',\n"
            "      'children': ['child_id_1', 'child_id_2']\n"
            "    }\n"
            "  }\n"
            "}\n"
            "Do not use markdown code blocks. Just the raw JSON string."
        )

        try:
            full_prompt = f"{system_prompt}\n\nContent:\n{text}"
            response = await self.model.generate_content_async(full_prompt)
            
            json_str = response.text.strip()
            if json_str.startswith("```json"): json_str = json_str[7:]
            if json_str.endswith("```"): json_str = json_str[:-3]
            json_str = json_str.strip()
            
            graph_data = json.loads(json_str)
            return graph_data
        except Exception as e:
            self.logger.log(f"Error extracting base graph: {e}", "ERROR", "KnowledgeArchitect")
            return {"nodes": {}}

    async def add_linking_edges(self, graph: Dict[str, Any]) -> Dict[str, Any]:
        """
        Reviews a fully enriched graph and adds new connections (`children` links) 
        between related concepts that might have been added independently.
        """
        self.logger.log("Performing final pass to link related concepts...", "INFO", "KnowledgeArchitect")
        if not self.model or not graph or "nodes" not in graph:
            return graph
            
        system_prompt = (
            "You are a Knowledge Architect. Review the following JSON representing a Knowledge Graph. "
            "Identify concepts/nodes that are conceptually related but do not have an edge between them. "
            "Add the relevant 'node_id's to the 'children' list of parent or related nodes to interlink them. "
            "Output the updated JSON maintaining the exact same schema. Do NOT remove any existing nodes or text.\n"
            "Output strictly valid JSON. Do not use markdown wrappers."
        )
        
        try:
            full_prompt = f"{system_prompt}\n\nGraph JSON:\n{json.dumps(graph)}"
            response = await self.model.generate_content_async(full_prompt)
            
            json_str = response.text.strip()
            if json_str.startswith("```json"): json_str = json_str[7:]
            if json_str.endswith("```"): json_str = json_str[:-3]
            json_str = json_str.strip()
            
            updated_graph = json.loads(json_str)
            return updated_graph
        except Exception as e:
            self.logger.log(f"Error linking edges: {e}", "ERROR", "KnowledgeArchitect")
            return graph

    def _create_dummy_graph(self, domain: str) -> str:
        """Creates a dummy graph for testing or when API fails."""
        data = {
            "nodes": {
                "root": {
                    "label": domain,
                    "summary": f"Root node for {domain}",
                    "type": "concept",
                    "children": ["c1", "c2"]
                },
                "c1": {
                    "label": "Concept 1",
                    "summary": "First major concept",
                    "type": "concept",
                    "children": []
                },
                "c2": {
                    "label": "Concept 2",
                    "summary": "Second major concept",
                    "type": "concept",
                    "children": []
                }
            }
        }
        filename = f"{domain}_dummy.json"
        filepath = os.path.join(SwarmIO.KG_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filename

    def query_graph(self, graph_path: str, node_id: str, depth: int=1) -> Dict[str, Any]:
        """
        Traverses the graph. Returns summary of target node and label of children.
        """
        full_path = os.path.join(SwarmIO.KG_DIR, graph_path)
        
        if not os.path.exists(full_path):
             return {"error": "Graph not found"}

        with open(full_path, 'r') as f:
            data = json.load(f)
            
        nodes = data.get("nodes", {})
        target = nodes.get(node_id)
        
        if not target:
            # Fallback: Search for term in label or summary
            results = []
            query_lower = node_id.lower()
            
            for nid, node in nodes.items():
                if (query_lower in nid.lower() or 
                    query_lower in node.get("label", "").lower() or 
                    query_lower in node.get("summary", "").lower()):
                    
                    results.append({
                        "id": nid,
                        "label": node.get("label"),
                        "summary": node.get("summary"),
                        "score": 1 # Placeholder for relevance
                    })
            
            if not results:
                return {"error": f"Node or term '{node_id}' not found"}
            
            # Limit results
            return {"search_results": results[:5]}
            
        result = {
            "id": node_id,
            "label": target.get("label"),
            "summary": target.get("summary"),
            "children": []
        }
        
        for child_id in target.get("children", []):
            child = nodes.get(child_id)
            if child:
                result["children"].append({
                    "id": child_id,
                    "label": child.get("label")
                })
                
        return result

    def retrieve_context(self, graph_path: str, query_text: str, top_k: int = 3) -> str:
        """
        Uses TF-IDF and Cosine Similarity to automatically retrieve relevant context 
        from the Knowledge Graph based on the given query text.
        """
        full_path = os.path.join(SwarmIO.KG_DIR, graph_path)
        
        if not os.path.exists(full_path):
             return "Knowledge Graph not found."

        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
                
            nodes = data.get("nodes", {})
            if not nodes:
                return "Knowledge Graph is empty."
                
            # Collect summaries and track node IDs
            documents = []
            node_labels = []
            node_ids = []
            for nid, node in nodes.items():
                summary = node.get("summary", "")
                label = node.get("label", nid)
                if summary and isinstance(summary, str) and summary.strip(): # keep only nodes with summaries
                    documents.append(summary)
                    node_labels.append(label)
                    node_ids.append(nid)
                    
            if not documents:
                return "No valid summaries found in the Knowledge Graph to match against."
                
            # Add the query to the list to vectorize them all together
            documents.append(query_text)
            
            # Compute TF-IDF
            vectorizer = TfidfVectorizer(stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            # Compute Cosine Similarity between the query (last row) and all other rows
            cosine_sims = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            
            # Get top K indices
            top_k_indices = cosine_sims.argsort()[-top_k:][::-1]
            
            retrieved_chunks = []
            visited = set()
            queue = [] # (node_id, current_depth)
            
            # Get settings for expansion
            kg_settings = self.settings.knowledge_graph
            max_depth = kg_settings.max_retrieval_depth if kg_settings else 1
            max_children_per_node = kg_settings.max_children_per_node if kg_settings else 3

            # Phase 1: Direct matches from cosine similarity
            for i in top_k_indices:
                score = cosine_sims[i]
                if score > 0.05: # minimum threshold to avoid completely irrelevant noise
                    nid = node_ids[i]
                    if nid not in visited:
                        visited.add(nid)
                        retrieved_chunks.append(f"[{node_labels[i]}]: {documents[i]}")
                        queue.append((nid, 0))
            
            # Phase 2: Breadth-First-Search traversal for connected nodes
            head = 0
            while head < len(queue):
                curr_id, curr_depth = queue[head]
                head += 1
                
                if curr_depth < max_depth:
                    node_data = nodes.get(curr_id)
                    if node_data and "children" in node_data:
                        child_count = 0
                        for child_id in node_data["children"]:
                            if child_count >= max_children_per_node:
                                break
                            
                            if child_id in nodes and child_id not in visited:
                                child_node = nodes[child_id]
                                summary = child_node.get("summary", "")
                                label = child_node.get("label", child_id)
                                
                                if summary:
                                    visited.add(child_id)
                                    retrieved_chunks.append(f"[{label}] (Connected): {summary}")
                                    queue.append((child_id, curr_depth + 1))
                                    child_count += 1
                    
            if not retrieved_chunks:
                return "No highly relevant information found."
                
            return "\n".join(retrieved_chunks)
            
        except Exception as e:
            self.logger.log(f"Error retrieving context via cosine similarity: {e}", "ERROR", "KnowledgeArchitect")
            return f"Error connecting to Knowledge Graph: {e}"

    async def enrich_system_prompt_async(self, base_prompt: str, graph_path: str) -> str:
        """
        Async version of enrich_system_prompt.
        """
        full_path = os.path.join(SwarmIO.KG_DIR, graph_path)
        if not os.path.exists(full_path):
             return f"{base_prompt} (Error: Knowledge Graph not found)"

        try:
            with open(full_path, 'r') as f:
                data = json.load(f)
            
            nodes = data.get("nodes", {})
            roots = [n for nid, n in nodes.items() if nid == "root" or nid == "main"]
            if not roots and nodes:
                roots = list(nodes.values())[:3]
                
            context_str = ""
            for node in list(nodes.values())[:10]:
                context_str += f"- {node.get('label')}: {node.get('summary')}\n"
            
            enhance_prompt = (
                f"You are an expert prompt engineer. "
                f"The user wants an AI agent with the following core persona: '{base_prompt}'. "
                f"This agent has access to a Knowledge Graph with these key concepts:\n{context_str}\n"
                f"Rewrite the system prompt to be more detailed, authoritative, and domain-specific, "
                f"incorporating the key terminology from the KG. "
                f"Keep it under 150 words. Return ONLY the new prompt."
                f"AGAIN ONLY RETURN THE PROMPT YOU HAVE GENERATED!!!! I DON'T NEED YOU TO PHRASE IT HELPFULLY!!! JUST THEN PROMPT!!!!!"
            )
            
            response = await self.model.generate_content_async(enhance_prompt)
            enhanced_text = response.text.strip()
            self.logger.log(f"Auto-Enhance Result: {enhanced_text[:100]}...", "DEBUG", "KnowledgeArchitect")
            return enhanced_text
            
        except Exception as e:
            self.logger.log(f"Error enhancing prompt: {e}", "ERROR", "KnowledgeArchitect")
            return f"{base_prompt} (Error enhancing with KG: {e})"

    def enrich_system_prompt(self, base_prompt: str, graph_path: str) -> str:
        """
        Synchronous wrapper for enrich_system_prompt_async.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if loop.is_running():
            return asyncio.run(self.enrich_system_prompt_async(base_prompt, graph_path))
        else:
            return loop.run_until_complete(self.enrich_system_prompt_async(base_prompt, graph_path))
