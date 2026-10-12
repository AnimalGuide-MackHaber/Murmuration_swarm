#!/usr/bin/env python3
"""
Automated Reflex-Learner Optimization Harness
Quantifies the learning rate of the AI agent on advanced C++ High-Performance Computing (HPC) optimizations.
"""

import sys
import os
import asyncio
import re

sys.path.insert(0, os.path.abspath("."))
PLUGIN_DIR = os.path.expanduser("~/.gemini/config/plugins/reflex-learner-plugin/scripts")
sys.path.insert(0, PLUGIN_DIR)

try:
    from core.swarm_gemini import GeminiLLM
    from core.settings_manager import SettingsManager
    from continuous_learning_db import ReflexDB
    from crystallize_rules import crystallize_rules
except ImportError as e:
    print(f"Import Error: {e}. Ensure you are in the Murmur_swarm directory.")
    sys.exit(1)

# High-Performance Computing Tasks
TASKS = [
    "Write a C++ N-Body gravitational simulation engine for 100,000 particles.",
    "Write a C++ molecular dynamics simulation calculating atomic interactions.",
    "Write a C++ astrophysics galaxy collision simulator.",
    "Write a C++ smoothed-particle hydrodynamics (SPH) fluid simulator.",
    "Write a C++ Boids flocking swarm simulator."
]

# The HPC optimizations we want the agent to learn
OPTIMIZATION_METRICS = {
    "SIMD_Intrinsics": r"#include\s*<immintrin\.h>|_mm256_|__m256",
    "Multithreading": r"#pragma\s+omp\s+parallel|std::async|std::thread|std::execution::par",
    "Spatial_Partitioning": r"Octree|Barnes-?Hut|Quadtree|SpatialGrid|BVH|KDTree"
}

async def run_evaluation():
    settings = SettingsManager()
    api_key = settings.get_api_key()
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return

    llm = GeminiLLM(model_name="gemma-4-31b-it", api_key=api_key, rate_limiter=None)
    db = ReflexDB(os.path.join(PLUGIN_DIR, "reflex_memory.db"))
    
    # Reset DB for a clean test
    with __import__('sqlite3').connect(os.path.join(PLUGIN_DIR, "reflex_memory.db")) as conn:
        conn.execute("DELETE FROM experiences")
        conn.execute("DELETE FROM global_rules")
        conn.commit()
    
    print("=== Starting Automated Optimization Evaluation ===\n")
    
    results = []
    
    for i, task in enumerate(TASKS):
        print(f"\n--- Iteration {i+1}: {task} ---")
        
        # 1. SKILL: pre-task-retrieval
        past_experiences = db.retrieve_similar_experiences("C++ simulation", limit=1)
        global_rules = db.retrieve_global_rules()
        
        context_injection = ""
        if global_rules:
            context_injection += "CRITICAL SYSTEM RULES:\n"
            for r in global_rules:
                context_injection += f"- {r['rule_text']}\n"
        elif past_experiences:
            context_injection += "PAST MISTAKES TO AVOID:\n"
            for e in past_experiences:
                context_injection += f"- {e['correction_notes']}\n"
                
        # 2. Execute Task
        prompt = f"Task: {task}\n\n{context_injection}\nOutput only the highly optimized C++ code."
        print("[Agent] Thinking...")
        try:
            response = await llm.generate_content_async(prompt)
            code_output = response.text
        except Exception as e:
            print(f"LLM Error: {e}")
            continue
            
        # 3. Grade the Optimization Level
        score = 0
        detected_opts = []
        for opt_name, regex in OPTIMIZATION_METRICS.items():
            if re.search(regex, code_output, re.IGNORECASE):
                score += 1
                detected_opts.append(opt_name)
                
        optimization_score = (score / len(OPTIMIZATION_METRICS)) * 100
        results.append(optimization_score)
        
        print(f"[Evaluator] Optimization Score: {optimization_score:.1f}%")
        print(f"[Evaluator] Detected Optimizations: {detected_opts if detected_opts else 'None (Naive O(N^2) loops)'}")
        
        # 4. SKILL: experience-reflection-hook
        if score < len(OPTIMIZATION_METRICS):
            correction = (
                "Your C++ simulation is too slow. You MUST always apply High-Performance Computing optimizations: "
                "1. Use SIMD (AVX2/AVX-512) intrinsics for math. "
                "2. Use Multithreading (OpenMP or std::execution::par). "
                "3. Use a Spatial Partitioning tree (like Octree or Grid) to reduce O(N^2) complexity to O(N log N)."
            )
            print("[Evaluator] Code is unoptimized. Logging HPC correction to memory...")
            db.log_experience("C++ simulation", "Generated unoptimized code", -1, correction)
            
            # 5. Trigger Catabolic Pruning check (Threshold set to 2)
            crystallize_rules(db_path=os.path.join(PLUGIN_DIR, "reflex_memory.db"), threshold=2)
        else:
            print("[Evaluator] Maximum Optimization Achieved! Logging success...")
            db.log_experience("C++ simulation", "Generated highly optimized code", 1, "Executed flawlessly with full HPC.")
            
    print("\n=== Final HPC Optimization Report ===")
    for i, opt in enumerate(results):
        print(f"Task {i+1}: {opt:.1f}% Optimization Score")
        
    if results[-1] == 100.0 and results[0] < 100.0:
        print("\nCONCLUSION: SUCCESS. The agent successfully learned to default to HPC architecture over time.")
    else:
        print("\nCONCLUSION: FAILED to achieve perfect HPC adaptation.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
