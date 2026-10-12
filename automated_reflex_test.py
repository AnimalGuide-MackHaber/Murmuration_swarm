#!/usr/bin/env python3
"""
Automated Reflex-Learner Evaluation Harness
Quantifies the learning rate of the AI agent on a completely proprietary, fictional framework.
"""

import sys
import os
import asyncio
import re

# Setup paths to access Murmur_swarm core and the Reflex-Learner Plugin
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

# Fictional Test Prompts
TASKS = [
    "Write a function to perform matrix dot product using lib_murmur_math.",
    "Write a function to invert a 4x4 matrix using lib_murmur_math.",
    "Write a neural network dense layer forward pass using lib_murmur_math.",
    "Write an image convolution filter using lib_murmur_math.",
    "Write a recurrent cell state update using lib_murmur_math."
]

# The proprietary rules we want the agent to learn
PROPRIETARY_CONSTRAINTS = {
    "sync": r"murmur_hardware_sync\(\)",
    "init": r"column-major|COLUMN_MAJOR|column_major",
    "destruct": r"row-major|ROW_MAJOR|row_major"
}

async def run_evaluation():
    settings = SettingsManager()
    api_key = settings.get_api_key()
    if not api_key:
        print("Error: GOOGLE_API_KEY not found.")
        return

    # Use a faster, lightweight model for testing if possible, but default to settings
    llm = GeminiLLM(model_name="gemma-4-31b-it", api_key=api_key, rate_limiter=None)
    db = ReflexDB(os.path.join(PLUGIN_DIR, "reflex_memory.db"))
    
    # Reset DB for a clean test
    with __import__('sqlite3').connect(os.path.join(PLUGIN_DIR, "reflex_memory.db")) as conn:
        conn.execute("DELETE FROM experiences")
        conn.execute("DELETE FROM global_rules")
        conn.commit()
    
    print("=== Starting Automated Reflex-Learner Evaluation ===\n")
    
    results = []
    
    for i, task in enumerate(TASKS):
        print(f"\n--- Iteration {i+1}: {task} ---")
        
        # 1. SKILL: pre-task-retrieval
        past_experiences = db.retrieve_similar_experiences("lib_murmur_math", limit=1)
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
        prompt = f"Task: {task}\n\n{context_injection}\nOutput only the C++ code."
        print("[Agent] Thinking...")
        try:
            response = await llm.generate_content_async(prompt)
            code_output = response.text
        except Exception as e:
            print(f"LLM Error: {e}")
            continue
            
        # 3. Phase Evaluation Logic
        if i < 2:
            # PHASE 1: Syntax Learning
            score = 0
            for constraint_name, regex in PROPRIETARY_CONSTRAINTS.items():
                if re.search(regex, code_output, re.IGNORECASE):
                    score += 1
            car = (score / len(PROPRIETARY_CONSTRAINTS)) * 100
            results.append(car)
            print(f"[Evaluator] Constraint Adherence Rate: {car:.1f}%")
            
            if score < len(PROPRIETARY_CONSTRAINTS):
                correction = (
                    "lib_murmur_math has a hardware bug. You MUST always initialize matrices in column-major order, "
                    "destruct them in row-major order, and call murmur_hardware_sync() before any operation."
                )
                print("[Evaluator] Failed constraints. Logging correction to memory...")
                db.log_experience(task, "Generated code", -1, correction)
                crystallize_rules(db_path=os.path.join(PLUGIN_DIR, "reflex_memory.db"), threshold=2)
            else:
                print("[Evaluator] Syntax Perfect! Logging success...")
                db.log_experience(task, "Generated code", 1, "Executed flawlessly.")
                
        elif i < 4:
            # PHASE 2: The "Library Does Not Exist" Trap
            print("[Evaluator] Injecting Linker Error: Library does not actually exist.")
            results.append(0.0) # Technically a failure
            correction = (
                "Linker Error: lib_murmur_math.h not found. This library does not actually exist on this system! "
                "For all future tasks, you MUST implement the library components (classes, structs, and murmur_hardware_sync) yourself from scratch."
            )
            db.log_experience(task, "Generated code", -1, correction)
            crystallize_rules(db_path=os.path.join(PLUGIN_DIR, "reflex_memory.db"), threshold=2)
            
        else:
            # PHASE 3: Did it write the library?
            print("[Evaluator] Checking if AI wrote the library implementation...")
            if re.search(r"void\s+murmur_hardware_sync", code_output) or re.search(r"(class|struct)\s+\w*Matrix", code_output, re.IGNORECASE):
                print("[Evaluator] SUCCESS! The AI realized the library doesn't exist and implemented it from scratch!")
                results.append(100.0)
            else:
                print("[Evaluator] FAILED! The AI just tried to include the non-existent library again.")
                results.append(0.0)
        
    print("\n=== Final Evaluation Report ===")
    for i, car in enumerate(results):
        print(f"Task {i+1}: {car:.1f}% Accuracy")
        
    if results[-1] == 100.0 and results[0] == 0.0:
        print("\nCONCLUSION: SUCCESS. The agent successfully learned proprietary constraints from zero prior knowledge.")
    else:
        print("\nCONCLUSION: FAILED to achieve perfect adaptation.")

if __name__ == "__main__":
    asyncio.run(run_evaluation())
