---
name: maai-topological-governor
description: "Configure, analyze, and troubleshoot the Topological Data Analysis (TDA) and Criticality Governor loop in the MAAI system."
---

# MAAI Topological Governor Skill

## Overview
This skill allows agents to analyze the topological states of the MAAI knowledge database, prevent semantic collapse/static convergence, and handle stagnation and divergence states.

## Codebase References
- [governor.py](file:///Users/mike/Documents/code/Murmur_swarm/experimental/maai_system/governor.py) - Computes telemetry (Betti-1, Spectral Radius) and controls VSE/XOH states.
- [agents.py](file:///Users/mike/Documents/code/Murmur_swarm/experimental/maai_system/agents.py) - Implements the iconoclast pruning logic and Void-State Epistemology (VSE) context retrieval.

## Usage Guide
1. Inspect current database stats and embeddings.
2. Run the topological debugger to get Spectral Radius and Betti-1 numbers.
3. If stagnation is detected: lower visitation thresholds and trigger XOH injection.
4. If divergence is detected: pause producers and run aggressive iconoclast pruning.

## Tooling & Scripts
- Run `python experimental/maai_system/governor.py` to evaluate telemetry.
- Run `python skills/maai-topological-governor/scripts/tda_debugger.py --db_path .milvus_maai.db --threshold 0.8` (when implemented) to debug topological structures.
