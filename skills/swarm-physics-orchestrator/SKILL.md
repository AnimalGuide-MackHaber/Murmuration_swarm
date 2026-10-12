---
name: swarm-physics-orchestrator
description: "Tune and simulate the 3D physics-based spatial boids engine and manage LangGraph orchestrator steps."
---

# Swarm Physics Orchestrator Skill

## Overview
This skill guides the agent in setting, tuning, and validating the 3D vector-based physics parameters (Separation, Cohesion, Alignment, Goal Seeking) which govern agent coordinates and the resulting influence matrix.

## Codebase References
- [swarm_physics.py](file:///Users/mike/Documents/code/Murmur_swarm/core/swarm_physics.py) - Implements boids algorithm and calculates the quadratic influence matrix.
- [swarm_orchestrator.py](file:///Users/mike/Documents/code/Murmur_swarm/core/swarm_orchestrator.py) - Coordinates step lifecycles and LangGraph state.
- [swarm_schema.py](file:///Users/mike/Documents/code/Murmur_swarm/core/swarm_schema.py) - Defines spatial data structures (Vector3, runtime state).

## Usage Guide
1. Inspect the physics parameters inside `swarm_settings.json` or through settings APIs.
2. Run trajectory simulations to ensure agent coordinate paths converge without spinning out.
3. Balance the weights of separation and cohesion to encourage localized agent clusters before consensus is reached.

## Tooling & Scripts
- Run physics simulations: `python skills/swarm-physics-orchestrator/scripts/simulate_trajectories.py` (when implemented) to test trajectory steps.
