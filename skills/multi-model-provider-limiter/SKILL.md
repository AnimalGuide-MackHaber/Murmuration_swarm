---
name: multi-model-provider-limiter
description: "Manage token rate limiters, retries, and settings for Ollama, NVIDIA, and Gemini LLM providers."
---

# Multi-Model Provider & Limiter Skill

## Overview
This skill provides instructions on configuring model endpoints, credentials, and managing rate limit parameters (RPM/TPM) to run parallelized agent loops safely without triggering rate-limit errors (HTTP 429).

## Codebase References
- [rate_limiter.py](file:///Users/mike/Documents/code/Murmur_swarm/core/rate_limiter.py) - Implements token-bucket rate limiting for concurrent calls.
- [llm_retry.py](file:///Users/mike/Documents/code/Murmur_swarm/core/llm_retry.py) - Implements exponential backoffs and retry hooks.
- [settings_manager.py](file:///Users/mike/Documents/code/Murmur_swarm/core/settings_manager.py) - Configuration models for API endpoints.

## Usage Guide
1. Ensure API keys (`GEMINI_API_KEY`, etc.) are configured.
2. Monitor simulation logs for `429 Too Many Requests` status codes.
3. If errors persist, verify token calculations or increase backoff delays.

## Tooling & Scripts
- Run rate limit tests: `python skills/multi-model-provider-limiter/scripts/rate_limit_tester.py` (when implemented).
