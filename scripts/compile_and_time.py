#!/usr/bin/env python3
"""
compile_and_time.py
------------------
Compiles a C++ source file with aggressive optimisations and returns the wall‑clock
runtime in milliseconds. If compilation fails, the stderr output is written to a
log file alongside the source (``<src>.compile.log``) and ``None`` is returned.
"""

import sys, subprocess, shlex, pathlib

def compile_and_time(src_path: str):
    # Build output binary path
    src = pathlib.Path(src_path)
    exe = src.with_suffix('.out')

    # Compile with clang (or g++) – we use the system's default C++ compiler
    compile_cmd = f"c++ -O3 -march=native -std=c++20 {shlex.quote(str(src))} -o {shlex.quote(str(exe))}"
    compile_res = subprocess.run(compile_cmd, shell=True, capture_output=True, text=True)
    if compile_res.returncode != 0:
        # Write compilation errors to a log file for debugging
        log_path = src.with_suffix('.compile.log')
        with open(log_path, 'w') as f:
            f.write("--- Compilation Failed ---\n")
            f.write(compile_res.stderr)
        return None

    # Run the binary once and measure elapsed time (seconds -> ms)
    start_cmd = f"/usr/bin/time -p {shlex.quote(str(exe))}"
    # Use time command to get real time (seconds). We'll parse its output.
    run_res = subprocess.run(start_cmd, shell=True, capture_output=True, text=True)
    if run_res.returncode != 0:
        # Runtime failure – also log
        log_path = src.with_suffix('.run.log')
        with open(log_path, 'w') as f:
            f.write("--- Runtime Failed ---\n")
            f.write(run_res.stderr)
        return None

    # Parse the 'real' line from /usr/bin/time output
    real_time = None
    for line in run_res.stderr.splitlines():
        if line.startswith('real'):
            parts = line.split()
            if len(parts) == 2:
                try:
                    real_time = float(parts[1])
                except ValueError:
                    pass
    if real_time is None:
        # fallback: use wall‑clock from Python timing (less precise)
        import time
        t0 = time.time()
        subprocess.run(str(exe), shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        real_time = time.time() - t0
    # Convert seconds to milliseconds
    return real_time * 1000

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: compile_and_time.py <source.cpp>")
        sys.exit(1)
    runtime_ms = compile_and_time(sys.argv[1])
    if runtime_ms is not None:
        print(f"{runtime_ms:.2f}")
    else:
        sys.exit(1)
