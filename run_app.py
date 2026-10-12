import os
import sys
import subprocess
import shutil

def run_command(command, cwd=None):
    print(f"Executing: {' '.join(command)} (cwd: {cwd or '.'})")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Print output in real time
    for line in iter(process.stdout.readline, ""):
        print(f"  {line.strip()}")
    
    process.stdout.close()
    return_code = process.wait()
    if return_code != 0:
        print(f"Error: Command failed with exit code {return_code}")
        return False
    return True

def main():
    print("==========================================================")
    # Vibrant colors for terminal output
    print("\033[94m   🐝 Swarm Intelligence Control Tower - Startup Script 🐝\033[0m")
    print("==========================================================")

    # 1. Verify/Install Python dependencies
    print("\n\033[93m[1/3] Verifying Python backend dependencies...\033[0m")
    try:
        import fastapi
        import uvicorn
        print("  FastAPI and Uvicorn are already installed.")
    except ImportError:
        print("  FastAPI/Uvicorn not found. Installing from requirements.txt...")
        # Try using the current virtual environment's pip if available, else system pip
        pip_cmd = [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"]
        if not run_command(pip_cmd):
            print("\033[91mFailed to install requirements. Please run 'pip install -r requirements.txt' manually.\033[0m")
            sys.exit(1)

    # 2. Check and build frontend if needed
    print("\n\033[93m[2/3] Checking frontend static assets...\033[0m")
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    dist_dir = os.path.join(frontend_dir, "dist")
    node_modules_dir = os.path.join(frontend_dir, "node_modules")

    build_needed = not os.path.exists(dist_dir) or "--force-build" in sys.argv

    if build_needed:
        print("  Building frontend assets. This may take a moment...")
        # Install node modules if missing
        if not os.path.exists(node_modules_dir):
            print("  node_modules not found. Running 'npm install'...")
            if not run_command(["npm", "install"], cwd=frontend_dir):
                print("\033[91mFailed to run 'npm install'. Make sure Node.js is installed.\033[0m")
                sys.exit(1)

        # Run npm build
        print("  Running 'npm run build'...")
        if not run_command(["npm", "run", "build"], cwd=frontend_dir):
            print("\033[91mFailed to build frontend application.\033[0m")
            sys.exit(1)
        print("  Frontend build completed successfully!")
    else:
        print("  Static assets found in frontend/dist. Skipping build.")
        print("  (Tip: Run with '--force-build' to force a rebuild of the React app)")

    # 3. Spin up backend FastAPI server
    print("\n\033[93m[3/3] Starting FastAPI backend on http://localhost:8501...\033[0m")
    
    try:
        import uvicorn
        uvicorn.run("backend:app", host="0.0.0.0", port=8501, reload=True)
    except KeyboardInterrupt:
        print("\n\033[92mControl Tower stopped by user. Goodbye!\033[0m")

if __name__ == "__main__":
    main()
