
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.swarm_io import SwarmIO

def test_save_paper():
    content = "# Test Paper Content"
    sim_id = "test_sim_123"
    print(f"Testing save_research_paper with sim_id={sim_id}...")
    
    path = SwarmIO.save_research_paper(content, sim_id)
    
    print(f"Returned path: {path}")
    
    if os.path.exists(path):
        print("SUCCESS: File exists.")
        with open(path, 'r') as f:
            saved_content = f.read()
            if saved_content == content:
                print("SUCCESS: Content matches.")
            else:
                print("FAILURE: Content mismatch.")
    else:
        print("FAILURE: File not found.")

if __name__ == "__main__":
    test_save_paper()
