import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import eigs
import networkx as nx
from database import client, CORE_KNOWLEDGE, XOH_SANDBOX, get_random_nodes
from config import config

def inject_xoh_nodes(count: int = 5):
    """Injects nodes from XOH_SANDBOX into CORE_KNOWLEDGE."""
    print(f"Injecting {count} nodes from XOH_SANDBOX...")
    xoh_nodes = get_random_nodes(XOH_SANDBOX, count)
    if not xoh_nodes:
        print("No nodes found in XOH_SANDBOX.")
        return
        
    # Prepare for insertion into CORE_KNOWLEDGE
    # We might want to keep the same embeddings but give them new IDs or keep them
    # For now, we'll just insert them as is.
    client.insert(collection_name=CORE_KNOWLEDGE, data=xoh_nodes)
    print(f"Successfully injected {len(xoh_nodes)} nodes.")

def extract_betti_1(adjacency_matrix):
    """Extracts Betti 1 numbers (number of independent cycles) using networkx."""
    try:
        # Create graph from adjacency matrix
        G = nx.from_numpy_array(adjacency_matrix)
        
        # Betti 1 number of a graph = Edges - Vertices + Connected Components
        edges = G.number_of_edges()
        vertices = G.number_of_nodes()
        components = nx.number_connected_components(G)
        
        betti_1 = edges - vertices + components
        return betti_1
    except Exception as e:
        print(f"Error calculating Betti-1: {e}")
        return 0

def calculate_telemetry(embeddings_matrix):
    """Calculates Spectral Radius and Betti-1 from embeddings."""
    if len(embeddings_matrix) == 0:
        return 0.0, 0
        
    # 1. Proxy Spectral Radius
    # Create adjacency matrix based on similarity threshold
    similarity_matrix = cosine_similarity(embeddings_matrix)
    adjacency_matrix = np.where(similarity_matrix > 0.8, 1, 0)
    
    spectral_radius = 0.0
    if len(embeddings_matrix) > 1:
        # Calculate largest eigenvalue
        # Use float64 to avoid type issues in scipy
        try:
            # k must be strictly less than the dimension of the matrix
            k_val = min(1, len(embeddings_matrix) - 1)
            if k_val > 0:
                eigenvalues, _ = eigs(adjacency_matrix.astype(float), k=k_val, which='LM')
                spectral_radius = np.abs(eigenvalues[0])
        except Exception as e:
            print(f"Error calculating spectral radius: {e}")
            pass

    # 2. Proxy Betti Numbers (H1) using networkx graph
    betti_1 = 0
    if len(embeddings_matrix) >= 3:
        betti_1 = extract_betti_1(adjacency_matrix)

    return spectral_radius, betti_1

def control_logic():
    """Evaluates telemetry and adjusts system state."""
    # Fetch all embeddings
    try:
        results = client.query(
            collection_name=CORE_KNOWLEDGE,
            filter="id != ''",
            output_fields=["embedding"],
            limit=1000
        )
        if not results:
            return "Insufficient data for telemetry."
            
        embeddings = np.array([res['embedding'] for res in results])
        spectral_radius, betti_1 = calculate_telemetry(embeddings)
        
        print(f"Telemetry -> Spectral Radius: {spectral_radius:.2f}, Betti-1: {betti_1}")
        
        if spectral_radius < 1.0 and betti_1 == 0:
            # System State: The graph is too dense/homogeneous.
            action = "Stagnation detected. Lowering visitation_threshold. Injecting XOH nodes."
            config.set_visitation_threshold(max(5, config.visitation_threshold - 10))
            config.set_producers_paused(False)
            inject_xoh_nodes(5)
            return action
        elif spectral_radius > 1.5:
            # System State: The graph is fragmenting.
            action = "Divergence detected. Pausing Producers. Triggering Iconoclast (0.85)."
            config.set_producers_paused(True)
            from agents import run_iconoclast
            run_iconoclast(0.85)
            return action
        else:
            # Nominal state: Ensure producers are running and reset threshold slowly
            if config.producers_paused:
                config.set_producers_paused(False)
            if config.visitation_threshold < 50:
                config.set_visitation_threshold(min(50, config.visitation_threshold + 5))
            return "System nominal."
            
    except Exception as e:
        return f"Governor Error: {e}"
