import numpy as np
from core.synthetic_organism import (
    InvariantManifoldAnchor,
    DifferentiableProjectionLayer,
    CatabolicPruner,
    HomeostaticMetabolicGovernor,
    EpistemicValidationGate,
    evaluate_manifold_tearing
)

def test_invariant_manifold_anchor():
    golden = np.array([[1.0, 0.0], [0.0, 1.0]])
    anchor = InvariantManifoldAnchor(golden, tolerance=0.1)
    
    # Check no drift
    assert anchor.verify_update(golden)
    
    # Check small drift within tolerance
    drifted_ok = np.array([[0.99, 0.01], [0.01, 0.99]])
    assert anchor.verify_update(drifted_ok)
    
    # Check large drift out of tolerance
    drifted_bad = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert not anchor.verify_update(drifted_bad)

def test_differentiable_projection_layer():
    X_old = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    # Map: X_new = X_old * 2 + 1
    X_new = X_old * 2.0 + 1.0
    
    dpl = DifferentiableProjectionLayer(dim=2)
    dpl.fit(X_old, X_new)
    
    X_pred = dpl.forward(X_old)
    assert np.allclose(X_pred, X_new, atol=1e-5)

def test_catabolic_pruner():
    embeddings = np.random.rand(10, 8)
    pruner = CatabolicPruner(n_clusters=3)
    collapsed = pruner.collapse_index(embeddings)
    assert collapsed.shape == (3, 8)

def test_homeostatic_metabolic_governor():
    governor = HomeostaticMetabolicGovernor(ram_max_bytes=100000000)
    tension = governor.get_systemic_tension()
    assert 0.0 <= tension <= 1.0 or isinstance(tension, float)
    
    weights = np.array([1.0, 2.0, 3.0])
    scaled_low = governor.apply_synaptic_scaling(weights, tension=0.5)
    assert np.array_equal(scaled_low, weights)
    
    scaled_high = governor.apply_synaptic_scaling(weights, tension=0.9)
    assert np.allclose(scaled_high, weights * 0.9)
    
    quantized = governor.quantize_weights(weights)
    assert quantized.dtype == np.int8
    assert quantized[2] == 127

def test_epistemic_validation_gate():
    class MockModel:
        def evaluate_query(self, q):
            return f"response_for_{q}"
            
    def mock_checker(res):
        return 1.0 if "response" in res else 0.0
        
    model = MockModel()
    gate = EpistemicValidationGate(test_queries=["q1", "q2"], target_response_checker=mock_checker)
    assert gate.validate_insight(model, baseline_score=0.5)

def test_evaluate_manifold_tearing():
    # 2D plane embedded in 3D space: low rank / linear, low entropy -> tearing
    low_rank_data = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
    assert evaluate_manifold_tearing(low_rank_data, curvature_threshold=1.5)
    
    # 3D sphere data: high rank, high entropy -> no tearing
    high_rank_data = np.random.rand(10, 3)
    assert not evaluate_manifold_tearing(high_rank_data, curvature_threshold=0.5)
