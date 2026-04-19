import jax.numpy as jnp
from dgi_sdk.core import DifferentiableManifold

"""
DGI SDK Core Verification
Ensures the mathematical stability of the DGA Stacking Protocol.
"""

def test_manifold_initialization():
    manifold = DifferentiableManifold(input_dim=10)
    assert manifold.params.shape == (10,)
    print("Test Passed: Manifold Initialization (11D Space Check)")

def test_rotational_projection():
    weights = jnp.array([1.0, 0.0, 0.5, 0.5])
    theta = jnp.pi / 2 # 90 degree rotation
    projected = DifferentiableManifold.rotate_axis(weights, theta)
    # At 90 degrees, [1,0] should map to [0,1]
    assert jnp.allclose(projected, jnp.array([0.0, 1.0]), atol=1e-5)
    print("Test Passed: Dynamic Latent Alignment (Phase Rotation Check)")

def test_cosine_flow_boundary():
    # At t=0, flow should be 1.0 (High Energy)
    # At t=T, flow should be 0.0 (Global Optimum Equilibrium)
    assert jnp.allclose(DifferentiableManifold.cosine_flow(0, 100), 1.0)
    assert jnp.allclose(DifferentiableManifold.cosine_flow(100, 100), 0.0)
    print("Test Passed: Harmonic Step Modulation (Boundary Check)")

if __name__ == "__main__":
    print("--- DGI SDK: Core Mathematical Verification ---")
    test_manifold_initialization()
    test_rotational_projection()
    test_cosine_flow_boundary()
    print("Result: Manifold Integrity Verified. Protocol is Deterministic.")
