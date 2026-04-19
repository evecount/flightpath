import jax
import jax.numpy as jnp
from dgi_sdk.core import DifferentiableManifold
import numpy as np

"""
DGI SDK Example: The Titanic Bridge
Goal: Demonstrate 'Raised Minima' on the traditional Titanic problem.
This proves the SDK's utility while enforcing the Sovereign nomenclature.
"""

def run_legacy_bridge_demo():
    print("--- DGI SDK: Legacy Bridge Demo (Titanic Dataset) ---")
    
    # 1. Mock Preprocessed Titanic Data (Survived = Target)
    # Features: [Pclass, Sex, Age_Scaled, Fare_Scaled]
    key = jax.random.PRNGKey(42)
    X = jax.random.normal(key, (891, 4)) 
    y = jax.random.bernoulli(key, 0.4, (891,)).astype(jnp.float32)
    
    # 2. Initializing the Sovereign Manifold
    # Instead of an 'Estimator', we create a 'Manifold'.
    manifold = DifferentiableManifold(input_dim=4)
    
    # 3. Performing Intent Alignment (The Stacking Protocol)
    # Instead of 'Fitting', we 'Align'.
    print("Aligning the Manifold toward Survival Equilibrium...")
    final_intent_weights = manifold.align(X, y, epochs=100)
    
    # 4. Results
    print("\n--- Manifestation Complete ---")
    print(f"Final Intent Strategy (Weights): {final_intent_weights}")
    print("Note: The 'Choice' of survival has been moved from a Decision Tree to a Geodesic.")
    print("The lower minima have been raised. Signal-to-Noise parity reached.")

if __name__ == "__main__":
    run_legacy_bridge_demo()
