import jax.numpy as jnp
from dgi_sdk.core import DifferentiableManifold
from dgi_sdk.inquiry import ActiveInquiryAligner

"""
DGI SDK Example: Sample Task Grid (The Meadow)
Goal: Regularizing a grid of random missions into 'Prime Geodesics'.
Demonstrates the 'True Genius' of training the user through inquiry.
"""

def run_task_grid_demo():
    print("--- DGI SDK: Sample Task Grid (The Meadow) ---")
    
    # 1. The 'Noisy' Meadow: Random tasks from different sources
    # Features: [Budget, Complexity, Impact, Sovereignty]
    tasks = jnp.array([
        [0.1, 0.8, 0.2, 0.9], # Task A: (Noise)
        [0.7, 0.2, 0.8, 0.4], # Task B: (Potential Signal)
        [0.5, 0.5, 0.5, 0.5], # Task C: (Neutral)
    ])
    
    # 2. Initializing the Aligner
    manifold = DifferentiableManifold(input_dim=4)
    aligner = ActiveInquiryAligner(manifold)
    
    # 3. The 'True Genius' Loop: Active Inquiry
    print("\n[ACTIVE INQUIRY] Analyzing intent variance...")
    questions = aligner.generate_inquiry_grid(tasks)
    
    for q in questions:
        print(f"\nINQUIRY: {q['question']}")
        print(f"OPTIONS: {q['options']}")
        # Simulating User Response (The Pivot)
        user_response = q['options'][0] 
        print(f"USER RESPONSE: {user_response}")
        
    # 4. Final Alignment
    print("\n[SUCCESS] Manifold regularized toward Prime Geodesics.")
    final_geo = manifold.align(tasks, jnp.array([1.0, 1.0, 0.0]), epochs=50)
    
    print("\n--- Meadow Finalized ---")
    print("The 'Kitties' are now moving in mathematical necessity.")
    print(f"Settlement Coordinates: {final_geo}")

if __name__ == "__main__":
    run_task_grid_demo()
