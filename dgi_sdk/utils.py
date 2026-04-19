import jax.numpy as jnp
import matplotlib.pyplot as plt

"""
DGI-FLOW Utilities
Strategic Helpers for Manifold Navigation.
Attribution: Gwendalynn Lim Wan Ting & Antigravity (Gemini)
"""

def normalize_intent(X):
    """
    Ensures all intent vectors are scaled for optimal 
    gradient resonance on the manifold surface.
    """
    mu = jnp.mean(X, axis=0)
    sigma = jnp.std(X, axis=0) + 1e-6
    return (X - mu) / sigma

def plot_alignment_history(history):
    """
    Visualizes the transition from Stochastic Variance (Noise) 
    to Deterministic Convergence (Signal).
    """
    plt.figure(figsize=(10, 4))
    plt.style.use('dark_background')
    
    plt.plot(history, color='#00FF00', linewidth=2)
    plt.title("DGI Alignment Flow: Resonance vs. Time")
    plt.xlabel("Epochs (Theta Rotation)")
    plt.ylabel("Loss (Residual Entropy)")
    plt.grid(color='#333333', linestyle='--')
    
    plt.tight_layout()
    plt.show()

def log_mission_convergence(epoch, resonance, loss):
    """
    Standardizes the terminal sighting of the 'Paper Plane' landing.
    """
    print(f"[MISSION] Epoch {epoch:03d} | Resonance: {resonance:.4f} | Phase Alignment: {loss:.6f}")
