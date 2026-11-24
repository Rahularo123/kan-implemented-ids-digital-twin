"""
src/explain.py
Explainable AI (XAI) Engine for the KAN Digital Twin.
Extracts learned B-spline curves and computes per-feature discrepancy attribution.
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from kan import KANLinear
from train import KANDigitalTwin

FEATURE_COLS = [
    'heart_rate', 'respiration_rate', 'spo2', 'bp_systolic', 'bp_diastolic',
    'temperature', 'packet_rate', 'packet_size', 'jitter', 'latency'
]

def load_system():
    """Loads the trained KAN model, the scaler, and the calibrated threshold."""
    with open('models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
        
    checkpoint = torch.load('models/kan_digital_twin.pth', map_location='cpu')
    model = KANDigitalTwin(
        in_features=checkpoint['in_features'],
        latent_dim=checkpoint['latent_dim'],
        grid_size=checkpoint['grid_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    threshold = checkpoint['threshold']
    return model, scaler, threshold

def explain_telemetry_sample(raw_sample_dict, model, scaler, threshold):
    """
    Performs root-cause analysis on a single incoming sample.
    Calculates per-feature reconstruction error (discrepancy).
    """
    df_sample = pd.DataFrame([raw_sample_dict])[FEATURE_COLS]
    x_scaled = scaler.transform(df_sample)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        reconstructed = model(x_tensor)
        # Calculate squared error for each feature individually
        feature_errors = ((x_tensor - reconstructed) ** 2).squeeze().numpy()
        total_error = float(feature_errors.mean())
        
    is_anomaly = total_error > threshold
    
    # Sort features from highest error to lowest
    attribution = sorted(
        zip(FEATURE_COLS, feature_errors),
        key=lambda x: x[1],
        reverse=True
    )
    
    return {
        'is_anomaly': is_anomaly,
        'anomaly_score': total_error,
        'threshold': threshold,
        'top_root_causes': attribution[:3],
        'all_feature_errors': dict(attribution)
    }

def plot_attack_attributions(model, scaler, threshold):
    """
    Generates a 4-panel bar chart showing which features triggered the alert
    for Normal, DoS, Data Injection, and Zero-Day attacks.
    """
    df = pd.read_csv('data/iomt_telemetry.csv')
    attack_types = ['Normal', 'DoS_Flooding', 'Data_Injection', 'ZeroDay_Replay']
    results = {}
    
    for att in attack_types:
        sample = df[df['label'] == att].iloc[0][FEATURE_COLS].to_dict()
        df_sample = pd.DataFrame([sample])[FEATURE_COLS]
        x_scaled = scaler.transform(df_sample)
        x_tensor = torch.tensor(x_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            reconstructed = model(x_tensor)
            errors = ((x_tensor - reconstructed) ** 2).squeeze().numpy()
            results[att] = errors

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()
    
    for i, att in enumerate(attack_types):
        ax = axes[i]
        errors = results[att]
        # Red bar if feature error exceeds threshold, blue if normal
        colors = ['#e74c3c' if e > threshold else '#3498db' for e in errors]
        ax.barh(FEATURE_COLS, errors, color=colors)
        ax.set_title(f"Telemetry Discrepancy: {att}", fontsize=12, fontweight='bold')
        ax.set_xlabel("Reconstruction Error (Residual)")
        ax.axvline(x=threshold, color='black', linestyle='--', label=f'Threshold ({threshold:.2f})')
        ax.legend()
        
    plt.tight_layout()
    os.makedirs('frontend/static', exist_ok=True)
    out_path = 'frontend/static/feature_attribution.png'
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Feature Attribution plot saved to: {out_path}")

def plot_kan_splines(model):
    """
    Extracts and plots the learned 1D B-spline activation curves
    from the edges of the KAN Encoder for 3 key features.
    """
    # Evaluate across 100 points in normalized feature space [-2.0, 2.0]
    x_eval = torch.linspace(-2.0, 2.0, 100).unsqueeze(-1)
    
    # 3 key features to visualize:
    key_features = {
        'Heart Rate (Biometric)': 0,
        'SpO2 Oxygen (Biometric)': 2,
        'Packet Rate (Network)': 6
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    with torch.no_grad():
        for idx, (feat_name, feat_idx) in enumerate(key_features.items()):
            ax = axes[idx]
            
            # Prepare an input matrix where only this feature varies
            single_x = torch.zeros(100, 10)
            single_x[:, feat_idx] = x_eval.squeeze()
            
            # Compute the B-spline basis activations
            bases = model.enc1.compute_b_splines(single_x)
            
            # Plot the resulting learned curves for the first 3 hidden nodes
            for node_idx in range(3):
                w = model.enc1.spline_weight[node_idx, feat_idx]
                curve = torch.einsum("bk,k->b", bases[:, feat_idx, :], w).numpy()
                ax.plot(x_eval.numpy(), curve, label=f"Hidden Node {node_idx+1}", linewidth=2)
                
            ax.set_title(f"Learned KAN Splines:\n{feat_name}", fontsize=11, fontweight='bold')
            ax.set_xlabel("Normalized Input Feature Value")
            ax.set_ylabel("Spline Activation ϕ(x)")
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(fontsize=9)
            
    plt.tight_layout()
    out_path = 'frontend/static/kan_splines.png'
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"KAN Spline Curves plot saved to: {out_path}")

if __name__ == '__main__':
    print("Generating Explainability (XAI) Reports...")
    model, scaler, threshold = load_system()
    
    # 1. Generate both visual plots
    plot_attack_attributions(model, scaler, threshold)
    plot_kan_splines(model)
    
    # 2. Test single-sample root-cause explanation on a Zero-Day sample
    print("\n--- Testing Real-Time Root Cause Explanation on a Zero-Day Sample ---")
    df = pd.read_csv('data/iomt_telemetry.csv')
    test_sample = df[df['label'] == 'ZeroDay_Replay'].iloc[0][FEATURE_COLS].to_dict()
    explanation = explain_telemetry_sample(test_sample, model, scaler, threshold)
    
    print(f"Is Anomaly?     : {explanation['is_anomaly']}")
    print(f"Anomaly Score   : {explanation['anomaly_score']:.4f} (Threshold: {explanation['threshold']:.4f})")
    print("Top Root Causes :")
    for feat, err in explanation['top_root_causes']:
        print(f"  -> {feat:18s}: Discrepancy Error = {err:.4f}")