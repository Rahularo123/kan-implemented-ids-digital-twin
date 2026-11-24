"""
src/train.py
Trains the KAN-based Behavioral Digital Twin for Zero-Day Intrusion Detection.
Evaluated under a strict Semi-Supervised Zero-Day / Leave-One-Attack-Out Protocol.
"""

import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from kan import KANLinear

# ---------------------------------------------------------------------
# Chunk 1: The KAN Digital Twin Autoencoder Architecture
# ---------------------------------------------------------------------
class KANDigitalTwin(nn.Module):
    def __init__(self, in_features=10, latent_dim=3, grid_size=5):
        super(KANDigitalTwin, self).__init__()
        
        # Encoder: Compresses 10 features down to latent_dim (3)
        self.enc1 = KANLinear(in_features, 6, grid_size=grid_size)
        self.enc2 = KANLinear(6, latent_dim, grid_size=grid_size)
        
        # Decoder: Reconstructs from latent_dim (3) back to 10 features
        self.dec1 = KANLinear(latent_dim, 6, grid_size=grid_size)
        self.dec2 = KANLinear(6, in_features, grid_size=grid_size)

    def forward(self, x):
        # Compression
        z = self.enc1(x)
        z = self.enc2(z)
        # Reconstruction
        out = self.dec1(z)
        out = self.dec2(out)
        return out

    def get_reconstruction_error(self, x):
        """Calculates Mean Squared Error per sample (Anomaly Score)"""
        with torch.no_grad():
            reconstructed = self.forward(x)
            # MSE along feature axis: shape (batch_size,)
            error = torch.mean((x - reconstructed) ** 2, dim=1)
        return error

# ---------------------------------------------------------------------
# Chunk 2: Data Loading & Zero-Day Partitioning
# ---------------------------------------------------------------------
def prepare_data(data_path='data/iomt_telemetry.csv', random_seed=42):
    print("Loading IoMT dataset...")
    df = pd.read_csv(data_path)
    
    feature_cols = [
        'heart_rate', 'respiration_rate', 'spo2', 'bp_systolic', 'bp_diastolic',
        'temperature', 'packet_rate', 'packet_size', 'jitter', 'latency'
    ]
    
    # Separate Normal traffic from Attacks
    df_normal = df[df['label'] == 'Normal'].copy()
    df_attacks = df[df['label'] != 'Normal'].copy()
    
    # Split Normal into: 80% Training (Digital Twin baseline), 20% Test
    normal_train, normal_test = train_test_split(df_normal, test_size=0.20, random_state=random_seed)
    
    # Scale features using only the Normal training data (zero data leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(normal_train[feature_cols])
    
    # Test set combines remaining 20% Normal + ALL Attacks (including Zero-Day!)
    df_test_combined = pd.concat([normal_test, df_attacks], ignore_index=True)
    X_test_scaled = scaler.transform(df_test_combined[feature_cols])
    y_test_labels = df_test_combined['label'].values
    y_test_binary = df_test_combined['is_attack'].values
    
    # Save the scaler so the Backend API can use it later
    os.makedirs('models', exist_ok=True)
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        
    print(f"Training Set (Normal only): {X_train_scaled.shape[0]} samples")
    print(f"Test Set (Normal + Attacks): {X_test_scaled.shape[0]} samples")
    print("Test Breakdown:")
    print(df_test_combined['label'].value_counts())
    
    return X_train_scaled, X_test_scaled, y_test_binary, y_test_labels, feature_cols

# ---------------------------------------------------------------------
# Chunk 3: Training the KAN Digital Twin
# ---------------------------------------------------------------------
def train_kan_twin(X_train, epochs=15, batch_size=64, lr=0.005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on device: {device}")
    
    model = KANDigitalTwin(in_features=10, latent_dim=3, grid_size=5).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    tensor_data = TensorDataset(torch.tensor(X_train, dtype=torch.float32))
    loader = DataLoader(tensor_data, batch_size=batch_size, shuffle=True)
    
    print("\n--- Training KAN Digital Twin (Learning Normal Baseline) ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in loader:
            x_batch = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(x_batch)
            
        avg_loss = total_loss / len(X_train)
        if epoch % 3 == 0 or epoch == 1:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Reconstruction Loss (MSE): {avg_loss:.6f}")
            
    return model, device

# ---------------------------------------------------------------------
# Chunk 4: Threshold Calibration & Zero-Day Evaluation
# ---------------------------------------------------------------------
def evaluate_model(model, device, X_train, X_test, y_test_binary, y_test_labels):
    model.eval()
    
    # 1. Calculate reconstruction error on normal training samples
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    train_errors = model.get_reconstruction_error(X_train_tensor).cpu().numpy()
    
    # Threshold = 98th percentile of normal reconstruction error
    # (Meaning: 98% of normal data falls below this threshold; anything higher is an anomaly)
    threshold = float(np.percentile(train_errors, 98))
    print(f"\nCalibrated Anomaly Threshold: {threshold:.6f}")
    
    # 2. Evaluate on the Test Set
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    test_errors = model.get_reconstruction_error(X_test_tensor).cpu().numpy()
    
    # Predicted 1 (Attack) if error > threshold, else 0 (Normal)
    y_pred_binary = (test_errors > threshold).astype(int)
    
    print("\n" + "="*50)
    print("      OVERALL INTRUSION DETECTION METRICS")
    print("="*50)
    print(classification_report(y_test_binary, y_pred_binary, target_names=['Normal', 'Attack'], digits=4))
    
    roc_auc = roc_auc_score(y_test_binary, test_errors)
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # 3. Dedicated Zero-Day Attack Analysis
    print("\n" + "="*50)
    print("      ZERO-DAY THREAT DETECTION PERFORMANCE")
    print("="*50)
    df_eval = pd.DataFrame({
        'label': y_test_labels,
        'true_attack': y_test_binary,
        'predicted_attack': y_pred_binary,
        'anomaly_score': test_errors
    })
    
    for attack_type in df_eval['label'].unique():
        sub = df_eval[df_eval['label'] == attack_type]
        detection_rate = (sub['predicted_attack'] == 1).mean() * 100 if attack_type != 'Normal' else (sub['predicted_attack'] == 0).mean() * 100
        metric_name = "Accuracy" if attack_type == 'Normal' else "Detection Rate (Recall)"
        print(f"[{attack_type:15s}] -> {metric_name}: {detection_rate:.2f}% (Avg Anomaly Score: {sub['anomaly_score'].mean():.4f})")
        
    # Save trained model weights & metadata
    torch.save({
        'model_state_dict': model.state_dict(),
        'threshold': threshold,
        'in_features': 10,
        'latent_dim': 3,
        'grid_size': 5
    }, 'models/kan_digital_twin.pth')
    print("\nTrained model and threshold saved to 'models/kan_digital_twin.pth'")

if __name__ == '__main__':
    X_train, X_test, y_test_binary, y_test_labels, feature_cols = prepare_data()
    model, device = train_kan_twin(X_train)
    evaluate_model(model, device, X_train, X_test, y_test_binary, y_test_labels)