"""
backend/main.py
FastAPI Backend for KAN Digital Twin IoMT Intrusion Detection System.
Directly loads PyTorch KAN model in memory and serves real-time inference,
root-cause XAI analysis, and live telemetry simulation.
"""

import sys
import os
import random
import pickle
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# Add src to path to import KAN modules
SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from kan import KANLinear
from train import KANDigitalTwin
from explain import explain_telemetry_sample, FEATURE_COLS

app = FastAPI(
    title="KAN-Twin: IoMT Intrusion Detection API",
    description="Intrinsically Explainable Kolmogorov-Arnold Network Digital Twin for Zero-Day Threat Detection",
    version="1.0.0"
)

# Enable CORS for frontend flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------
# Global State: Load Model, Scaler, Dataset, and Threshold into RAM
# ---------------------------------------------------------------------
STATE = {
    'model': None,
    'scaler': None,
    'threshold': 1.1492,
    'df_data': None
}

@app.on_event("startup")
def startup_load_models():
    print("Loading KAN Digital Twin System into memory...")
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))
    
    scaler_path = os.path.join(models_dir, 'scaler.pkl')
    checkpoint_path = os.path.join(models_dir, 'kan_digital_twin.pth')
    data_path = os.path.join(data_dir, 'iomt_telemetry.csv')

    if not os.path.exists(scaler_path) or not os.path.exists(checkpoint_path):
        raise RuntimeError("Model checkpoint or scaler not found. Please train model first.")

    with open(scaler_path, 'rb') as f:
        STATE['scaler'] = pickle.load(f)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = KANDigitalTwin(
        in_features=checkpoint['in_features'],
        latent_dim=checkpoint['latent_dim'],
        grid_size=checkpoint['grid_size']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    STATE['model'] = model
    STATE['threshold'] = float(checkpoint.get('threshold', 1.1492))

    if os.path.exists(data_path):
        STATE['df_data'] = pd.read_csv(data_path)
        print(f"Loaded {len(STATE['df_data'])} telemetry rows for live simulation.")
        
    print(f"System ready! Calibrated Anomaly Threshold: {STATE['threshold']:.4f}")

# ---------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------
class TelemetryInput(BaseModel):
    heart_rate: float = Field(75.0, description="Heart Rate in BPM")
    respiration_rate: float = Field(16.0, description="Respiration in breaths/min")
    spo2: float = Field(98.0, description="Blood Oxygen %")
    bp_systolic: float = Field(120.0, description="Systolic Blood Pressure (mmHg)")
    bp_diastolic: float = Field(80.0, description="Diastolic Blood Pressure (mmHg)")
    temperature: float = Field(98.6, description="Body Temperature (°F)")
    packet_rate: float = Field(35.0, description="Packets per second")
    packet_size: float = Field(320.0, description="Mean packet size (bytes)")
    jitter: float = Field(2.0, description="Packet jitter (ms)")
    latency: float = Field(15.0, description="Network roundtrip latency (ms)")

# ---------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------
@app.get("/api/status")
def get_system_status():
    """Returns model health, calibrated threshold, and architecture specs."""
    return {
        "status": "online",
        "model_architecture": "Kolmogorov-Arnold Network (KAN) Autoencoder",
        "layer_spec": "10 -> 6 -> 3 -> 6 -> 10",
        "threshold": STATE['threshold'],
        "features": FEATURE_COLS,
        "device": "cpu"
    }

@app.post("/api/predict")
def predict_telemetry(payload: TelemetryInput):
    """
    Feeds an incoming telemetry packet into the KAN Digital Twin.
    Computes reconstruction error (MSE) and flags attacks/zero-days.
    """
    if STATE['model'] is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    sample_dict = payload.dict()
    df_sample = pd.DataFrame([sample_dict])[FEATURE_COLS]
    
    x_scaled = STATE['scaler'].transform(df_sample)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32)

    with torch.no_grad():
        reconstructed = STATE['model'](x_tensor)
        # Compute total MSE across all 10 features
        error = torch.mean((x_tensor - reconstructed) ** 2, dim=1).item()

    is_anomaly = bool(error > STATE['threshold'])
    
    # Calculate twin state divergence percentage
    twin_divergence_pct = min(100.0, (error / STATE['threshold']) * 50.0 if not is_anomaly else 50.0 + (error / (STATE['threshold'] * 10)) * 50.0)

    return {
        "is_anomaly": is_anomaly,
        "status": "ANOMALY_BREACH" if is_anomaly else "TWIN_SYNCHRONIZED",
        "anomaly_score": round(error, 4),
        "threshold": round(STATE['threshold'], 4),
        "twin_divergence_pct": round(twin_divergence_pct, 1),
        "input": sample_dict
    }

@app.post("/api/explain")
def explain_telemetry(payload: TelemetryInput):
    """
    Computes root-cause analysis: measures individual reconstruction residual
    per feature to determine which biometric or network features violated the Twin.
    """
    if STATE['model'] is None:
        raise HTTPException(status_code=500, detail="Model is not loaded.")

    sample_dict = payload.dict()
    explanation = explain_telemetry_sample(
        sample_dict, STATE['model'], STATE['scaler'], STATE['threshold']
    )
    
    return {
        "is_anomaly": explanation['is_anomaly'],
        "anomaly_score": round(explanation['anomaly_score'], 4),
        "threshold": round(explanation['threshold'], 4),
        "top_root_causes": [
            {"feature": feat, "discrepancy": round(float(err), 4)}
            for feat, err in explanation['top_root_causes']
        ],
        "all_feature_discrepancies": {
            feat: round(float(err), 4)
            for feat, err in explanation['all_feature_errors'].items()
        }
    }

@app.get("/api/telemetry/sample")
def get_sample_telemetry(scenario: str = "Normal"):
    """
    Returns a sample from the real benchmark dataset for a specific scenario:
    - Normal
    - DoS_Flooding
    - Data_Injection
    - ZeroDay_Replay
    """
    if STATE['df_data'] is None:
        raise HTTPException(status_code=500, detail="Dataset not loaded.")

    matched = STATE['df_data'][STATE['df_data']['label'].str.lower() == scenario.lower()]
    if len(matched) == 0:
        matched = STATE['df_data'][STATE['df_data']['label'] == 'Normal']

    sample_row = matched.sample(1).iloc[0]
    sample_data = {col: float(sample_row[col]) for col in FEATURE_COLS}
    
    return {
        "scenario": sample_row['label'],
        "telemetry": sample_data
    }

# -------------------------------------------------------------
# Serve Frontend Assets & Static Images
# -------------------------------------------------------------
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

if os.path.exists(FRONTEND_DIR):
    static_dir = os.path.join(FRONTEND_DIR, 'static')
    if os.path.exists(static_dir):
        app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    # Mount frontend directory for js and css
    app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")

    @app.get("/")
    def serve_frontend_index():
        index_file = os.path.join(FRONTEND_DIR, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"message": "Frontend index.html not yet created."}
