"""
src/dataset_generator.py
Generates a realistic Cyber-Physical IoMT Benchmark Dataset based on 
published WUSTL-EHMS-2020 statistical distributions.
"""

import numpy as np
import pandas as pd
import os

def generate_iomt_dataset(n_samples=25000, random_seed=42):
    np.random.seed(random_seed)
    
    # 70% Normal, 10% DoS Flooding, 10% False Data Injection, 10% Zero-Day Replay
    n_normal = int(n_samples * 0.70)
    n_dos = int(n_samples * 0.10)
    n_injection = int(n_samples * 0.10)
    n_zeroday = n_samples - (n_normal + n_dos + n_injection)
    
    print(f"Generating IoMT Telemetry: Normal={n_normal}, DoS={n_dos}, Injection={n_injection}, Zero-Day={n_zeroday}")
    
    # -------------------------------------------------------------
    # 1. BENIGN / NORMAL SAMPLES (The baseline for our Digital Twin)
    # -------------------------------------------------------------
    # Physical/Biometric features
    hr_normal = np.random.normal(loc=75.0, scale=8.0, size=n_normal)         # Heart Rate (BPM)
    resp_normal = np.random.normal(loc=16.0, scale=2.5, size=n_normal)       # Respiration Rate
    spo2_normal = np.random.uniform(low=96.0, high=99.5, size=n_normal)       # SpO2 %
    bp_sys_normal = hr_normal * 0.8 + np.random.normal(loc=65, scale=5, size=n_normal) # Systolic BP
    bp_dia_normal = bp_sys_normal * 0.65 + np.random.normal(loc=0, scale=3, size=n_normal) # Diastolic BP
    temp_normal = np.random.normal(loc=98.6, scale=0.4, size=n_normal)        # Temperature (°F)
    
    # Cyber/Network features
    pkt_rate_normal = np.random.normal(loc=35.0, scale=4.0, size=n_normal)   # Packets / sec
    pkt_size_normal = np.random.normal(loc=320.0, scale=25.0, size=n_normal) # Mean packet size
    jitter_normal = np.random.exponential(scale=2.0, size=n_normal)          # Jitter (ms)
    latency_normal = np.random.normal(loc=15.0, scale=3.0, size=n_normal)     # Latency (ms)
    
    df_normal = pd.DataFrame({
        'heart_rate': hr_normal,
        'respiration_rate': resp_normal,
        'spo2': spo2_normal,
        'bp_systolic': bp_sys_normal,
        'bp_diastolic': bp_dia_normal,
        'temperature': temp_normal,
        'packet_rate': pkt_rate_normal,
        'packet_size': pkt_size_normal,
        'jitter': jitter_normal,
        'latency': latency_normal,
        'label': 'Normal',
        'is_attack': 0
    })
    
    # -------------------------------------------------------------
    # 2. KNOWN ATTACK: DoS FLOODING (Network anomaly)
    # -------------------------------------------------------------
    hr_dos = np.random.normal(loc=76.0, scale=9.0, size=n_dos)
    resp_dos = np.random.normal(loc=16.2, scale=2.6, size=n_dos)
    spo2_dos = np.random.uniform(low=95.0, high=99.0, size=n_dos)
    bp_sys_dos = np.random.normal(loc=122, scale=6, size=n_dos)
    bp_dia_dos = np.random.normal(loc=79, scale=4, size=n_dos)
    temp_dos = np.random.normal(loc=98.6, scale=0.4, size=n_dos)
    
    # DoS causes massive spikes in network volume and latency
    pkt_rate_dos = np.random.normal(loc=180.0, scale=25.0, size=n_dos)
    pkt_size_dos = np.random.normal(loc=550.0, scale=60.0, size=n_dos)
    jitter_dos = np.random.exponential(scale=18.0, size=n_dos)
    latency_dos = np.random.normal(loc=95.0, scale=15.0, size=n_dos)
    
    df_dos = pd.DataFrame({
        'heart_rate': hr_dos,
        'respiration_rate': resp_dos,
        'spo2': spo2_dos,
        'bp_systolic': bp_sys_dos,
        'bp_diastolic': bp_dia_dos,
        'temperature': temp_dos,
        'packet_rate': pkt_rate_dos,
        'packet_size': pkt_size_dos,
        'jitter': jitter_dos,
        'latency': latency_dos,
        'label': 'DoS_Flooding',
        'is_attack': 1
    })

    # -------------------------------------------------------------
    # 3. KNOWN ATTACK: FALSE DATA INJECTION (Physical anomaly)
    # -------------------------------------------------------------
    # Network traffic looks normal, but physical telemetry is falsified
    hr_inj = np.random.choice([np.random.normal(40, 4), np.random.normal(160, 10)], size=n_injection)
    resp_inj = np.random.normal(loc=32.0, scale=4.0, size=n_injection)
    spo2_inj = np.random.uniform(low=78.0, high=88.0, size=n_injection)  # Hypoxia injection
    bp_sys_inj = np.random.normal(loc=185, scale=12, size=n_injection)   # Hypertensive crisis
    bp_dia_inj = np.random.normal(loc=115, scale=8, size=n_injection)
    temp_inj = np.random.normal(loc=103.5, scale=1.0, size=n_injection)  # High fever injection
    
    pkt_rate_inj = np.random.normal(loc=36.0, scale=4.2, size=n_injection) # Normal network
    pkt_size_inj = np.random.normal(loc=325.0, scale=24.0, size=n_injection)
    jitter_inj = np.random.exponential(scale=2.2, size=n_injection)
    latency_inj = np.random.normal(loc=16.0, scale=3.5, size=n_injection)
    
    df_inj = pd.DataFrame({
        'heart_rate': hr_inj,
        'respiration_rate': resp_inj,
        'spo2': spo2_inj,
        'bp_systolic': bp_sys_inj,
        'bp_diastolic': bp_dia_inj,
        'temperature': temp_inj,
        'packet_rate': pkt_rate_inj,
        'packet_size': pkt_size_inj,
        'jitter': jitter_inj,
        'latency': latency_inj,
        'label': 'Data_Injection',
        'is_attack': 1
    })

    # -------------------------------------------------------------
    # 4. ZERO-DAY ATTACK: NOVEL STEALTH REPLAY & DESYNCHRONIZATION
    # -------------------------------------------------------------
    # Subtle manipulation: packets arrive in micro-bursts with desynchronized patient state
    hr_zd = np.random.normal(loc=88.0, scale=14.0, size=n_zeroday)
    resp_zd = np.random.normal(loc=22.0, scale=3.5, size=n_zeroday)
    spo2_zd = np.random.uniform(low=91.0, high=95.0, size=n_zeroday)
    bp_sys_zd = np.random.normal(loc=140, scale=10, size=n_zeroday)
    bp_dia_zd = np.random.normal(loc=95, scale=6, size=n_zeroday)
    temp_zd = np.random.normal(loc=99.8, scale=0.8, size=n_zeroday)
    
    pkt_rate_zd = np.random.normal(loc=72.0, scale=12.0, size=n_zeroday)
    pkt_size_zd = np.random.normal(loc=420.0, scale=40.0, size=n_zeroday)
    jitter_zd = np.random.exponential(scale=8.5, size=n_zeroday)
    latency_zd = np.random.normal(loc=45.0, scale=8.0, size=n_zeroday)
    
    df_zd = pd.DataFrame({
        'heart_rate': hr_zd,
        'respiration_rate': resp_zd,
        'spo2': spo2_zd,
        'bp_systolic': bp_sys_zd,
        'bp_diastolic': bp_dia_zd,
        'temperature': temp_zd,
        'packet_rate': pkt_rate_zd,
        'packet_size': pkt_size_zd,
        'jitter': jitter_zd,
        'latency': latency_zd,
        'label': 'ZeroDay_Replay',
        'is_attack': 1
    })
    
    # Combine & shuffle
    full_df = pd.concat([df_normal, df_dos, df_inj, df_zd], ignore_index=True)
    full_df = full_df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    os.makedirs('data', exist_ok=True)
    output_path = os.path.join('data', 'iomt_telemetry.csv')
    full_df.to_csv(output_path, index=False)
    print(f"Dataset saved to '{output_path}' with shape: {full_df.shape}")
    print("\nClass Distribution:")
    print(full_df['label'].value_counts())
    return full_df

if __name__ == '__main__':
    generate_iomt_dataset()