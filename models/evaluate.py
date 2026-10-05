"""
Model Evaluation Suite for Spatio-Temporal CNN-LSTM Forecaster.
Calculates MAE, RMSE, MAPE, and R2 Score across individual corridors.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.cnn_lstm_model import SpatioTemporalCNNLSTM
from models.train_lstm import create_sequences, load_traffic_data, WEIGHTS_PATH

def evaluate_model():
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[Error] Weights file not found at {WEIGHTS_PATH}. Please train the model first.")
        return None
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Evaluate] Loading model weights from {WEIGHTS_PATH} onto {device}...")
    
    checkpoint = torch.load(WEIGHTS_PATH, map_location=device)
    seq_len = checkpoint.get('seq_len', 12)
    horizon = checkpoint.get('horizon', 3)
    num_segments = checkpoint.get('num_segments', 8)
    
    model = SpatioTemporalCNNLSTM(num_segments=num_segments, seq_len=seq_len, forecast_horizon=horizon)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load test split
    data = load_traffic_data()
    X, Y = create_sequences(data, seq_len=seq_len, horizon=horizon)
    n_train_val = int(len(X) * 0.85)
    X_test, Y_test = X[n_train_val:], Y[n_train_val:]
    
    with torch.no_grad():
        inputs = torch.from_numpy(X_test).to(device)
        predictions = model(inputs).cpu().numpy()
        
    # Overall metrics across all horizons and corridors
    flat_y_true = Y_test.reshape(-1)
    flat_y_pred = predictions.reshape(-1)
    
    mae = mean_absolute_error(flat_y_true, flat_y_pred)
    rmse = np.sqrt(mean_squared_error(flat_y_true, flat_y_pred))
    r2 = r2_score(flat_y_true, flat_y_pred)
    mape = np.mean(np.abs((flat_y_true - flat_y_pred) / np.clip(flat_y_true, 1e-4, 1.0))) * 100.0
    accuracy_tol = np.mean(np.abs(flat_y_true - flat_y_pred) <= 0.10) * 100.0
    
    print("\n" + "=" * 65)
    print("        CNN-LSTM CONGESTION PREDICTION EVALUATION REPORT        ")
    print("=" * 65)
    print(f"  Test Samples Evaluated       : {len(X_test)}")
    print(f"  Mean Absolute Error (MAE)    : {mae:.4f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"  Mean Abs Percentage Error    : {mape:.2f}%")
    print(f"  R^2 Score (Goodness of Fit)  : {r2:.4f}")
    print(f"  Forecast Accuracy (+/-10%): {accuracy_tol:.2f}%")
    print("=" * 65)
    
    print("\nCorridor-Specific Performance (Horizon t+15m):")
    print("-" * 65)
    print(f"{'Corridor ID':^14} | {'MAE':^12} | {'RMSE':^12} | {'R2 Score':^12}")
    print("-" * 65)
    
    horizon_idx = 2 # t+15m
    for seg_idx in range(num_segments):
        y_t = Y_test[:, horizon_idx, seg_idx]
        y_p = predictions[:, horizon_idx, seg_idx]
        seg_mae = mean_absolute_error(y_t, y_p)
        seg_rmse = np.sqrt(mean_squared_error(y_t, y_p))
        seg_r2 = r2_score(y_t, y_p)
        print(f"{f'seg_{seg_idx+1}':^14} | {seg_mae:^12.4f} | {seg_rmse:^12.4f} | {seg_r2:^12.4f}")
    print("-" * 65 + "\n")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'accuracy_tol': accuracy_tol
    }

if __name__ == "__main__":
    evaluate_model()
