"""
Training Pipeline for Spatio-Temporal CNN-LSTM Congestion Forecaster.
Trains on multi-segment traffic dataset and saves model weights to models/saved_weights/.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from models.cnn_lstm_model import SpatioTemporalCNNLSTM
from data.download_datasets import run_download_pipeline
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "multi_segment_traffic.csv")
WEIGHTS_DIR = os.path.join(BASE_DIR, "models", "saved_weights")
WEIGHTS_PATH = os.path.join(WEIGHTS_DIR, "cnn_lstm_traffic.pt")

def create_sequences(data: np.ndarray, seq_len: int = 12, horizon: int = 3):
    """
    Transforms 2D array of shape (N_timesteps, N_segments) into sliding window sequences.
    X: (Samples, seq_len, N_segments)
    Y: (Samples, horizon, N_segments)
    """
    xs, ys = [], []
    total_len = len(data)
    for i in range(total_len - seq_len - horizon + 1):
        x = data[i : i + seq_len]
        y = data[i + seq_len : i + seq_len + horizon]
        xs.append(x)
        ys.append(y)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)

def load_traffic_data():
    if not os.path.exists(DATA_PATH):
        print(f"[Train] Processed dataset not found at {DATA_PATH}. Triggering data pipeline...")
        run_download_pipeline()
        
    df = pd.read_csv(DATA_PATH)
    corridor_cols = [f'seg_{i}' for i in range(1, 9)]
    traffic_matrix = df[corridor_cols].values # (N, 8)
    return traffic_matrix

def train_model(epochs: int = 15, batch_size: int = 64, lr: float = 0.001, quick: bool = False):
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Train] Training CNN-LSTM model on device: {device}")
    
    traffic_data = load_traffic_data()
    if quick:
        # Use a subset of 3,000 samples for ultra-fast training/verification
        traffic_data = traffic_data[:3000]
        epochs = min(epochs, 5)
        
    seq_len = 12
    horizon = 3
    num_segments = 8
    
    X, Y = create_sequences(traffic_data, seq_len=seq_len, horizon=horizon)
    print(f"[Train] Dataset prepared: {len(X)} sequences (X: {X.shape}, Y: {Y.shape})")
    
    # Train / Val / Test split (70 / 15 / 15)
    n_train = int(len(X) * 0.70)
    n_val = int(len(X) * 0.15)
    
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:n_train+n_val], Y[n_train:n_train+n_val]
    X_test, Y_test = X[n_train+n_val:], Y[n_train+n_val:]
    
    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(Y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(Y_val))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = SpatioTemporalCNNLSTM(
        num_segments=num_segments,
        seq_len=seq_len,
        forecast_horizon=horizon
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    best_val_loss = float('inf')
    
    print("\n" + "=" * 60)
    print(f"{'Epoch':^8} | {'Train MSE Loss':^18} | {'Val MSE Loss':^16} | {'Status':^12}")
    print("=" * 60)
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)
                
        val_loss /= len(val_dataset)
        scheduler.step(val_loss)
        
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'seq_len': seq_len,
                'horizon': horizon,
                'num_segments': num_segments,
                'val_loss': best_val_loss
            }, WEIGHTS_PATH)
            status = "* Saved Best"
            
        print(f"{epoch:^8} | {train_loss:^18.6f} | {val_loss:^16.6f} | {status:^12}")
        
    print("=" * 60)
    print(f"[Train Complete] Optimal weights saved to {WEIGHTS_PATH} (Best Val Loss: {best_val_loss:.6f})\n")
    return WEIGHTS_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-LSTM Traffic Predictor")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=64, help="Batch size")
    parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
    parser.add_argument('--quick', action='store_true', help="Fast training sanity run")
    args = parser.parse_args()
    
    train_model(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, quick=args.quick)
