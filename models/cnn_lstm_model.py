"""
Spatio-Temporal CNN-LSTM Architecture for Traffic Congestion Forecasting.
Extracts spatial road-corridor correlation patterns via Convolutional layers
and models time-series temporal dynamics via Long Short-Term Memory (LSTM).
"""

import torch
import torch.nn as nn

class SpatioTemporalCNNLSTM(nn.Module):
    def __init__(self,
                 num_segments: int = 8,
                 num_features_per_segment: int = 1,
                 seq_len: int = 12,
                 forecast_horizon: int = 3,
                 cnn_out_channels: int = 32,
                 lstm_hidden_dim: int = 64,
                 lstm_layers: int = 2,
                 dropout: float = 0.2):
        """
        Args:
            num_segments: Number of road segments/corridors (e.g., 8).
            num_features_per_segment: Features per segment (e.g., congestion index).
            seq_len: Lookback window length (e.g., 12 timesteps).
            forecast_horizon: Number of future timesteps to predict (e.g., 3: t+5m, t+10m, t+15m).
            cnn_out_channels: Output feature channels from spatial CNN.
            lstm_hidden_dim: Hidden dimension for LSTM.
            lstm_layers: Number of stacked LSTM layers.
        """
        super(SpatioTemporalCNNLSTM, self).__init__()
        
        self.num_segments = num_segments
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        
        # 1. Spatial Feature Extraction (1D CNN over corridors at each timestep)
        # Input shape: (Batch * Seq_Len, Channels, Num_Segments)
        self.spatial_cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_features_per_segment,
                      out_channels=cnn_out_channels,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(cnn_out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels=cnn_out_channels,
                      out_channels=cnn_out_channels * 2,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm1d(cnn_out_channels * 2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(num_segments)
        )
        
        spatial_feature_dim = (cnn_out_channels * 2) * num_segments
        
        # 2. Temporal Modeling (Stacked LSTM)
        self.lstm = nn.LSTM(
            input_size=spatial_feature_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )
        
        # 3. Multi-Horizon Forecasting Head
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, forecast_horizon * num_segments),
            nn.Sigmoid() # Congestion index is bounded in [0, 1]
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Seq_Len, Num_Segments, Features) or (Batch, Seq_Len, Num_Segments)
        Returns:
            Tensor of shape (Batch, Forecast_Horizon, Num_Segments)
        """
        if x.dim() == 3:
            # (Batch, Seq_Len, Num_Segments) -> (Batch, Seq_Len, Num_Segments, 1)
            x = x.unsqueeze(-1)
            
        batch_size, seq_len, num_segments, num_feats = x.shape
        
        # Reshape for Spatial CNN: (Batch * Seq_Len, Features, Num_Segments)
        x_reshaped = x.permute(0, 1, 3, 2).contiguous()
        x_reshaped = x_reshaped.view(batch_size * seq_len, num_feats, num_segments)
        
        # Spatial feature extraction
        spatial_feats = self.spatial_cnn(x_reshaped) # (B*T, C_out, N)
        spatial_feats = spatial_feats.view(batch_size, seq_len, -1) # (B, T, C_out * N)
        
        # Temporal LSTM processing
        lstm_out, (hn, cn) = self.lstm(spatial_feats) # (B, T, H)
        last_hidden = lstm_out[:, -1, :] # Take last timestep representation (B, H)
        
        # Multi-horizon forecast
        out = self.fc(last_hidden) # (B, Horizon * Num_Segments)
        out = out.view(batch_size, self.forecast_horizon, self.num_segments)
        
        return out

if __name__ == "__main__":
    model = SpatioTemporalCNNLSTM()
    dummy_input = torch.randn(4, 12, 8)
    output = model(dummy_input)
    print("Model initialized successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape} (Batch, Horizon=3, Segments=8)")
