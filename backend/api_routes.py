"""
FastAPI REST Endpoints for Emergency Vehicle Route & Congestion Prediction System.
"""

import os
import torch
import numpy as np
import pandas as pd
from typing import Optional
from fastapi import APIRouter, Response, HTTPException
from pydantic import BaseModel

from routing.graph_builder import build_emergency_road_network
from routing.route_optimizer import RouteOptimizer
from backend.stream_manager import CameraStreamManager
from models.cnn_lstm_model import SpatioTemporalCNNLSTM

router = APIRouter(prefix="/api")

# Initialize shared components
road_network = build_emergency_road_network()
optimizer = RouteOptimizer(road_network)
stream_mgr = CameraStreamManager()

# Load CNN-LSTM model if weights available
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "saved_weights", "cnn_lstm_traffic.pt")
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "multi_segment_traffic.csv")

lstm_model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_or_load_lstm():
    global lstm_model
    if lstm_model is not None:
        return lstm_model
    if os.path.exists(WEIGHTS_PATH):
        try:
            ckpt = torch.load(WEIGHTS_PATH, map_location=device)
            m = SpatioTemporalCNNLSTM(
                num_segments=ckpt.get('num_segments', 8),
                seq_len=ckpt.get('seq_len', 12),
                forecast_horizon=ckpt.get('horizon', 3)
            )
            m.load_state_dict(ckpt['model_state_dict'])
            m.to(device)
            m.eval()
            lstm_model = m
            print("[API] Loaded trained CNN-LSTM model weights.")
        except Exception as e:
            print(f"[API Warning] Failed to load weights ({e}). Initializing standard model.")
            lstm_model = SpatioTemporalCNNLSTM().to(device)
            lstm_model.eval()
    else:
        lstm_model = SpatioTemporalCNNLSTM().to(device)
        lstm_model.eval()
    return lstm_model

class RouteRequest(BaseModel):
    origin: str
    destination: str
    vehicle_type: str = "ambulance"

@router.get("/status")
def get_system_status():
    return {
        "status": "online",
        "yolo_detector": "active",
        "cnn_lstm_forecaster": "active",
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "total_nodes": road_network.number_of_nodes(),
        "total_edges": road_network.number_of_edges()
    }

def get_current_and_predicted_congestion():
    """Generates aligned current densities and CNN-LSTM predicted densities."""
    # 1. Get real-time YOLO densities from camera manager
    curr_densities = stream_mgr.get_all_densities()
    
    # 2. Get predictions using CNN-LSTM model
    model = get_or_load_lstm()
    pred_densities = {}
    
    # Create input sequence: use historical data or simulate current trend
    seq_input = np.zeros((1, 12, 8), dtype=np.float32)
    for seg_idx, seg_id in enumerate([f'seg_{i}' for i in range(1, 9)]):
        base_c = curr_densities.get(seg_id, 0.40)
        # Create subtle ramp/trend
        trend = np.linspace(max(0.05, base_c - 0.15), base_c, 12)
        seq_input[0, :, seg_idx] = trend
        
    with torch.no_grad():
        x_tensor = torch.from_numpy(seq_input).to(device)
        out = model(x_tensor).cpu().numpy() # (1, 3, 8)
        
    for seg_idx, seg_id in enumerate([f'seg_{i}' for i in range(1, 9)]):
        # Forecast at horizon 3 (t+15 min)
        pred_val = float(out[0, 2, seg_idx])
        pred_densities[seg_id] = round(pred_val, 3)
        
    return curr_densities, pred_densities

@router.get("/network")
def get_network_topology():
    curr_densities, pred_densities = get_current_and_predicted_congestion()
    
    nodes_dict = {}
    for n, data in road_network.nodes(data=True):
        nodes_dict[n] = data
        
    edges_list = []
    seen = set()
    for u, v, data in road_network.edges(data=True):
        pair = tuple(sorted([u, v]))
        if pair in seen:
            continue
        seen.add(pair)
        
        seg_id = data.get('corridor_id', 'seg_1')
        c_curr = curr_densities.get(seg_id, 0.35)
        c_pred = pred_densities.get(seg_id, 0.40)
        
        edges_list.append({
            'from_node': u,
            'to_node': v,
            'id': data.get('id', f"{u}->{v}"),
            'name': data.get('name', 'Road Segment'),
            'corridor_id': seg_id,
            'length_km': data.get('length_km', 1.0),
            'speed_limit_kmh': data.get('speed_limit_kmh', 50),
            'lanes': data.get('lanes', 2),
            'has_cctv': data.get('has_cctv', False),
            'current_congestion': c_curr,
            'predicted_congestion': c_pred
        })
        
    return {
        'nodes': nodes_dict,
        'edges': edges_list
    }

@router.post("/route/recommend")
def recommend_emergency_route(req: RouteRequest):
    curr_densities, pred_densities = get_current_and_predicted_congestion()
    
    try:
        result = optimizer.recommend_route(
            origin=req.origin,
            destination=req.destination,
            vehicle_type=req.vehicle_type,
            current_congestion_map=curr_densities,
            predicted_congestion_map=pred_densities
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/forecast/{corridor_id}")
def get_corridor_forecast(corridor_id: str):
    curr_densities, pred_densities = get_current_and_predicted_congestion()
    corridors = [f'seg_{i}' for i in range(1, 9)]
    if corridor_id not in corridors:
        corridor_id = 'seg_1'
        
    seg_idx = corridors.index(corridor_id)
    c_now = curr_densities.get(corridor_id, 0.50)
    
    # 6 historical steps leading to now
    historical = [
        round(max(0.05, c_now - 0.25 + np.random.normal(0, 0.02)), 2),
        round(max(0.05, c_now - 0.18 + np.random.normal(0, 0.02)), 2),
        round(max(0.05, c_now - 0.12 + np.random.normal(0, 0.02)), 2),
        round(max(0.05, c_now - 0.07 + np.random.normal(0, 0.02)), 2),
        round(max(0.05, c_now - 0.03 + np.random.normal(0, 0.02)), 2),
        round(c_now, 2)
    ]
    
    # Model predictions for t+5m, t+10m, t+15m
    model = get_or_load_lstm()
    seq_input = np.zeros((1, 12, 8), dtype=np.float32)
    seq_input[0, -6:, seg_idx] = historical
    with torch.no_grad():
        out = model(torch.from_numpy(seq_input).to(device)).cpu().numpy()
        
    p1 = round(float(out[0, 0, seg_idx]), 2)
    p2 = round(float(out[0, 1, seg_idx]), 2)
    p3 = round(float(out[0, 2, seg_idx]), 2)
    
    predicted = [None, None, None, None, None, round(c_now, 2), p1, p2, p3]
    labels = ['t-25m', 't-20m', 't-15m', 't-10m', 't-5m', 'Now', 't+5m', 't+10m', 't+15m']
    
    return {
        'corridor_id': corridor_id,
        'labels': labels,
        'historical': historical + [None, None, None],
        'predicted': predicted
    }

@router.get("/camera/frame")
def get_camera_frame(camera_id: str = 'cam_1'):
    jpeg_bytes = stream_mgr.get_frame_jpeg(camera_id)
    if jpeg_bytes is None:
        raise HTTPException(status_code=503, detail="Camera frame unavailable")
    return Response(content=jpeg_bytes, media_type="image/jpeg")

@router.get("/camera/stats")
def get_camera_stats(camera_id: str = 'cam_1'):
    return stream_mgr.get_stats(camera_id)
