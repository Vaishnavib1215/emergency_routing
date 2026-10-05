"""
Master Demo Launcher for Emergency Vehicle Route & Traffic Congestion Prediction System.
Runs data pipeline, initializes models, verifies weights, and launches the cloud dashboard.
"""

import os
import sys
import time
import webbrowser
import uvicorn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from data.download_datasets import run_download_pipeline
from models.train_lstm import WEIGHTS_PATH, train_model
from models.evaluate import evaluate_model

def banner():
    print("=" * 75)
    print("  EMERGENCY VEHICLE ROUTE & TRAFFIC CONGESTION PREDICTION SYSTEM")
    print("  A YOLO and LSTM-Based Cloud System for Predictive Emergency Routing")
    print("=" * 75)

def main():
    banner()
    
    # 1. Dataset Verification & Acquisition
    print("\n[Step 1/4] Verifying and Downloading Datasets...")
    dataset_info = run_download_pipeline()
    
    # 2. Model Training & Weights Verification
    print("\n[Step 2/4] Verifying Spatio-Temporal CNN-LSTM Model Weights...")
    if not os.path.exists(WEIGHTS_PATH):
        print(f"[Model] Weights not found at {WEIGHTS_PATH}. Training CNN-LSTM model...")
        train_model(epochs=6, batch_size=64, lr=0.001, quick=True)
    else:
        print(f"[Model] Found existing trained weights at {WEIGHTS_PATH}")
        
    # 3. Model Performance Evaluation
    print("\n[Step 3/4] Evaluating Model Prediction Metrics...")
    evaluate_model()
    
    # 4. Launch Cloud Dashboard
    print("\n[Step 4/4] Starting FastAPI Cloud Service & Real-Time Dashboard...")
    print("=" * 75)
    print("  >> Emergency Control Dashboard : http://127.0.0.1:8000")
    print("  >> REST API Documentation      : http://127.0.0.1:8000/docs")
    print("  >> Live CCTV YOLO Stream       : http://127.0.0.1:8000/api/camera/frame")
    print("=" * 75)
    print("Press Ctrl+C to terminate the server.\n")
    
    # Open browser automatically after a short delay
    def open_browser():
        time.sleep(1.5)
        try:
            webbrowser.open("http://127.0.0.1:8000")
        except Exception:
            pass
            
    import threading
    threading.Thread(target=open_browser, daemon=True).start()
    
    # Run FastAPI server
    from backend.main import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")

if __name__ == "__main__":
    main()
