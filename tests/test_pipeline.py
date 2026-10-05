"""
Integration & Unit Test Suite for Emergency Vehicle Route & Congestion Prediction System.
"""

import os
import sys
import unittest
import numpy as np
import torch
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from routing.graph_builder import build_emergency_road_network
from routing.route_optimizer import RouteOptimizer
from models.cnn_lstm_model import SpatioTemporalCNNLSTM
from models.yolo_detector import YOLOVehicleDetector
from backend.main import app

class TestEmergencyRoutingPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.network = build_emergency_road_network()
        cls.optimizer = RouteOptimizer(cls.network)

    def test_01_road_network_integrity(self):
        """Verify the road network graph has nodes, edges, and valid coordinates."""
        self.assertGreaterEqual(self.network.number_of_nodes(), 10)
        self.assertGreaterEqual(self.network.number_of_edges(), 15)
        
        # Verify key landmark nodes exist
        self.assertIn('STATION_1', self.network.nodes)
        self.assertIn('HOSPITAL_CITY', self.network.nodes)
        self.assertIn('DISPATCH_HQ', self.network.nodes)
        
        node_data = self.network.nodes['HOSPITAL_CITY']
        self.assertIn('lat', node_data)
        self.assertIn('lon', node_data)
        self.assertEqual(node_data['type'], 'hospital')

    def test_02_cnn_lstm_model_forward(self):
        """Verify the Spatio-Temporal CNN-LSTM forward pass and output dimensions."""
        model = SpatioTemporalCNNLSTM(num_segments=8, seq_len=12, forecast_horizon=3)
        dummy_x = torch.rand(4, 12, 8) # Batch=4, Lookback=12, Segments=8
        out = model(dummy_x)
        
        self.assertEqual(out.shape, (4, 3, 8)) # Batch=4, Horizon=3, Segments=8
        # Output should be strictly in [0, 1] (congestion range)
        self.assertTrue(torch.all(out >= 0.0) and torch.all(out <= 1.0))

    def test_03_yolo_vehicle_detector(self):
        """Verify YOLO detector processes video frames and computes density."""
        detector = YOLOVehicleDetector()
        test_frame = np.full((360, 640, 3), 100, dtype=np.uint8)
        ann_frame, stats = detector.process_frame(test_frame)
        
        self.assertEqual(ann_frame.shape, test_frame.shape)
        self.assertIn('total_vehicles', stats)
        self.assertIn('density', stats)
        self.assertTrue(0.0 <= stats['density'] <= 1.0)

    def test_04_route_optimizer(self):
        """Verify route optimizer computes candidate routes and baselines."""
        res = self.optimizer.recommend_route(
            origin='STATION_1',
            destination='HOSPITAL_CITY',
            vehicle_type='ambulance'
        )
        
        self.assertIn('recommended_route', res)
        self.assertIn('comparison', res)
        self.assertIn('candidate_routes', res)
        self.assertGreaterEqual(len(res['candidate_routes']), 1)
        
        rec = res['recommended_route']
        self.assertGreater(rec['total_time_min'], 0)
        self.assertGreater(rec['total_distance_km'], 0)
        self.assertGreater(len(rec['segments']), 0)
        
        comp = res['comparison']
        self.assertIn('static_gps', comp)
        self.assertIn('proposed_predictive', comp)
        self.assertIn('time_saved_pct', comp)

    def test_05_api_endpoints(self):
        """Verify FastAPI REST endpoints."""
        # 1. Status
        r_status = self.client.get("/api/status")
        self.assertEqual(r_status.status_code, 200)
        self.assertEqual(r_status.json()["status"], "online")

        # 2. Network topology
        r_net = self.client.get("/api/network")
        self.assertEqual(r_net.status_code, 200)
        data = r_net.json()
        self.assertIn("nodes", data)
        self.assertIn("edges", data)

        # 3. Route Recommendation
        r_route = self.client.post("/api/route/recommend", json={
            "origin": "STATION_1",
            "destination": "HOSPITAL_CITY",
            "vehicle_type": "ambulance"
        })
        self.assertEqual(r_route.status_code, 200)
        route_data = r_route.json()
        self.assertIn("recommended_route", route_data)

        # 4. Forecast
        r_fc = self.client.get("/api/forecast/seg_1")
        self.assertEqual(r_fc.status_code, 200)
        fc_data = r_fc.json()
        self.assertIn("labels", fc_data)
        self.assertIn("historical", fc_data)
        self.assertIn("predicted", fc_data)

        # 5. Camera Frame & Stats
        r_stats = self.client.get("/api/camera/stats?camera_id=cam_1")
        self.assertEqual(r_stats.status_code, 200)
        self.assertIn("density", r_stats.json())

        r_frame = self.client.get("/api/camera/frame?camera_id=cam_1")
        self.assertEqual(r_frame.status_code, 200)
        self.assertEqual(r_frame.headers["content-type"], "image/jpeg")

if __name__ == "__main__":
    unittest.main()
