"""
Stream & Camera Feed Manager.
Streams video frames from traffic camera recordings / synthesized CCTV cameras,
runs YOLOv8 object detection on frames, and caches real-time vehicle counts & density.
"""

import os
import cv2
import time
import threading
import numpy as np
from typing import Dict, Any, Optional
from models.yolo_detector import YOLOVehicleDetector

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, "data", "raw", "traffic_cam1.mp4")

class CameraStreamManager:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CameraStreamManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        
        self.detector = YOLOVehicleDetector()
        self.lock = threading.Lock()
        
        # Camera Registry (8 cameras corresponding to the 8 corridors)
        self.cameras = {
            'cam_1': {'name': 'CAM-01: I-94 Interchange', 'corridor': 'seg_1', 'frame': None, 'stats': {}},
            'cam_2': {'name': 'CAM-02: North Bypass', 'corridor': 'seg_2', 'frame': None, 'stats': {}},
            'cam_3': {'name': 'CAM-03: Downtown Plaza', 'corridor': 'seg_3', 'frame': None, 'stats': {}},
            'cam_4': {'name': 'CAM-04: University Blvd', 'corridor': 'seg_4', 'frame': None, 'stats': {}},
            'cam_5': {'name': 'CAM-05: River Bridge West', 'corridor': 'seg_5', 'frame': None, 'stats': {}},
            'cam_6': {'name': 'CAM-06: Hospital Expressway', 'corridor': 'seg_6', 'frame': None, 'stats': {}},
            'cam_7': {'name': 'CAM-07: Industrial Ring', 'corridor': 'seg_7', 'frame': None, 'stats': {}},
            'cam_8': {'name': 'CAM-08: Midtown Link', 'corridor': 'seg_8', 'frame': None, 'stats': {}}
        }
        
        self.cap = None
        self.running = False
        self._init_video_capture()
        self._start_background_processor()

    def _init_video_capture(self):
        if os.path.exists(VIDEO_PATH) and os.path.getsize(VIDEO_PATH) > 10000:
            self.cap = cv2.VideoCapture(VIDEO_PATH)
        else:
            self.cap = None

    def _start_background_processor(self):
        self.running = True
        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _generate_traffic_frame(self, camera_id: str, t_step: int) -> np.ndarray:
        """Generates realistic CCTV frame if video file is not open."""
        w, h = 640, 360
        frame = np.full((h, w, 3), 35, dtype=np.uint8)
        
        # Road pavement & lane lines
        cv2.rectangle(frame, (0, 70), (w, 290), (55, 55, 55), -1)
        cv2.line(frame, (0, 70), (w, 70), (220, 220, 220), 2)
        cv2.line(frame, (0, 290), (w, 290), (220, 220, 220), 2)
        
        for x in range(0, w, 50):
            cv2.line(frame, (x, 140), (x + 25, 140), (255, 255, 255), 2)
            cv2.line(frame, (x, 215), (x + 25, 215), (255, 255, 255), 2)
            
        # Draw moving vehicle boxes
        cam_info = self.cameras.get(camera_id, {})
        corridor = cam_info.get('corridor', 'seg_1')
        num_v = 8 + (hash(corridor) % 8)
        
        for i in range(num_v):
            speed = 5 + (i % 4) * 2
            x = int((t_step * speed + i * 80) % (w + 100)) - 50
            y = 100 + (i % 3) * 65
            color = (50, 180, 240) if i % 3 == 0 else ((200, 100, 50) if i % 2 == 0 else (70, 220, 100))
            cv2.rectangle(frame, (x, y), (x + 55, y + 32), color, -1)
            cv2.rectangle(frame, (x, y), (x + 55, y + 32), (255, 255, 255), 1)
            
        # Overlay camera label
        cv2.putText(frame, f"{cam_info.get('name', camera_id)} | CCTV-LIVE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)
        return frame

    def _process_loop(self):
        t_step = 0
        while self.running:
            try:
                for cam_id in list(self.cameras.keys()):
                    raw_frame = None
                    if cam_id == 'cam_1' and self.cap is not None and self.cap.isOpened():
                        ret, f = self.cap.read()
                        if not ret:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, f = self.cap.read()
                        if ret:
                            raw_frame = cv2.resize(f, (640, 360))
                            
                    if raw_frame is None:
                        raw_frame = self._generate_traffic_frame(cam_id, t_step)
                        
                    # Process frame through YOLO detector
                    annotated, stats = self.detector.process_frame(raw_frame)
                    
                    with self.lock:
                        self.cameras[cam_id]['frame'] = annotated
                        self.cameras[cam_id]['stats'] = stats
                        
                t_step += 1
                time.sleep(1) # Update at ~2 FPS for smooth inference without CPU thrashing
            except Exception as e:
                time.sleep(1.0)

    def get_frame_jpeg(self, camera_id: str = 'cam_1') -> Optional[bytes]:
        with self.lock:
            cam_info = self.cameras.get(camera_id, self.cameras.get('cam_1'))
            frame = cam_info.get('frame')
            if frame is None:
                # Fallback blank frame
                frame = np.full((360, 640, 3), 20, dtype=np.uint8)
                cv2.putText(frame, "CCTV OFFLINE / INITIALIZING", (120, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if ret:
                return jpeg.tobytes()
        return None

    def get_stats(self, camera_id: str = 'cam_1') -> Dict[str, Any]:
        with self.lock:
            cam_info = self.cameras.get(camera_id, self.cameras.get('cam_1'))
            return cam_info.get('stats', {
                'total_vehicles': 12, 'cars': 9, 'trucks': 2, 'buses': 1, 'motorcycles': 0, 'density': 0.55
            })

    def get_all_densities(self) -> Dict[str, float]:
        """Returns instantaneous density map for all 8 corridors."""
        with self.lock:
            densities = {}
            for cam_id, info in self.cameras.items():
                corridor = info.get('corridor', 'seg_1')
                stats = info.get('stats', {})
                densities[corridor] = stats.get('density', 0.40)
            return densities
