"""
Predictive Route Optimization Engine for Emergency Vehicles.
Evaluates candidate paths using:
1. Static Baseline (Standard GPS shortest path)
2. Reactive Real-Time Baseline (Current congestion only)
3. Proposed Predictive CNN-LSTM Route (Forecast-aware traversal with signal preemption)
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any
from routing.graph_builder import build_emergency_road_network
from routing.emergency_preemption import VEHICLE_PROFILES, get_preemption_multiplier

class RouteOptimizer:
    def __init__(self, network=None):
        self.network = network if network is not None else build_emergency_road_network()
        
    def calculate_segment_travel_time(self,
                                      edge_data: Dict[str, Any],
                                      vehicle_type: str,
                                      current_congestion: float,
                                      predicted_congestion: float,
                                      use_prediction: bool = True,
                                      use_realtime: bool = True) -> Dict[str, float]:
        """
        Calculates expected traversal duration (in minutes) for an emergency vehicle on an edge.
        Cost model:
        Base Time = (Distance / Speed Limit) * 60 (minutes)
        Delay Factor = 1 + alpha * Current_Congestion + beta * Predicted_Congestion
        Emergency Vehicle Factor = Speed Factor * Preemption Multiplier
        """
        length_km = edge_data['length_km']
        speed_limit = edge_data['speed_limit_kmh']
        lanes = edge_data.get('lanes', 2)
        
        base_time_min = (length_km / speed_limit) * 60.0
        
        profile = VEHICLE_PROFILES.get(vehicle_type.lower(), VEHICLE_PROFILES['ambulance'])
        preemption_mult = get_preemption_multiplier(vehicle_type, lanes)
        
        if not use_realtime and not use_prediction:
            # Static GPS: only considers base speed limit
            effective_time = base_time_min / profile['speed_factor']
            congestion_level = 0.0
        elif use_realtime and not use_prediction:
            # Reactive: considers only current congestion
            congestion_level = current_congestion
            penalty = profile['congestion_penalty']
            # Non-linear congestion penalty as road approaches saturation
            delay_factor = 1.0 + (current_congestion ** 1.8) * penalty * 2.2
            effective_time = (base_time_min * delay_factor * preemption_mult) / profile['speed_factor']
        else:
            # Proposed: blend of current and predicted congestion based on travel time horizon
            # As vehicle travels, predicted congestion dominates
            blended_congestion = 0.35 * current_congestion + 0.65 * predicted_congestion
            congestion_level = blended_congestion
            penalty = profile['congestion_penalty']
            delay_factor = 1.0 + (blended_congestion ** 1.8) * penalty * 2.2
            effective_time = (base_time_min * delay_factor * preemption_mult) / profile['speed_factor']
            
        return {
            'travel_time_min': round(effective_time, 2),
            'base_time_min': round(base_time_min, 2),
            'congestion_level': round(congestion_level, 3),
            'length_km': length_km
        }

    def recommend_route(self,
                        origin: str,
                        destination: str,
                        vehicle_type: str = 'ambulance',
                        current_congestion_map: Dict[str, float] = None,
                        predicted_congestion_map: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Finds optimal route and compares against static and reactive baselines.
        """
        if origin not in self.network or destination not in self.network:
            raise ValueError(f"Origin '{origin}' or Destination '{destination}' not in network.")
            
        current_map = current_congestion_map or {f"seg_{i}": 0.30 for i in range(1, 9)}
        predicted_map = predicted_congestion_map or {f"seg_{i}": 0.35 for i in range(1, 9)}
        
        # 1. Static Shortest Path
        def weight_static(u, v, d):
            res = self.calculate_segment_travel_time(d, vehicle_type, 0, 0, False, False)
            return res['travel_time_min']
            
        static_path = nx.dijkstra_path(self.network, origin, destination, weight=weight_static)
        static_metrics = self._evaluate_path(static_path, vehicle_type, current_map, predicted_map, mode='static')
        
        # 2. Reactive Path (Current Congestion Only)
        def weight_reactive(u, v, d):
            seg_id = d.get('corridor_id', 'seg_1')
            c_curr = current_map.get(seg_id, 0.3)
            res = self.calculate_segment_travel_time(d, vehicle_type, c_curr, 0, False, True)
            return res['travel_time_min']
            
        reactive_path = nx.dijkstra_path(self.network, origin, destination, weight=weight_reactive)
        reactive_metrics = self._evaluate_path(reactive_path, vehicle_type, current_map, predicted_map, mode='reactive')
        
        # 3. Proposed Predictive Route (CNN-LSTM forecast aware)
        def weight_predictive(u, v, d):
            seg_id = d.get('corridor_id', 'seg_1')
            c_curr = current_map.get(seg_id, 0.3)
            c_pred = predicted_map.get(seg_id, 0.3)
            res = self.calculate_segment_travel_time(d, vehicle_type, c_curr, c_pred, True, True)
            return res['travel_time_min']
            
        optimal_path = nx.dijkstra_path(self.network, origin, destination, weight=weight_predictive)
        optimal_metrics = self._evaluate_path(optimal_path, vehicle_type, current_map, predicted_map, mode='predictive')
        
        # Candidate Paths: Find top alternative paths
        candidate_paths = []
        try:
            # Use k-shortest paths
            k_gen = nx.shortest_simple_paths(self.network, origin, destination, weight=weight_predictive)
            seen_nodes = set()
            count = 0
            for p in k_gen:
                p_tuple = tuple(p)
                if p_tuple in seen_nodes:
                    continue
                seen_nodes.add(p_tuple)
                
                m = self._evaluate_path(p, vehicle_type, current_map, predicted_map, mode='predictive')
                candidate_paths.append({
                    'rank': count + 1,
                    'path': p,
                    'path_names': [self.network.nodes[n]['name'] for n in p],
                    'total_time_min': m['total_time_min'],
                    'total_distance_km': m['total_distance_km'],
                    'avg_congestion': m['avg_congestion'],
                    'max_congestion': m['max_congestion'],
                    'bottleneck': m['bottleneck'],
                    'segments': m['segments'],
                    'is_recommended': (p == optimal_path)
                })
                count += 1
                if count >= 3:
                    break
        except Exception:
            candidate_paths = [{
                'rank': 1,
                'path': optimal_path,
                'path_names': [self.network.nodes[n]['name'] for n in optimal_path],
                'total_time_min': optimal_metrics['total_time_min'],
                'total_distance_km': optimal_metrics['total_distance_km'],
                'avg_congestion': optimal_metrics['avg_congestion'],
                'max_congestion': optimal_metrics['max_congestion'],
                'bottleneck': optimal_metrics['bottleneck'],
                'segments': optimal_metrics['segments'],
                'is_recommended': True
            }]

        # Calculate Savings & Comparison
        static_time = static_metrics['total_time_min']
        optimal_time = optimal_metrics['total_time_min']
        time_saved_min = round(max(0.0, static_time - optimal_time), 2)
        pct_saved = round((time_saved_min / static_time * 100.0) if static_time > 0 else 0.0, 1)

        return {
            'origin': origin,
            'origin_name': self.network.nodes[origin]['name'],
            'destination': destination,
            'destination_name': self.network.nodes[destination]['name'],
            'vehicle_type': vehicle_type,
            'vehicle_profile': VEHICLE_PROFILES.get(vehicle_type, VEHICLE_PROFILES['ambulance']),
            'recommended_route': candidate_paths[0] if candidate_paths else None,
            'candidate_routes': candidate_paths,
            'comparison': {
                'static_gps': {
                    'name': 'Traditional Static GPS',
                    'path': static_path,
                    'total_time_min': static_metrics['total_time_min'],
                    'total_distance_km': static_metrics['total_distance_km'],
                    'bottleneck': static_metrics['bottleneck']
                },
                'reactive_realtime': {
                    'name': 'Reactive Current-Traffic Only',
                    'path': reactive_path,
                    'total_time_min': reactive_metrics['total_time_min'],
                    'total_distance_km': reactive_metrics['total_distance_km']
                },
                'proposed_predictive': {
                    'name': 'Proposed YOLO & CNN-LSTM Predictive Route',
                    'path': optimal_path,
                    'total_time_min': optimal_metrics['total_time_min'],
                    'total_distance_km': optimal_metrics['total_distance_km']
                },
                'time_saved_min': time_saved_min,
                'time_saved_pct': pct_saved
            }
        }

    def _evaluate_path(self, path: List[str], vehicle_type: str, current_map, predicted_map, mode='predictive'):
        """Computes step-by-step segment metrics along a path."""
        total_time = 0.0
        total_dist = 0.0
        congestions = []
        segments = []
        bottleneck = "None"
        max_cong = -1.0
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            edge_data = self.network[u][v]
            seg_id = edge_data.get('corridor_id', 'seg_1')
            c_curr = current_map.get(seg_id, 0.3)
            c_pred = predicted_map.get(seg_id, 0.3)
            
            use_p = (mode == 'predictive')
            use_r = (mode in ['predictive', 'reactive'])
            
            res = self.calculate_segment_travel_time(edge_data, vehicle_type, c_curr, c_pred, use_p, use_r)
            total_time += res['travel_time_min']
            total_dist += res['length_km']
            congestions.append(res['congestion_level'])
            
            if res['congestion_level'] > max_cong:
                max_cong = res['congestion_level']
                bottleneck = f"{edge_data['name']} ({seg_id})"
                
            segments.append({
                'from_node': u,
                'from_name': self.network.nodes[u]['name'],
                'from_coords': [self.network.nodes[u]['lat'], self.network.nodes[u]['lon']],
                'to_node': v,
                'to_name': self.network.nodes[v]['name'],
                'to_coords': [self.network.nodes[v]['lat'], self.network.nodes[v]['lon']],
                'edge_name': edge_data['name'],
                'corridor_id': seg_id,
                'length_km': res['length_km'],
                'travel_time_min': res['travel_time_min'],
                'congestion_level': res['congestion_level'],
                'has_cctv': edge_data.get('has_cctv', False)
            })
            
        return {
            'total_time_min': round(total_time, 2),
            'total_distance_km': round(total_dist, 2),
            'avg_congestion': round(float(np.mean(congestions)), 3) if congestions else 0.0,
            'max_congestion': round(float(max_cong), 3) if max_cong >= 0 else 0.0,
            'bottleneck': bottleneck,
            'segments': segments
        }
