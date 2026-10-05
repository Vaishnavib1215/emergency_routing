"""
Emergency Vehicle Preemption Rules & Vehicle Profiles.
Based on Federal Highway Administration (FHWA) Traffic Signal Preemption Guidelines.
"""

VEHICLE_PROFILES = {
    'ambulance': {
        'name': 'Emergency Medical Service (Ambulance)',
        'icon': '🚑',
        'speed_factor': 1.15,       # Capable of exceeding speed limit when safe
        'preemption_level': 0.85,   # Green wave reduces intersection delay by 85%
        'congestion_penalty': 1.4,  # Heavily impacted if gridlock prevents lane pulling
        'priority': 1,
        'description': 'Rapid patient transport; requires steady traversal and minimized lateral jostling.'
    },
    'fire_truck': {
        'name': 'Heavy Fire Apparatus / Engine',
        'icon': '🚒',
        'speed_factor': 0.95,       # Slower acceleration due to vehicle mass
        'preemption_level': 0.95,   # Full intersection override (all opposing red)
        'congestion_penalty': 2.0,  # Severe delay if narrow streets or jammed lanes
        'priority': 1,
        'description': 'Massive water tender/ladder truck; requires wide turning radii and clear multi-lane avenues.'
    },
    'police': {
        'name': 'Police Patrol / Tactical Interceptor',
        'icon': '🚓',
        'speed_factor': 1.30,       # High speed capability
        'preemption_level': 0.70,   # Acoustic & strobe preemption
        'congestion_penalty': 1.1,  # Maneuverable through shoulders and tight bottlenecks
        'priority': 2,
        'description': 'High-acceleration pursuit vehicle; can navigate shoulder lanes and tighter detours.'
    }
}

def get_preemption_multiplier(vehicle_type: str, lanes: int, has_signal: bool = True) -> float:
    """
    Computes traversal efficiency multiplier considering vehicle preemption and lane clearance.
    Returns a factor where lower means faster traversal for the emergency vehicle.
    """
    profile = VEHICLE_PROFILES.get(vehicle_type.lower(), VEHICLE_PROFILES['ambulance'])
    
    # Preemption bonus is higher on roads with multiple lanes (easier for traffic to yield)
    lane_clearance_bonus = 0.05 * min(lanes - 1, 3)
    effective_preemption = profile['preemption_level'] + lane_clearance_bonus
    
    # Reduction in intersection/traversal resistance
    multiplier = max(0.20, 1.0 - effective_preemption * 0.4)
    return multiplier
