"""
Road Network Graph Builder for Emergency Vehicle Routing.
Models a realistic urban emergency grid with:
- Emergency Dispatch Hubs, Fire Stations, Level-1 Trauma Hospitals, and Intersections.
- Corridors linked to sensor/camera surveillance segments (seg_1 to seg_8).
- Multi-lane highways, arterials, river bridges, and downtown local links.
"""

import networkx as nx

def build_emergency_road_network():
    """
    Constructs a NetworkX MultiDiGraph representing the city emergency road network.
    Coordinates are based around the Twin Cities I-94 corridor (matching UCI dataset).
    """
    G = nx.DiGraph()

    # Define Node Locations (lat, lon) and Metadata
    nodes = {
        'DISPATCH_HQ': {
            'name': 'Central Emergency Dispatch HQ',
            'type': 'station',
            'lat': 44.9780, 'lon': -93.2680,
            'icon': 'dispatch'
        },
        'STATION_1': {
            'name': 'Station Alpha (North Fire & EMS)',
            'type': 'station',
            'lat': 44.9920, 'lon': -93.2750,
            'icon': 'fire_station'
        },
        'STATION_2': {
            'name': 'Station Bravo (East EMS Depot)',
            'type': 'station',
            'lat': 44.9710, 'lon': -93.2350,
            'icon': 'ambulance_depot'
        },
        'HOSPITAL_CITY': {
            'name': 'City General Level-1 Trauma Center',
            'type': 'hospital',
            'lat': 44.9650, 'lon': -93.2550,
            'icon': 'hospital'
        },
        'HOSPITAL_MERCY': {
            'name': 'Mercy Regional Medical Center',
            'type': 'hospital',
            'lat': 44.9850, 'lon': -93.2150,
            'icon': 'hospital'
        },
        'JUNCTION_1': {
            'name': 'Junction 1: I-94 & 1st Ave Hub',
            'type': 'intersection',
            'lat': 44.9820, 'lon': -93.2700,
            'icon': 'junction'
        },
        'JUNCTION_2': {
            'name': 'Junction 2: North Bypass Interchange',
            'type': 'intersection',
            'lat': 44.9950, 'lon': -93.2500,
            'icon': 'junction'
        },
        'JUNCTION_3': {
            'name': 'Junction 3: Central Downtown Plaza',
            'type': 'intersection',
            'lat': 44.9770, 'lon': -93.2600,
            'icon': 'junction'
        },
        'JUNCTION_4': {
            'name': 'Junction 4: University Ave Junction',
            'type': 'intersection',
            'lat': 44.9810, 'lon': -93.2420,
            'icon': 'junction'
        },
        'JUNCTION_5': {
            'name': 'Junction 5: River Bridge West Ramp',
            'type': 'intersection',
            'lat': 44.9730, 'lon': -93.2500,
            'icon': 'junction'
        },
        'JUNCTION_6': {
            'name': 'Junction 6: River Bridge East Ramp',
            'type': 'intersection',
            'lat': 44.9730, 'lon': -93.2380,
            'icon': 'junction'
        },
        'JUNCTION_7': {
            'name': 'Junction 7: Industrial Ring Interchange',
            'type': 'intersection',
            'lat': 44.9600, 'lon': -93.2700,
            'icon': 'junction'
        },
        'JUNCTION_8': {
            'name': 'Junction 8: South Midtown Link',
            'type': 'intersection',
            'lat': 44.9620, 'lon': -93.2450,
            'icon': 'junction'
        },
        'INCIDENT_SITE': {
            'name': 'Active Emergency Incident Site',
            'type': 'incident',
            'lat': 44.9750, 'lon': -93.2520,
            'icon': 'incident'
        }
    }

    for node_id, data in nodes.items():
        G.add_node(node_id, **data)

    # Road Segments (Bidirectional edges)
    # Each edge references one of the 8 monitored corridors
    edges = [
        # (u, v, segment_id, name, length_km, speed_limit_kmh, lanes, has_cctv)
        ('DISPATCH_HQ', 'JUNCTION_1', 'seg_1', 'Dispatch Ramp to I-94', 0.8, 60, 2, False),
        ('DISPATCH_HQ', 'JUNCTION_3', 'seg_3', 'City Hall Transit Link', 0.7, 45, 2, False),
        
        ('STATION_1', 'JUNCTION_1', 'seg_1', 'North Station Feeder', 1.2, 50, 2, False),
        ('STATION_1', 'JUNCTION_2', 'seg_2', 'North Station to Express', 1.5, 60, 2, False),
        
        ('JUNCTION_1', 'JUNCTION_2', 'seg_2', 'North Bypass Expressway', 2.1, 85, 3, True),
        ('JUNCTION_1', 'JUNCTION_3', 'seg_1', 'Interstate 94 Downtown Stretch', 1.4, 90, 4, True),
        ('JUNCTION_1', 'JUNCTION_7', 'seg_7', 'West Beltway to Ring Road', 2.5, 80, 3, False),
        
        ('JUNCTION_2', 'JUNCTION_4', 'seg_2', 'North Bypass East Section', 2.3, 85, 3, False),
        ('JUNCTION_2', 'HOSPITAL_MERCY', 'seg_6', 'Mercy North Accessway', 2.8, 70, 2, False),
        
        ('JUNCTION_3', 'JUNCTION_4', 'seg_4', 'University Boulevard Transit', 1.6, 50, 3, True),
        ('JUNCTION_3', 'JUNCTION_5', 'seg_3', 'Downtown Main Street Arterial', 1.1, 40, 2, True),
        ('JUNCTION_3', 'INCIDENT_SITE', 'seg_3', 'Downtown Civic Route', 0.9, 40, 2, False),
        
        ('JUNCTION_4', 'JUNCTION_6', 'seg_4', 'East Bank River Connect', 1.3, 50, 2, False),
        ('JUNCTION_4', 'HOSPITAL_MERCY', 'seg_6', 'Mercy Medical Boulevard', 2.2, 65, 3, True),
        
        ('JUNCTION_5', 'JUNCTION_6', 'seg_5', 'Grand River Crossing Bridge', 1.0, 55, 2, True),
        ('JUNCTION_5', 'HOSPITAL_CITY', 'seg_6', 'City Hospital North Corridor', 1.2, 55, 3, True),
        ('JUNCTION_5', 'JUNCTION_8', 'seg_8', 'Midtown Parkway North', 1.4, 50, 2, False),
        
        ('JUNCTION_6', 'STATION_2', 'seg_8', 'East Station Connect', 1.1, 50, 2, False),
        ('JUNCTION_6', 'JUNCTION_8', 'seg_8', 'Midtown Parkway East', 1.5, 50, 2, False),
        ('JUNCTION_6', 'HOSPITAL_MERCY', 'seg_6', 'Mercy South Link', 1.9, 60, 2, False),
        
        ('JUNCTION_7', 'HOSPITAL_CITY', 'seg_7', 'Industrial Ring South Access', 1.8, 75, 3, False),
        ('JUNCTION_7', 'JUNCTION_8', 'seg_7', 'Southern Outer Bypass', 2.4, 80, 3, False),
        
        ('JUNCTION_8', 'HOSPITAL_CITY', 'seg_6', 'Hospital Emergency Trauma Bay', 0.9, 50, 3, True),
        ('STATION_2', 'JUNCTION_8', 'seg_8', 'Station 2 Direct Response', 1.3, 55, 2, False),
        ('STATION_2', 'HOSPITAL_MERCY', 'seg_6', 'Station 2 to Mercy', 1.7, 60, 2, False),
        
        ('INCIDENT_SITE', 'JUNCTION_5', 'seg_5', 'Incident to River Bridge', 0.6, 45, 2, False),
        ('INCIDENT_SITE', 'HOSPITAL_CITY', 'seg_6', 'Incident Direct to Hospital', 1.4, 45, 2, False),
    ]

    for u, v, seg_id, name, length, speed, lanes, cctv in edges:
        # Add both forward and reverse edges
        G.add_edge(u, v, id=f"{u}->{v}", corridor_id=seg_id, name=name,
                   length_km=length, speed_limit_kmh=speed, lanes=lanes, has_cctv=cctv)
        G.add_edge(v, u, id=f"{v}->{u}", corridor_id=seg_id, name=f"{name} (Rev)",
                   length_km=length, speed_limit_kmh=speed, lanes=lanes, has_cctv=cctv)

    return G

if __name__ == "__main__":
    net = build_emergency_road_network()
    print(f"Emergency Network Built: {net.number_of_nodes()} Nodes, {net.number_of_edges()} Edges")
    for n, d in list(net.nodes(data=True))[:5]:
        print(f"  Node: {n} -> {d['name']} ({d['lat']}, {d['lon']})")
