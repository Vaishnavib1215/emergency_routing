/**
 * Emergency Vehicle Route & Traffic Congestion Prediction
 * Client Dashboard & Interactive Operations Controller
 */

let map;
let networkData = null;
let currentRouteResult = null;
let selectedVehicle = 'ambulance';
let segmentLayers = {};
let nodeMarkers = {};
let routePolylines = [];
let dispatchMarker = null;
let dispatchInterval = null;
let forecastChart = null;
let cctvInterval = null;

// Node custom marker icons
const ICONS = {
  station: '🚒',
  ambulance_depot: '🚑',
  fire_station: '🚒',
  hospital: '🏥',
  intersection: '⚪',
  incident: '⚠️',
  dispatch: '🏢'
};

const VEHICLE_EMOJIS = {
  ambulance: '🚑',
  fire_truck: '🚒',
  police: '🚓'
};

const VEHICLE_DESCS = {
  ambulance: 'Rapid patient transport; priority traffic signal preemption enabled.',
  fire_truck: 'Heavy apparatus; wide turning clearance & full intersection override.',
  police: 'High-speed interceptor; shoulder lane usage & agile bottleneck bypass.'
};

// Initialize Dashboard
document.addEventListener('DOMContentLoaded', () => {
  initClock();
  initMap();
  initChart();
  loadNetwork();
  startCameraStream();
});

function initClock() {
  function updateTime() {
    const now = new Date();
    document.getElementById('system-time').innerText = now.toLocaleTimeString();
  }
  updateTime();
  setInterval(updateTime, 1000);
}

function initMap() {
  // Initialize Leaflet map
  map = L.map('map', {
    center: [44.9780, -93.2550],
    zoom: 14,
    zoomControl: true
  });

  // Free OpenStreetMap tiles with dark CSS filter (no API key required)
  L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a> contributors',
    maxZoom: 19,
    className: 'map-tiles-dark'
  }).addTo(map);
}

function initChart() {
  const ctx = document.getElementById('forecastChart').getContext('2d');
  forecastChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ['t-25m', 't-20m', 't-15m', 't-10m', 't-5m', 'Now', 't+5m', 't+10m', 't+15m'],
      datasets: [
        {
          label: 'Historical Congestion',
          data: [0.35, 0.42, 0.48, 0.55, 0.62, 0.70, null, null, null],
          borderColor: '#94a3b8',
          borderWidth: 2,
          pointBackgroundColor: '#94a3b8',
          tension: 0.3
        },
        {
          label: 'CNN-LSTM Forecast',
          data: [null, null, null, null, null, 0.70, 0.82, 0.88, 0.91],
          borderColor: '#f43f5e',
          borderDash: [5, 5],
          borderWidth: 2.5,
          pointBackgroundColor: '#f43f5e',
          pointRadius: 4,
          tension: 0.3
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          labels: { color: '#94a3b8', font: { size: 10 } }
        }
      },
      scales: {
        x: {
          ticks: { color: '#94a3b8', font: { size: 9 } },
          grid: { color: 'rgba(255,255,255,0.05)' }
        },
        y: {
          min: 0,
          max: 1.0,
          ticks: {
            color: '#94a3b8',
            font: { size: 9 },
            callback: (v) => `${Math.round(v * 100)}%`
          },
          grid: { color: 'rgba(255,255,255,0.05)' }
        }
      }
    }
  });
}

function getCongestionColor(c) {
  if (c < 0.35) return '#10b981'; // Free flow (Green)
  if (c < 0.65) return '#f59e0b'; // Moderate (Amber)
  if (c < 0.85) return '#ef4444'; // Heavy (Red)
  return '#7f1d1d';               // Severe Gridlock (Dark Red)
}

async function loadNetwork() {
  try {
    const res = await fetch('/api/network');
    networkData = await res.json();
    renderNetwork(networkData);
    computeRoute(); // initial computation
  } catch (err) {
    console.error('Failed to load network:', err);
  }
}

function renderNetwork(data) {
  // Clear previous layers
  Object.values(segmentLayers).forEach(l => map.removeLayer(l));
  Object.values(nodeMarkers).forEach(m => map.removeLayer(m));
  segmentLayers = {};
  nodeMarkers = {};

  // Render edges / road segments
  data.edges.forEach(edge => {
    const u = data.nodes[edge.from_node];
    const v = data.nodes[edge.to_node];
    if (!u || !v) return;

    const latlngs = [[u.lat, u.lon], [v.lat, v.lon]];
    const color = getCongestionColor(edge.current_congestion);

    const poly = L.polyline(latlngs, {
      color: color,
      weight: 4,
      opacity: 0.75,
      dashArray: null
    }).addTo(map);

    poly.bindPopup(`
      <div style="font-family: sans-serif; font-size: 12px; color: #1e293b;">
        <strong>${edge.name}</strong><br/>
        Corridor: <code>${edge.corridor_id}</code><br/>
        Speed Limit: ${edge.speed_limit_kmh} km/h | Lanes: ${edge.lanes}<br/>
        Current Congestion: <strong>${Math.round(edge.current_congestion * 100)}%</strong><br/>
        Predicted (15m): <strong>${Math.round(edge.predicted_congestion * 100)}%</strong>
      </div>
    `);

    poly.on('click', () => {
      fetchSegmentForecast(edge.corridor_id, edge.name);
    });

    segmentLayers[`${edge.from_node}->${edge.to_node}`] = poly;
  });

  // Render nodes
  Object.entries(data.nodes).forEach(([id, node]) => {
    const iconChar = ICONS[node.icon] || ICONS[node.type] || '⚪';
    const marker = L.marker([node.lat, node.lon], {
      icon: L.divIcon({
        className: 'custom-map-icon',
        html: `<div style="font-size: 20px; text-shadow: 0 0 6px rgba(0,0,0,0.8);">${iconChar}</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12]
      })
    }).addTo(map);

    marker.bindPopup(`
      <div style="font-family: sans-serif; font-size: 12px; color: #1e293b;">
        <strong>${node.name}</strong><br/>
        ID: <code>${id}</code><br/>
        Category: ${node.type.toUpperCase()}
      </div>
    `);

    nodeMarkers[id] = marker;
  });
}

function selectVehicle(type) {
  selectedVehicle = type;
  document.querySelectorAll('.vehicle-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.type === type);
  });
  document.getElementById('vehicle-desc').innerText = VEHICLE_DESCS[type] || '';
  computeRoute();
}

function onLocationChanged() {
  computeRoute();
}

async function computeRoute() {
  const origin = document.getElementById('origin-select').value;
  const destination = document.getElementById('destination-select').value;

  if (origin === destination) {
    alert("Origin and Destination cannot be the same!");
    return;
  }

  try {
    const res = await fetch('/api/route/recommend', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        origin: origin,
        destination: destination,
        vehicle_type: selectedVehicle
      })
    });

    currentRouteResult = await res.json();
    displayRouteResults(currentRouteResult);
  } catch (err) {
    console.error('Failed to compute route:', err);
  }
}

function displayRouteResults(data) {
  // Update Analytics HUD
  const comp = data.comparison;
  document.getElementById('val-time-saved').innerText = `-${comp.time_saved_pct}%`;
  document.getElementById('val-eta').innerText = `${comp.proposed_predictive.total_time_min} min`;
  document.getElementById('val-static-eta').innerText = `${comp.static_gps.total_time_min} min`;
  document.getElementById('val-bottleneck').innerText = comp.static_gps.bottleneck ? 'Bypassed' : 'Optimal';

  // Populate Candidate Routes List
  const listEl = document.getElementById('candidate-routes-list');
  listEl.innerHTML = '';

  data.candidate_routes.forEach((route, idx) => {
    const item = document.createElement('div');
    item.className = `route-item ${route.is_recommended ? 'selected' : ''}`;
    item.innerHTML = `
      <div class="route-item-header">
        <span>Route ${String.fromCharCode(65 + idx)} (${route.total_distance_km} km)</span>
        ${route.is_recommended ? '<span class="badge-rec">RECOMMENDED</span>' : `<span style="color: #94a3b8;">+${(route.total_time_min - data.recommended_route.total_time_min).toFixed(1)}m</span>`}
      </div>
      <div class="route-item-body">
        <strong>${route.total_time_min} min</strong> &bull; Avg Congestion: ${Math.round(route.avg_congestion * 100)}%
        <br/><span style="font-size: 10px; color: #64748b;">Via: ${route.path.slice(0, 3).join(' → ')}${route.path.length > 3 ? '...' : ''}</span>
      </div>
    `;
    item.onclick = () => highlightCandidateRoute(route);
    listEl.appendChild(item);
  });

  // Highlight recommended route on map
  if (data.recommended_route) {
    highlightCandidateRoute(data.recommended_route);
    // Auto-fetch forecast for first segment in recommended path
    if (data.recommended_route.segments.length > 0) {
      const firstSeg = data.recommended_route.segments[0];
      fetchSegmentForecast(firstSeg.corridor_id, firstSeg.edge_name);
    }
  }
}

function highlightCandidateRoute(route) {
  // Clear old route lines
  routePolylines.forEach(p => map.removeLayer(p));
  routePolylines = [];

  const latlngs = [];
  route.segments.forEach(seg => {
    latlngs.push(seg.from_coords);
    latlngs.push(seg.to_coords);
  });

  // Glowing recommended path
  const glowLine = L.polyline(latlngs, {
    color: '#06b6d4',
    weight: 9,
    opacity: 0.45
  }).addTo(map);

  const mainLine = L.polyline(latlngs, {
    color: route.is_recommended ? '#10b981' : '#3b82f6',
    weight: 5,
    opacity: 0.95
  }).addTo(map);

  routePolylines.push(glowLine, mainLine);

  // Zoom to fit path
  if (latlngs.length > 0) {
    map.fitBounds(mainLine.getBounds(), { padding: [60, 60] });
  }
}

function dispatchVehicle() {
  if (!currentRouteResult || !currentRouteResult.recommended_route) {
    alert("Please compute a route first.");
    return;
  }

  // Clear existing dispatch animation
  if (dispatchInterval) clearInterval(dispatchInterval);
  if (dispatchMarker) map.removeLayer(dispatchMarker);

  const route = currentRouteResult.recommended_route;
  const waypoints = [];
  route.segments.forEach((seg, i) => {
    waypoints.push(seg.from_coords);
    if (i === route.segments.length - 1) {
      waypoints.push(seg.to_coords);
    }
  });

  const vIcon = VEHICLE_EMOJIS[selectedVehicle] || '🚑';
  dispatchMarker = L.marker(waypoints[0], {
    icon: L.divIcon({
      className: 'vehicle-transit-icon',
      html: `<div style="font-size: 28px; filter: drop-shadow(0 0 10px #10b981); transform: scale(1.2);">${vIcon}</div>`,
      iconSize: [32, 32],
      iconAnchor: [16, 16]
    })
  }).addTo(map);

  let currentIdx = 0;
  let subStep = 0;
  const stepsPerLeg = 25;

  dispatchInterval = setInterval(() => {
    if (currentIdx >= waypoints.length - 1) {
      clearInterval(dispatchInterval);
      alert(`[MISSION COMPLETE] ${VEHICLE_DESCS[selectedVehicle] || 'Emergency vehicle'} has safely reached destination!`);
      return;
    }

    const start = waypoints[currentIdx];
    const end = waypoints[currentIdx + 1];

    const lat = start[0] + (end[0] - start[0]) * (subStep / stepsPerLeg);
    const lon = start[1] + (end[1] - start[1]) * (subStep / stepsPerLeg);
    dispatchMarker.setLatLng([lat, lon]);

    subStep++;
    if (subStep > stepsPerLeg) {
      subStep = 0;
      currentIdx++;
    }
  }, 80);
}

async function fetchSegmentForecast(corridorId, corridorName) {
  try {
    document.getElementById('forecast-segment-name').innerText = `Corridor: ${corridorName || corridorId}`;
    const res = await fetch(`/api/forecast/${corridorId}`);
    const data = await res.json();

    if (forecastChart && data.labels) {
      forecastChart.data.labels = data.labels;
      forecastChart.data.datasets[0].data = data.historical;
      forecastChart.data.datasets[1].data = data.predicted;
      forecastChart.update();
    }
  } catch (err) {
    console.error('Failed to fetch forecast:', err);
  }
}

function startCameraStream() {
  const imgEl = document.getElementById('cctv-image');
  
  // Periodically refresh the camera frame and detection metrics
  async function refreshCCTV() {
    const camId = document.getElementById('camera-select').value;
    try {
      const res = await fetch(`/api/camera/stats?camera_id=${camId}`);
      const stats = await res.json();
      
      document.getElementById('stat-cars').innerText = stats.cars || 0;
      document.getElementById('stat-trucks').innerText = stats.trucks || 0;
      document.getElementById('stat-buses').innerText = stats.buses || 0;
      document.getElementById('stat-density').innerText = `${Math.round(stats.density * 100)}%`;
      
      // Update image source timestamp to trigger refresh
      imgEl.src = `/api/camera/frame?camera_id=${camId}&t=${Date.now()}`;
    } catch (e) {
      // Backend not yet running or still loading
    }
  }

  refreshCCTV();
  cctvInterval = setInterval(refreshCCTV, 1200);
}

function onCameraChange() {
  const camId = document.getElementById('camera-select').value;
  const corridorMap = {
    cam_1: 'seg_1',
    cam_3: 'seg_3',
    cam_5: 'seg_5',
    cam_6: 'seg_6'
  };
  const segId = corridorMap[camId] || 'seg_1';
  fetchSegmentForecast(segId, `Camera ${camId.toUpperCase()}`);
}
