// State Management
const state = {
  role: 'STUDENT',
  vehicleType: 'TWO_WHEELER',
  currentUser: null,
  activeReservation: null,
  countdownInterval: null,
  slots: [],
  selectedSlotForModal: null,
  activeCameraArea: 'P1',
  ws: null
};

// Real GPS Anchors for Reference
const CAMPUS_GPS = {
  gate: [12.823115, 80.041077],
  p1:   [12.823339, 80.042857],
  p3:   [12.823572, 80.046243]
};

// Initialization on DOM Load
document.addEventListener('DOMContentLoaded', () => {
  initUser();
  initWebSocket();
  loadAllSlots();
  loadAnalytics();
  populateSimulatorDropdown();
  loadCameraConfigs();
  
  // Periodic polling fallback for analytics & anomalies
  setInterval(() => {
    loadAnalytics();
    checkAnomalies();
  }, 4000);
});

// Portal Switching
function switchPortal(portalName) {
  document.querySelectorAll('.portal-view').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));

  if (portalName === 'customer') {
    document.getElementById('customerPortal').classList.add('active');
    document.getElementById('tabCustomerBtn').classList.add('active');
  } else {
    document.getElementById('adminPortal').classList.add('active');
    document.getElementById('tabAdminBtn').classList.add('active');
    loadAllSlots();
    loadAnalytics();
    loadCameraConfigs();
  }
}

// User Profile & Role Setup
function setRole(roleName) {
  state.role = roleName;
  document.getElementById('roleBadge').textContent = roleName;
  document.querySelectorAll('.role-selector .role-btn').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.role === roleName);
  });

  // Students can only select TWO_WHEELER
  const carBtn = document.getElementById('vehCarBtn');
  if (roleName === 'STUDENT') {
    setVehicleType('TWO_WHEELER');
    carBtn.disabled = true;
    carBtn.style.opacity = '0.4';
    carBtn.title = 'Students restricted to 2-wheelers only';
  } else {
    carBtn.disabled = false;
    carBtn.style.opacity = '1.0';
    carBtn.title = '';
  }
}

function setVehicleType(type) {
  state.vehicleType = type;
  document.getElementById('vehBikeBtn').classList.toggle('active', type === 'TWO_WHEELER');
  document.getElementById('vehCarBtn').classList.toggle('active', type === 'FOUR_WHEELER');
}

async function initUser() {
  const name = document.getElementById('userNameInput').value.trim();
  const plate = document.getElementById('userPlateInput').value.trim();
  try {
    const res = await fetch('/api/v1/auth/register', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name,
        role: state.role,
        vehicle_number: plate,
        vehicle_type: state.vehicleType
      })
    });
    if (res.ok) {
      state.currentUser = await res.json();
      checkExistingActiveReservation();
    }
  } catch (e) {
    console.error('User register error:', e);
  }
}

async function checkExistingActiveReservation() {
  if (!state.currentUser) return;
  try {
    const res = await fetch(`/api/v1/parking/reservation/active/${state.currentUser.id}`);
    const data = await res.json();
    if (data.has_active && data.reservation) {
      displayActiveReservation(data.reservation);
    }
  } catch (e) {
    console.error('Active reservation check failed:', e);
  }
}

// 1-Tap Allocation Request
async function requestParkingAllocation() {
  const errBox = document.getElementById('allocErrorMsg');
  errBox.style.display = 'none';

  // Ensure current user is synced
  await initUser();
  if (!state.currentUser) {
    errBox.textContent = 'Could not initialize user profile.';
    errBox.style.display = 'block';
    return;
  }

  const btn = document.getElementById('btnFindParking');
  btn.disabled = true;
  btn.innerHTML = '<span>⏳ Checking Camera &amp; Allocating Spot...</span>';

  // Generate idempotency key for this click
  const idempKey = 'req_' + Date.now() + '_' + Math.random().toString(36).substring(2, 9);

  try {
    const res = await fetch('/api/v1/parking/allocate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Idempotency-Key': idempKey
      },
      body: JSON.stringify({
        user_id: state.currentUser.id,
        idempotency_key: idempKey
      })
    });

    const data = await res.json();
    if (!res.ok) {
      errBox.textContent = data.detail || 'Allocation failed';
      errBox.style.display = 'block';
    } else {
      displayActiveReservation(data);
      loadAllSlots();
    }
  } catch (e) {
    errBox.textContent = 'Network error connecting to allocation service.';
    errBox.style.display = 'block';
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<span>⚡ Find &amp; Reserve Best Spot</span>';
  }
}

// Display Reservation & Start Countdown
function displayActiveReservation(reservation) {
  state.activeReservation = reservation;
  const card = document.getElementById('activeReservationCard');
  card.style.display = 'block';

  document.getElementById('resSlotBadge').textContent = reservation.slot_id;

  const areaNameMap = {
    'P1': 'Student Parking (Bikes Only) • 12.8233°N',
    'P3': 'Faculty & Staff Parking (Cars/Bikes) • 12.8236°N'
  };
  document.getElementById('resAreaName').textContent = areaNameMap[reservation.area_id] || reservation.area_id;

  // Live Camera verification indicator
  const camBadge = document.getElementById('resCamVerificationBadge');
  if (reservation.camera_verified) {
    camBadge.innerHTML = '<span>📸 Live Camera Verified Empty</span>';
    camBadge.style.background = 'rgba(16,185,129,0.18)';
    camBadge.style.color = '#34d399';
  } else {
    camBadge.innerHTML = '<span>📡 System Allocated (Camera Offline / Tolerant Fallback)</span>';
    camBadge.style.background = 'rgba(59,130,246,0.18)';
    camBadge.style.color = '#60a5fa';
  }

  // Show real Haversine distance
  const dist = reservation.route ? reservation.route.total_distance_meters : '—';
  document.getElementById('resDistanceDisplay').textContent = dist + ' m';

  // Store Google Maps URL on button
  const mapsBtn = document.getElementById('btnOpenMaps');
  mapsBtn._mapsUrl = reservation.google_maps_url || null;
  mapsBtn.disabled = !reservation.google_maps_url;

  // Render turn-by-turn steps
  const stepsList = document.getElementById('turnByTurnList');
  stepsList.innerHTML = '';
  if (reservation.route && reservation.route.turn_by_turn) {
    reservation.route.turn_by_turn.forEach(step => {
      const div = document.createElement('div');
      div.className = 'turn-step';
      div.textContent = step;
      stepsList.appendChild(div);
    });
  }

  startHoldCountdown(reservation.seconds_remaining);
}

// Open real Google Maps directions in a new tab
function openGoogleMaps() {
  const btn = document.getElementById('btnOpenMaps');
  const url = btn && btn._mapsUrl;
  if (url) {
    window.open(url, '_blank', 'noopener');
  } else if (state.activeReservation && state.activeReservation.google_maps_url) {
    window.open(state.activeReservation.google_maps_url, '_blank', 'noopener');
  } else {
    alert('Google Maps link not available. Please try again after allocation.');
  }
}

function startHoldCountdown(secondsRemaining) {
  if (state.countdownInterval) clearInterval(state.countdownInterval);
  let sec = secondsRemaining;

  const updateDisplay = () => {
    if (sec <= 0) {
      document.getElementById('resCountdownDisplay').textContent = '00:00';
      clearInterval(state.countdownInterval);
      alert('Your hold timer has expired. The parking slot has been released.');
      document.getElementById('activeReservationCard').style.display = 'none';
      state.activeReservation = null;
      loadAllSlots();
      return;
    }
    const mins = Math.floor(sec / 60);
    const s = sec % 60;
    document.getElementById('resCountdownDisplay').textContent = 
      String(mins).padStart(2, '0') + ':' + String(s).padStart(2, '0');
    sec--;
  };

  updateDisplay();
  state.countdownInterval = setInterval(updateDisplay, 1000);
}

async function cancelCurrentReservation() {
  if (!state.activeReservation) return;
  if (!confirm('Are you sure you want to release this parking reservation?')) return;

  try {
    const res = await fetch(`/api/v1/parking/reservation/${state.activeReservation.reservation_id}/cancel?user_id=${state.currentUser.id}`, {
      method: 'POST'
    });
    if (res.ok) {
      clearInterval(state.countdownInterval);
      document.getElementById('activeReservationCard').style.display = 'none';
      state.activeReservation = null;
      loadAllSlots();
    }
  } catch (e) {
    alert('Cancellation failed: ' + e);
  }
}

async function extendCurrentHold() {
  if (!state.activeReservation) return;
  try {
    const res = await fetch(`/api/v1/parking/reservation/${state.activeReservation.reservation_id}/extend?user_id=${state.currentUser.id}`, {
      method: 'POST'
    });
    const data = await res.json();
    if (res.ok) {
      alert(data.message);
      startHoldCountdown(data.seconds_remaining);
      document.getElementById('btnExtendHold').disabled = true;
      document.getElementById('btnExtendHold').style.opacity = '0.5';
    } else {
      alert(data.detail || 'Could not extend hold timer.');
    }
  } catch (e) {
    alert('Extension failed: ' + e);
  }
}

// WebSocket Connection Hub
function initWebSocket() {
  const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  const wsUrl = `${protocol}//${window.location.host}/ws/live-parking`;
  
  state.ws = new WebSocket(wsUrl);

  state.ws.onopen = () => {
    document.getElementById('wsIndicator').classList.remove('disconnected');
    document.getElementById('wsStatusText').textContent = 'Live Sync Active';
    setInterval(() => {
      if (state.ws && state.ws.readyState === WebSocket.OPEN) {
        state.ws.send('ping');
      }
    }, 15000);
  };

  state.ws.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      if (msg.type === 'SLOT_UPDATE' && Array.isArray(msg.slots)) {
        updateSlotsState(msg.slots);
      }
    } catch (e) {}
  };

  state.ws.onclose = () => {
    document.getElementById('wsIndicator').classList.add('disconnected');
    document.getElementById('wsStatusText').textContent = 'Reconnecting...';
    setTimeout(initWebSocket, 3000);
  };
}

function updateSlotsState(updatedSlotsList) {
  const updateMap = new Map(updatedSlotsList.map(s => [s.id, s]));
  state.slots.forEach(s => {
    if (updateMap.has(s.id)) {
      const u = updateMap.get(s.id);
      s.db_state = u.state;
      s.current_bikes = u.current_bikes;
      s.has_car = u.has_car;
      s.cv_detected_state = u.detected_state;
    }
  });

  renderAdminSlotsMatrix();
}

// Fetch all slots
async function loadAllSlots() {
  try {
    const res = await fetch('/api/v1/admin/slots');
    if (res.ok) {
      state.slots = await res.json();
      renderAdminSlotsMatrix();
    }
  } catch (e) {
    console.error('Failed to load slots:', e);
  }
}

// ─────────────────────────────────────────────────────────────
// ADMIN 35-SLOTS MATRIX RENDERER
// ─────────────────────────────────────────────────────────────
function renderAdminSlotsMatrix() {
  const container = document.getElementById('slotsMatrixContainer');
  if (!container) return;
  container.innerHTML = '';

  const areas = ['P1', 'P3'];
  const areaLabels = {
    'P1': 'Zone P1: Student Parking (20 Bike Slots)',
    'P3': 'Zone P3: Faculty & Staff Parking (15 Car Slots)'
  };

  areas.forEach(areaId => {
    const areaSlots = state.slots.filter(s => s.area_id === areaId);
    if (!areaSlots.length) return;

    const groupDiv = document.createElement('div');
    groupDiv.className = 'area-group';

    const header = document.createElement('div');
    header.className = 'area-header';
    header.innerHTML = `
      <div class="area-title">
        <span>📍 ${areaLabels[areaId]}</span>
      </div>
      <div style="font-size: 0.75rem; color: var(--text-muted);">
        ${areaSlots.filter(s => s.db_state === 'FREE').length} / ${areaSlots.length} Free
      </div>
    `;
    groupDiv.appendChild(header);

    const slotsGrid = document.createElement('div');
    slotsGrid.className = 'slots-container';

    areaSlots.forEach(s => {
      const card = document.createElement('div');
      card.className = `slot-box ${s.db_state.toLowerCase()}`;
      card.onclick = () => openSlotModal(s.id);

      let icon = s.slot_type === 'BIKES_ONLY' ? '🏍️' : '🚗';
      let statusText = s.db_state;
      if (s.slot_type === 'BIKES_ONLY' && s.current_bikes > 0) {
        statusText = `${s.current_bikes}/${s.bike_capacity} 🏍️`;
      }

      card.innerHTML = `
        <div class="slot-box-id">${s.id}</div>
        <div style="font-size: 1.1rem; margin: 2px 0;">${icon}</div>
        <div class="slot-box-status">${statusText}</div>
      `;
      slotsGrid.appendChild(card);
    });

    groupDiv.appendChild(slotsGrid);
    container.appendChild(groupDiv);
  });
}

// Slot Management Dialog
function openSlotModal(slotId) {
  state.selectedSlotForModal = slotId;
  const slot = state.slots.find(s => s.id === slotId);
  if (!slot) return;

  document.getElementById('modalSlotTitle').textContent = `Manage Slot ${slotId}`;
  document.getElementById('modalSlotDesc').textContent = 
    `Zone: ${slot.area_id} | Type: ${slot.slot_type} | Current: ${slot.db_state} | CV: ${slot.cv_detected_state || 'FREE'}`;

  document.getElementById('slotModal').showModal();
}

async function applySlotOverride(newState) {
  if (!state.selectedSlotForModal) return;
  try {
    const res = await fetch(`/api/v1/admin/slot/${state.selectedSlotForModal}/override`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ new_state: newState })
    });
    if (res.ok) {
      document.getElementById('slotModal').close();
      loadAllSlots();
    }
  } catch (e) {
    alert('Override failed: ' + e);
  }
}

async function applySlotRelease() {
  if (!state.selectedSlotForModal) return;
  try {
    const res = await fetch(`/api/v1/admin/slot/${state.selectedSlotForModal}/release`, {
      method: 'POST'
    });
    if (res.ok) {
      document.getElementById('slotModal').close();
      loadAllSlots();
    }
  } catch (e) {
    alert('Release failed: ' + e);
  }
}

// Camera Feed Switcher
function switchCameraFeed(areaId) {
  state.activeCameraArea = areaId;
  document.getElementById('activeCamBadge').textContent = `CAM ${areaId}`;
  const btnP1 = document.getElementById('btnCamP1');
  const btnP3 = document.getElementById('btnCamP3');
  if (btnP1) btnP1.classList.toggle('active', areaId === 'P1');
  if (btnP3) btnP3.classList.toggle('active', areaId === 'P3');

  const img = document.getElementById('liveCameraImg');
  img.src = `/api/v1/admin/cameras/${areaId}/feed?t=` + Date.now();

  const label = document.getElementById('camSourceAreaName');
  if (label) label.textContent = `Zone ${areaId}`;

  loadCameraConfigs();
}

// Quick DroidCam / ESP-Cam Connect
async function quickConnectCamera() {
  const urlInput = document.getElementById('quickCamUrl');
  const rawUrl = urlInput.value.trim();
  if (!rawUrl) {
    alert('Please enter a DroidCam/ESP-Cam IP stream URL (e.g. http://192.168.1.5:4747/video) or 0 for local webcam.');
    return;
  }

  const areaId = state.activeCameraArea;
  const isWebcam = !isNaN(rawUrl);
  const camType = isWebcam ? 'webcam' : 'droidcam';

  try {
    const res = await fetch('/api/v1/admin/cameras/config', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        area_id: areaId,
        type: camType,
        source: rawUrl
      })
    });
    const data = await res.json();
    if (res.ok) {
      alert(`Connected Zone ${areaId} to ${rawUrl}`);
      switchCameraFeed(areaId);
    } else {
      alert(data.detail || 'Could not connect camera');
    }
  } catch (e) {
    alert('Connection error: ' + e);
  }
}

async function loadCameraConfigs() {
  try {
    const res = await fetch('/api/v1/admin/cameras/config');
    if (!res.ok) return;
    const data = await res.json();
    const camCfg = data.cameras && data.cameras[state.activeCameraArea];
    if (camCfg && camCfg.camera_source) {
      const typeSpan = document.getElementById('camSourceCurrentType');
      if (typeSpan) {
        typeSpan.textContent = camCfg.camera_source.type.toUpperCase();
      }
      const urlInput = document.getElementById('quickCamUrl');
      if (urlInput && camCfg.camera_source.source && camCfg.camera_source.type !== 'mock') {
        urlInput.value = camCfg.camera_source.source;
      }
    }
  } catch (e) {}
}

// Prototype Simulator
function populateSimulatorDropdown() {
  const select = document.getElementById('simSlotSelect');
  if (!select) return;
  select.innerHTML = '';

  // 35 slots: P1-01..20 and P3-01..15
  const areas = [
    { id: 'P1', count: 20 },
    { id: 'P3', count: 15 }
  ];

  areas.forEach(a => {
    for (let i = 1; i <= a.count; i++) {
      const slotId = `${a.id}-${String(i).padStart(2, '0')}`;
      const opt = document.createElement('option');
      opt.value = slotId;
      opt.textContent = `${slotId} (${a.id === 'P1' ? 'Student Bike' : 'Faculty Car'})`;
      select.appendChild(opt);
    }
  });
}

async function simulateVehicle(type) {
  const slotId = document.getElementById('simSlotSelect').value;
  try {
    const res = await fetch('/api/v1/admin/simulate/slot', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        slot_id: slotId,
        state: type
      })
    });
    if (res.ok) {
      loadAllSlots();
    } else {
      const data = await res.json();
      alert(data.detail || 'Simulation error');
    }
  } catch (e) {
    console.error('Simulation call failed:', e);
  }
}

// Analytics and Anomalies
async function loadAnalytics() {
  try {
    const res = await fetch('/api/v1/admin/analytics');
    if (res.ok) {
      const data = await res.json();
      document.getElementById('metricTotalSlots').textContent = data.total_slots;
      document.getElementById('metricOccupancyRate').textContent = data.occupancy_rate_pct + '%';
      document.getElementById('metricFreeCount').textContent = data.slots_breakdown.free + ' Available';
      document.getElementById('metricActiveBookings').textContent = data.reservations_summary.currently_active;
      document.getElementById('metricNoShowRate').textContent = data.no_show_rate_pct + '%';
      document.getElementById('metricNoShowCount').textContent = data.reservations_summary.no_shows + ' auto-released';
    }
  } catch (e) {}
}

async function checkAnomalies() {
  try {
    const res = await fetch('/api/v1/admin/anomalies');
    if (res.ok) {
      const anomalies = await res.json();
      const container = document.getElementById('anomaliesList');
      if (!anomalies || anomalies.length === 0) {
        container.innerHTML = '<div style="font-size: 0.8rem; color: var(--text-dim); font-style: italic;">No misparking anomalies detected.</div>';
        return;
      }
      container.innerHTML = '';
      anomalies.forEach(a => {
        const div = document.createElement('div');
        div.style.padding = '0.5rem';
        div.style.marginBottom = '0.4rem';
        div.style.background = 'rgba(239, 68, 68, 0.15)';
        div.style.border = '1px solid rgba(239, 68, 68, 0.4)';
        div.style.borderRadius = 'var(--radius-sm)';
        div.style.fontSize = '0.78rem';
        div.innerHTML = `⚠️ <b>${a.type}</b> in Slot <b>${a.slot_id}</b> (${a.anomaly_reason})`;
        container.appendChild(div);
      });
    }
  } catch (e) {}
}
