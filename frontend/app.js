/**
 * frontend/app.js
 * Live Stream Controller & Digital Twin SOC Dashboard Logic
 * Polls telemetry samples, requests real-time inference & XAI from FastAPI backend,
 * and updates UI with zero latency.
 */

let currentScenario = 'Normal';
let isStreaming = true;
let streamInterval = null;

// Initial setup on page load
document.addEventListener('DOMContentLoaded', () => {
  fetchSystemStatus();
  startTelemetryStream();
});

// Fetch system health and calibrated threshold from backend
async function fetchSystemStatus() {
  try {
    const res = await fetch('/api/status');
    if (!res.ok) throw new Error('Status endpoint failed');
    const data = await res.json();
    document.getElementById('thresholdBadge').innerText = `THRESHOLD: ${data.threshold.toFixed(4)}`;
    document.getElementById('valThreshold').innerText = data.threshold.toFixed(4);
    document.getElementById('modelArchBadge').innerText = `ARCH: ${data.model_architecture.split(' ')[0]}-AE (${data.layer_spec})`;
  } catch (err) {
    console.warn('Backend not yet reachable:', err.message);
  }
}

// Set active simulation scenario
function setScenario(scenario) {
  currentScenario = scenario;
  
  // Update button active state
  document.querySelectorAll('.control-deck button').forEach(btn => btn.classList.remove('active'));
  if (scenario === 'Normal') document.getElementById('btnNormal').classList.add('active');
  if (scenario === 'DoS_Flooding') document.getElementById('btnDos').classList.add('active');
  if (scenario === 'Data_Injection') document.getElementById('btnInjection').classList.add('active');
  if (scenario === 'ZeroDay_Replay') document.getElementById('btnZeroDay').classList.add('active');

  // Immediately poll one sample for responsive feedback
  pollTelemetryStep();
}

// Toggle stream play / pause
function toggleStream() {
  isStreaming = !isStreaming;
  const btn = document.getElementById('btnPause');
  if (isStreaming) {
    btn.innerText = '⏸ Pause Stream';
    btn.classList.remove('active');
  } else {
    btn.innerText = '▶ Resume Stream';
    btn.classList.add('active');
  }
}

// Telemetry Polling Loop (Runs every 1500ms)
function startTelemetryStream() {
  if (streamInterval) clearInterval(streamInterval);
  streamInterval = setInterval(() => {
    if (isStreaming) {
      pollTelemetryStep();
    }
  }, 1500);
}

// Single step: Fetch Sample -> Run Predict -> Run Explain -> Update DOM
async function pollTelemetryStep() {
  try {
    // 1. Get raw telemetry sample from dataset
    const sampleRes = await fetch(`/api/telemetry/sample?scenario=${currentScenario}`);
    if (!sampleRes.ok) return;
    const sampleJson = await sampleRes.json();
    const telemetry = sampleJson.telemetry;

    // 2. Update telemetry card values
    updateTelemetryCards(telemetry);

    // 3. Post to KAN Digital Twin predict endpoint
    const predictRes = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(telemetry)
    });
    if (!predictRes.ok) return;
    const predictData = await predictRes.json();

    // 4. Post to explain endpoint for root-cause analysis
    const explainRes = await fetch('/api/explain', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(telemetry)
    });
    let explainData = null;
    if (explainRes.ok) {
      explainData = await explainRes.json();
    }

    // 5. Update UI with prediction and XAI
    updateDashboardUI(predictData, explainData, sampleJson.scenario);

  } catch (err) {
    console.error('Telemetry polling error:', err);
  }
}

// Update telemetry numbers in the grid
function updateTelemetryCards(t) {
  document.getElementById('val_hr').innerText = Math.round(t.heart_rate);
  document.getElementById('val_resp').innerText = t.respiration_rate.toFixed(1);
  document.getElementById('val_spo2').innerText = t.spo2.toFixed(1);
  document.getElementById('val_bpsys').innerText = Math.round(t.bp_systolic);
  document.getElementById('val_bpdia').innerText = Math.round(t.bp_diastolic);
  document.getElementById('val_temp').innerText = t.temperature.toFixed(1);
  document.getElementById('val_prate').innerText = Math.round(t.packet_rate);
  document.getElementById('val_psize').innerText = Math.round(t.packet_size);
  document.getElementById('val_jitter').innerText = t.jitter.toFixed(1);
  document.getElementById('val_latency').innerText = t.latency.toFixed(1);
}

// Update Hero Status Banner and XAI Bars
function updateDashboardUI(pred, explain, scenarioName) {
  const banner = document.getElementById('twinBanner');
  const statusText = document.getElementById('twinStatusText');
  const xaiStatus = document.getElementById('xaiStatusPill');

  document.getElementById('valAnomalyScore').innerText = pred.anomaly_score.toFixed(4);
  document.getElementById('valDivergence').innerText = `${pred.twin_divergence_pct}%`;

  if (pred.is_anomaly) {
    banner.classList.add('breach');
    statusText.innerHTML = `<span>⚠</span> ZERO-DAY BREACH DETECTED [${scenarioName.toUpperCase()}]`;
    xaiStatus.innerText = 'DESYNCHRONIZATION DETECTED';
    xaiStatus.style.color = 'var(--status-danger-text)';
  } else {
    banner.classList.remove('breach');
    statusText.innerHTML = `<span>✓</span> DIGITAL TWIN SYNCHRONIZED [NOMINAL]`;
    xaiStatus.innerText = 'BASELINE NOMINAL';
    xaiStatus.style.color = 'var(--status-safe-text)';
  }

  // Render Root-Cause Breakdown Bars
  if (explain && explain.all_feature_discrepancies) {
    renderRootCauses(explain.all_feature_discrepancies, pred.threshold, pred.is_anomaly);
  }
}

// Render feature attribution bars
function renderRootCauses(discrepancies, threshold, isAnomaly) {
  const container = document.getElementById('causeList');
  container.innerHTML = '';

  // Sort descending by discrepancy
  const entries = Object.entries(discrepancies).sort((a, b) => b[1] - a[1]);

  // Max value for relative bar width
  const maxVal = Math.max(...entries.map(e => e[1]), threshold * 1.5);

  entries.forEach(([feat, val]) => {
    const isExceeded = val > threshold;
    const widthPct = Math.min(100, Math.max(3, (val / maxVal) * 100));

    // Highlight card if anomalous
    highlightCard(feat, isExceeded);

    const row = document.createElement('div');
    row.className = 'cause-row';
    row.innerHTML = `
      <div class="cause-info">
        <span style="font-weight: ${isExceeded ? '800' : '500'}; color: ${isExceeded ? 'var(--status-danger-text)' : 'var(--text-main)'}">
          ${isExceeded ? '⚠ ' : ''}${feat}
        </span>
        <span style="color: ${isExceeded ? 'var(--status-danger-text)' : 'var(--text-muted)'}">
          Error: ${val.toFixed(3)} ${isExceeded ? '[VIOLATION]' : ''}
        </span>
      </div>
      <div class="bar-container">
        <div class="bar-fill ${isExceeded ? 'danger' : ''}" style="width: ${widthPct}%"></div>
      </div>
    `;
    container.appendChild(row);
  });
}

// Add danger highlight to telemetry cards exceeding threshold
function highlightCard(feat, isExceeded) {
  const map = {
    'heart_rate': 'card_hr',
    'respiration_rate': 'card_resp',
    'spo2': 'card_spo2',
    'bp_systolic': 'card_bpsys',
    'bp_diastolic': 'card_bpdia',
    'temperature': 'card_temp',
    'packet_rate': 'card_prate',
    'packet_size': 'card_psize',
    'jitter': 'card_jitter',
    'latency': 'card_latency'
  };
  const cardId = map[feat];
  if (!cardId) return;
  const card = document.getElementById(cardId);
  if (card) {
    if (isExceeded) {
      card.classList.add('danger');
    } else {
      card.classList.remove('danger');
    }
  }
}

// Switch between KAN Splines image and Feature Attribution Matrix image
function switchVisualTab(tab) {
  const img = document.getElementById('visualImage');
  const btnSplines = document.getElementById('tabSplines');
  const btnAttr = document.getElementById('tabAttribution');

  if (tab === 'splines') {
    img.src = '/static/kan_splines.png';
    btnSplines.classList.add('active');
    btnAttr.classList.remove('active');
  } else {
    img.src = '/static/feature_attribution.png';
    btnAttr.classList.add('active');
    btnSplines.classList.remove('active');
  }
}
