/**
 * MEDGUARD AI: Advanced Motion, ECG Waveforms, Neural Particles & Simulation Player
 */

document.addEventListener("DOMContentLoaded", () => {
  initECGWaveformMonitor();
  initNeuralParticleNetwork();
  initSimulationPlayer();
  initScrollReveals();
  initTiltEffects();
  initCountUpCounters();
});

// =============================================================================
// 1. LIVE REAL-TIME ECG & ARTERIAL LINE WAVEFORM ENGINE (60 FPS CANVAS)
// =============================================================================
function initECGWaveformMonitor() {
  const canvas = document.getElementById("ecgCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let width = (canvas.width = canvas.offsetWidth);
  let height = (canvas.height = canvas.offsetHeight);

  window.addEventListener("resize", () => {
    if (!canvas) return;
    width = canvas.width = canvas.offsetWidth;
    height = canvas.height = canvas.offsetHeight;
  });

  let sweepX = 0;
  const speed = 2.4;
  const points = [];
  const maxPoints = Math.ceil(width);

  // Pre-fill
  for (let i = 0; i < maxPoints; i++) {
    points[i] = height / 2;
  }

  let beatPhase = 0;
  let bpm = 88;

  function getEcgY(phase) {
    const p = phase % 1.0;
    const mid = height / 2;
    // P wave
    if (p > 0.1 && p < 0.2) {
      return mid - Math.sin((p - 0.1) * Math.PI * 10) * 8;
    }
    // Q dip
    if (p >= 0.22 && p < 0.26) {
      return mid + 6;
    }
    // R spike
    if (p >= 0.26 && p < 0.32) {
      return mid - Math.sin((p - 0.26) * Math.PI * 16.6) * 32;
    }
    // S dip
    if (p >= 0.32 && p < 0.38) {
      return mid + 10;
    }
    // T wave
    if (p > 0.45 && p < 0.65) {
      return mid - Math.sin((p - 0.45) * Math.PI * 5) * 12;
    }
    // Baseline with slight noise
    return mid + (Math.random() * 2 - 1);
  }

  function animate() {
    beatPhase += (bpm / 60) * 0.016;
    sweepX = (sweepX + speed) % width;
    const currentIdx = Math.floor(sweepX);

    points[currentIdx] = getEcgY(beatPhase);

    // Render ECG Waveform
    ctx.clearRect(0, 0, width, height);

    // Draw phosphor glow trace
    ctx.lineWidth = 2.4;
    ctx.strokeStyle = "#3ECFB2";
    ctx.shadowBlur = 10;
    ctx.shadowColor = "#3ECFB2";
    ctx.beginPath();

    for (let x = 0; x < width; x++) {
      // Clear 20px ahead of sweep line
      if (Math.abs(x - sweepX) < 18) continue;
      
      const y = points[x] || height / 2;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Draw glowing sweep head
    ctx.shadowBlur = 14;
    ctx.shadowColor = "#7CEFD8";
    ctx.fillStyle = "#7CEFD8";
    ctx.beginPath();
    ctx.arc(sweepX, points[currentIdx] || height / 2, 4.0, 0, Math.PI * 2);
    ctx.fill();

    requestAnimationFrame(animate);
  }

  animate();
}

// =============================================================================
// 2. MULTIMODAL NEURAL PARTICLE NETWORK & DATA FLOW PIPELINE
// =============================================================================
function initNeuralParticleNetwork() {
  const canvas = document.getElementById("neuralCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let width = (canvas.width = canvas.offsetWidth);
  let height = (canvas.height = canvas.offsetHeight);

  window.addEventListener("resize", () => {
    if (!canvas) return;
    width = canvas.width = canvas.offsetWidth;
    height = canvas.height = canvas.offsetHeight;
    initNodes();
  });

  let mouse = { x: width / 2, y: height / 2, active: false };
  canvas.addEventListener("mousemove", (e) => {
    const rect = canvas.getBoundingClientRect();
    mouse.x = e.clientX - rect.left;
    mouse.y = e.clientY - rect.top;
    mouse.active = true;
  });
  canvas.addEventListener("mouseleave", () => { mouse.active = false; });

  let nodes = [];
  const nodeCount = 38;

  function initNodes() {
    nodes = [];
    // Input Group A (Physiology / Clinera Teal)
    for (let i = 0; i < 12; i++) {
      nodes.push({
        x: width * 0.15 + (Math.random() * 80 - 40),
        y: height * 0.2 + (i / 12) * height * 0.6,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: 4,
        type: "physiology",
        color: "#3ECFB2",
        pulse: Math.random() * Math.PI
      });
    }

    // Input Group B (Clinical Text / Soft Purple)
    for (let i = 0; i < 12; i++) {
      nodes.push({
        x: width * 0.45 + (Math.random() * 80 - 40),
        y: height * 0.15 + (i / 12) * height * 0.7,
        vx: (Math.random() - 0.5) * 0.5,
        vy: (Math.random() - 0.5) * 0.5,
        radius: 4,
        type: "text",
        color: "#A78BFA",
        pulse: Math.random() * Math.PI
      });
    }

    // Gated Fusion Center Core
    nodes.push({
      x: width * 0.75,
      y: height * 0.5,
      vx: 0,
      vy: 0,
      radius: 12,
      type: "fusion_core",
      color: "#3ECFB2",
      pulse: 0
    });
  }

  initNodes();

  // Animated Data Packets
  const packets = [];
  for (let p = 0; p < 16; p++) {
    packets.push({
      progress: Math.random(),
      fromIdx: Math.floor(Math.random() * 24),
      speed: 0.006 + Math.random() * 0.008
    });
  }

  function renderNeuralNet() {
    ctx.clearRect(0, 0, width, height);

    // Draw Connections between nodes
    for (let i = 0; i < nodes.length; i++) {
      const n1 = nodes[i];
      for (let j = i + 1; j < nodes.length; j++) {
        const n2 = nodes[j];
        const dist = Math.hypot(n1.x - n2.x, n1.y - n2.y);
        
        if (dist < 140) {
          ctx.beginPath();
          ctx.strokeStyle = `rgba(0, 162, 237, ${0.35 * (1 - dist / 140)})`;
          ctx.lineWidth = 1.2;
          ctx.moveTo(n1.x, n1.y);
          ctx.lineTo(n2.x, n2.y);
          ctx.stroke();
        }
      }
    }

    // Draw Data Flow Packets toward Fusion Core
    const coreNode = nodes[nodes.length - 1];
    packets.forEach((pkt) => {
      pkt.progress += pkt.speed;
      if (pkt.progress >= 1.0) {
        pkt.progress = 0;
        pkt.fromIdx = Math.floor(Math.random() * 24);
      }
      const source = nodes[pkt.fromIdx];
      if (source && coreNode) {
        const px = source.x + (coreNode.x - source.x) * pkt.progress;
        const py = source.y + (coreNode.y - source.y) * pkt.progress;
        
        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = source.color;
        ctx.shadowBlur = 8;
        ctx.shadowColor = source.color;
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    });

    // Draw Nodes
    nodes.forEach((node) => {
      node.pulse += 0.04;
      const pulseSize = Math.sin(node.pulse) * 1.5;

      ctx.beginPath();
      ctx.arc(node.x, node.y, node.radius + pulseSize, 0, Math.PI * 2);
      ctx.fillStyle = node.color;
      ctx.shadowBlur = 10;
      ctx.shadowColor = node.color;
      ctx.fill();
      ctx.shadowBlur = 0;

      // Mouse interaction
      if (mouse.active) {
        const dMouse = Math.hypot(node.x - mouse.x, node.y - mouse.y);
        if (dMouse < 80) {
          node.x += (node.x - mouse.x) * 0.03;
          node.y += (node.y - mouse.y) * 0.03;
        }
      }
    });

    // Draw rotating energy halo on Gated Fusion Core
    if (coreNode) {
      ctx.beginPath();
      ctx.arc(coreNode.x, coreNode.y, 24 + Math.sin(Date.now() * 0.005) * 4, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    requestAnimationFrame(renderNeuralNet);
  }

  renderNeuralNet();
}

// =============================================================================
// 3. INTERACTIVE CLINICAL SIMULATION WALKTHROUGH PLAYER & TELEMETRY ENGINE
// =============================================================================
let simPlaying = false;
let simProgress = 0;
let simTimer = null;
let simSpeed = 1;
let simAudioEnabled = false;
let audioCtx = null;

const SIMULATION_STAGES = [
  { hour: 0, title: "ICU Admission", desc: "Patient admitted post-op. Vital signs baseline stable. MAP 88 mmHg, Lactate 1.2 mmol/L. Normal sinus rhythm.", hr: 78, map: 88, lactate: 1.2, risk: 0.12, vaso: false, status: "Normal Baseline" },
  { hour: 8, title: "Early Tachycardia & Oliguria", desc: "Heart rate increases to 98 bpm. Urine output declining (< 35 mL/hr). Fluid bolus administered.", hr: 98, map: 78, lactate: 1.8, risk: 0.28, vaso: false, status: "Elevated Monitoring" },
  { hour: 14, title: "Refractory Hypotension & Sepsis", desc: "MAP drops to 64 mmHg. Norepinephrine initiated at 0.06 mcg/kg/min. Clinical nursing note entered.", hr: 108, map: 64, lactate: 2.9, risk: 0.58, vaso: true, status: "Sepsis Alert" },
  { hour: 20, title: "Severe Shock Escalation", desc: "Lactate surges to 4.2 mmol/L. Vasopressor titrated to 0.18 mcg/kg/min. Severe metabolic acidosis.", hr: 122, map: 56, lactate: 4.2, risk: 0.82, vaso: true, status: "Critical Deterioration" },
  { hour: 24, title: "Multimodal Early Warning Triggered", desc: "48-Hour deterioration forecast triggers at 84.2% mortality probability. SHAP & text attribution highlighted for ICU team.", hr: 128, map: 52, lactate: 4.8, risk: 0.842, vaso: true, status: "AI Early Warning (High Risk)" }
];

function initSimulationPlayer() {
  const playBtn = document.getElementById("simPlayBtn");
  if (!playBtn) return;

  playBtn.addEventListener("click", () => {
    if (simPlaying) pauseSimulation();
    else startSimulation();
  });

  const timeline = document.getElementById("simTimelineBar");
  if (timeline) {
    timeline.addEventListener("click", (e) => {
      const rect = timeline.getBoundingClientRect();
      const clickRatio = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      simProgress = clickRatio;
      updateSimulationHUD();
    });
  }

  initSimMonitorCanvas();
  updateSimulationHUD();
}

function startSimulation() {
  simPlaying = true;
  const playBtn = document.getElementById("simPlayBtn");
  if (playBtn) playBtn.innerHTML = "⏸ Pause Demo";
  
  if (simTimer) clearInterval(simTimer);
  simTimer = setInterval(() => {
    simProgress += 0.003 * simSpeed;
    if (simProgress >= 1.0) {
      simProgress = 1.0;
      pauseSimulation();
    }
    updateSimulationHUD();
  }, 40);
}

function pauseSimulation() {
  simPlaying = false;
  if (simTimer) clearInterval(simTimer);
  const btn = document.getElementById("simPlayBtn");
  if (btn) btn.innerHTML = "▶ Play Simulation";
}

function setSimSpeed(speed) {
  simSpeed = speed;
  document.querySelectorAll(".speed-btn").forEach((b) => {
    b.classList.toggle("active", parseFloat(b.getAttribute("data-speed")) === speed);
  });
}

function jumpToSimStage(stageIdx) {
  simProgress = stageIdx / (SIMULATION_STAGES.length - 1);
  document.querySelectorAll(".chapter-tab").forEach((t, i) => {
    t.classList.toggle("active", i === stageIdx);
  });
  updateSimulationHUD();
}

function stepSim(hourDelta) {
  const currentHour = simProgress * 24;
  const newHour = Math.max(0, Math.min(24, currentHour + hourDelta));
  simProgress = newHour / 24;
  updateSimulationHUD();
}

function toggleSimAudio() {
  simAudioEnabled = !simAudioEnabled;
  const label = document.getElementById("simAudioLabel");
  const btn = document.getElementById("simAudioBtn");
  if (label) label.innerText = simAudioEnabled ? "ON (Beeper Active)" : "Off";
  if (btn) btn.classList.toggle("btn-primary", simAudioEnabled);

  if (simAudioEnabled && !audioCtx) {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch (e) {
      console.warn("Web Audio not supported:", e);
    }
  }
}

function playTelemetryBeep(freq = 880) {
  if (!simAudioEnabled || !audioCtx) return;
  try {
    if (audioCtx.state === "suspended") audioCtx.resume();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.type = "sine";
    osc.frequency.setValueAtTime(freq, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.04, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.08);
    osc.connect(gain);
    gain.connect(audioCtx.destination);
    osc.start();
    osc.stop(audioCtx.currentTime + 0.08);
  } catch (e) {}
}

function updateSimulationHUD() {
  const progressBar = document.getElementById("simTimelineProgress");
  if (progressBar) progressBar.style.width = `${simProgress * 100}%`;

  const totalStages = SIMULATION_STAGES.length - 1;
  const currentProgExact = simProgress * totalStages;
  const stageIdx = Math.min(totalStages, Math.floor(currentProgExact));
  const nextStageIdx = Math.min(totalStages, stageIdx + 1);
  const frac = currentProgExact - stageIdx;

  const s1 = SIMULATION_STAGES[stageIdx];
  const s2 = SIMULATION_STAGES[nextStageIdx];

  const currentHr = Math.round(s1.hr + (s2.hr - s1.hr) * frac);
  const currentMap = Math.round(s1.map + (s2.map - s1.map) * frac);
  const currentLact = parseFloat((s1.lactate + (s2.lactate - s1.lactate) * frac).toFixed(1));
  const currentRisk = parseFloat((s1.risk + (s2.risk - s1.risk) * frac).toFixed(3));
  const exactHour = Math.round(simProgress * 24);

  // Update HUD text
  const hudHour = document.getElementById("simHudHour");
  const hudTitle = document.getElementById("simHudTitle");
  const hudDesc = document.getElementById("simHudDesc");
  const hudRisk = document.getElementById("simHudRisk");
  const hudBadge = document.getElementById("simHudBadge");
  const hudMap = document.getElementById("simHudMap");
  const hudLact = document.getElementById("simHudLact");
  const hudVaso = document.getElementById("simHudVasoTag");
  const alertOverlay = document.getElementById("simAlertOverlay");

  if (hudHour) hudHour.innerText = `Hour ${exactHour} / 24`;
  if (hudTitle) hudTitle.innerText = s1.title;
  if (hudDesc) hudDesc.innerText = s1.desc;
  if (hudRisk) hudRisk.innerText = `${(currentRisk * 100).toFixed(1)}%`;
  if (hudMap) hudMap.innerText = `${currentMap} mmHg`;
  if (hudLact) hudLact.innerText = `${currentLact} mmol/L`;
  
  if (hudVaso) {
    hudVaso.style.display = s1.vaso ? "inline-block" : "none";
  }

  if (hudBadge) {
    hudBadge.innerText = s1.status;
    hudBadge.className = currentRisk >= 0.5 ? "risk-level-badge badge-high" : (currentRisk >= 0.25 ? "risk-level-badge badge-moderate" : "risk-level-badge badge-low");
  }

  // Active chapter button sync
  document.querySelectorAll(".chapter-tab").forEach((t, i) => {
    t.classList.toggle("active", i === stageIdx);
  });

  // Alert overlay at Hour 24
  if (alertOverlay) {
    alertOverlay.style.display = simProgress >= 0.95 ? "block" : "none";
  }

  // Update monitor state
  window._simMonitorState = {
    hr: currentHr,
    map: currentMap,
    lactate: currentLact,
    risk: currentRisk,
    hour: exactHour,
    isAlert: currentRisk >= 0.5
  };
}

// =============================================================================
// SIMULATION CANVAS GRAPHICS VIEWPORT
// =============================================================================
function initSimMonitorCanvas() {
  const canvas = document.getElementById("simMonitorCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  let width = (canvas.width = canvas.offsetWidth || 800);
  let height = (canvas.height = canvas.offsetHeight || 220);

  window.addEventListener("resize", () => {
    if (!canvas) return;
    width = canvas.width = canvas.offsetWidth;
    height = canvas.height = canvas.offsetHeight;
  });

  let sweepX = 0;
  const trace1 = [];
  const trace2 = [];
  for (let i = 0; i < 1000; i++) {
    trace1[i] = height * 0.35;
    trace2[i] = height * 0.75;
  }

  let phase = 0;
  let lastBeepTime = 0;

  function renderSimFrame() {
    const state = window._simMonitorState || { hr: 78, map: 88, lactate: 1.2, risk: 0.12, isAlert: false };
    const hr = state.hr;
    const isCritical = state.risk >= 0.5;

    phase += (hr / 60) * 0.02;
    sweepX = (sweepX + 3.2) % width;
    const currIdx = Math.floor(sweepX);

    // ECG lead
    const p = phase % 1.0;
    const mid1 = height * 0.32;
    let ecgY = mid1;
    if (p > 0.1 && p < 0.2) ecgY = mid1 - Math.sin((p - 0.1) * Math.PI * 10) * 7;
    else if (p >= 0.22 && p < 0.26) ecgY = mid1 + 5;
    else if (p >= 0.26 && p < 0.32) {
      ecgY = mid1 - Math.sin((p - 0.26) * Math.PI * 16.6) * (isCritical ? 38 : 28);
      // Trigger audio beep on R peak
      if (Date.now() - lastBeepTime > (60000 / hr) * 0.8) {
        playTelemetryBeep(isCritical ? 1040 : 880);
        lastBeepTime = Date.now();
      }
    }
    else if (p >= 0.32 && p < 0.38) ecgY = mid1 + 8;
    else if (p > 0.45 && p < 0.65) ecgY = mid1 - Math.sin((p - 0.45) * Math.PI * 5) * 10;
    else ecgY = mid1 + (Math.random() * 1.5 - 0.75);

    trace1[currIdx] = ecgY;

    // Arterial line waveform (dicrotic notch)
    const mid2 = height * 0.72;
    const pArt = (phase + 0.15) % 1.0;
    let artY = mid2;
    const artAmp = isCritical ? 12 : 24;
    if (pArt < 0.3) artY = mid2 - Math.sin(pArt * Math.PI * 3.3) * artAmp;
    else if (pArt >= 0.3 && pArt < 0.45) artY = mid2 - 6; // Dicrotic notch
    else if (pArt >= 0.45 && pArt < 0.7) artY = mid2 - Math.sin((pArt - 0.45) * Math.PI * 4) * (artAmp * 0.4);
    else artY = mid2;

    trace2[currIdx] = artY;

    // Background Clear
    ctx.fillStyle = "#020710";
    ctx.fillRect(0, 0, width, height);

    // Subtle Grid
    ctx.strokeStyle = "rgba(0, 162, 237, 0.08)";
    ctx.lineWidth = 1;
    for (let gx = 0; gx < width; gx += 25) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, height);
      ctx.stroke();
    }
    for (let gy = 0; gy < height; gy += 25) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(width, gy);
      ctx.stroke();
    }

    // Channel 1: Lead II ECG (Green/Red)
    ctx.strokeStyle = isCritical ? "#EF4444" : "#12A57F";
    ctx.shadowColor = ctx.strokeStyle;
    ctx.shadowBlur = 8;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      if (Math.abs(x - sweepX) < 16) continue;
      const y = trace1[x] || mid1;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Channel 2: Arterial Blood Pressure Wave (Red/Cyan)
    ctx.strokeStyle = isCritical ? "#F87171" : "#38BDF8";
    ctx.shadowColor = ctx.strokeStyle;
    ctx.shadowBlur = 6;
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    for (let x = 0; x < width; x++) {
      if (Math.abs(x - sweepX) < 16) continue;
      const y = trace2[x] || mid2;
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Channel Labels Overlay
    ctx.font = "600 12px 'JetBrains Mono', monospace";
    ctx.fillStyle = isCritical ? "#EF4444" : "#12A57F";
    ctx.fillText(`ECG (Lead II) — HR: ${state.hr} bpm`, 16, 24);

    ctx.fillStyle = isCritical ? "#F87171" : "#38BDF8";
    ctx.fillText(`ART (Arterial Line) — MAP: ${state.map} mmHg`, 16, height * 0.52);

    // Glowing Sweep Heads
    ctx.fillStyle = "#fff";
    ctx.beginPath();
    ctx.arc(sweepX, trace1[currIdx] || mid1, 3.5, 0, Math.PI * 2);
    ctx.arc(sweepX, trace2[currIdx] || mid2, 3.5, 0, Math.PI * 2);
    ctx.fill();

    requestAnimationFrame(renderSimFrame);
  }

  renderSimFrame();
}

// =============================================================================
// 4. ANIMATED METRIC COUNTERS & SCROLL REVEAL OBSERVERS
// =============================================================================
function initCountUpCounters() {
  const counters = document.querySelectorAll(".stat-value[data-count]");
  counters.forEach((el) => {
    const targetText = el.getAttribute("data-count");
    const targetNum = parseFloat(targetText);
    if (!isNaN(targetNum)) {
      let current = 0;
      const step = targetNum / 40;
      const isDecimal = targetText.includes(".");
      const timer = setInterval(() => {
        current += step;
        if (current >= targetNum) {
          el.innerText = isDecimal ? targetNum.toFixed(3) : targetNum.toString();
          clearInterval(timer);
        } else {
          el.innerText = isDecimal ? current.toFixed(3) : Math.floor(current).toString();
        }
      }, 25);
    }
  });
}

function initScrollReveals() {
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.classList.add("revealed");
        }
      });
    },
    { threshold: 0.1 }
  );

  document.querySelectorAll(".panel-card, .stat-card").forEach((el) => {
    el.classList.add("reveal-on-scroll");
    observer.observe(el);
  });
}

// 3D Card Hover Tilt Effect
function initTiltEffects() {
  document.querySelectorAll(".stat-card, .bed-card").forEach((card) => {
    card.addEventListener("mousemove", (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left - rect.width / 2;
      const y = e.clientY - rect.top - rect.height / 2;
      const tiltX = (y / (rect.height / 2)) * -6;
      const tiltY = (x / (rect.width / 2)) * 6;
      card.style.transform = `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg) translateY(-2px)`;
    });
    card.addEventListener("mouseleave", () => {
      card.style.transform = "perspective(1000px) rotateX(0deg) rotateY(0deg) translateY(0px)";
    });
  });
}

