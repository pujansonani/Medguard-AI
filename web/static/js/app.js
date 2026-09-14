/**
 * MEDGUARD AI: Unified Clinical Web Platform Logic (Cliexa Healthtech Architecture)
 */

// Global State
let currentPatientCase = "sepsis_shock";
let activeCharts = {};
window._currentObservations = [];

// Clinical Presets
const CLINICAL_PRESETS = {
  sepsis_shock: {
    id: 10042,
    stay_id: 30042,
    age: 68,
    gender: "M",
    icu_type: "MICU",
    admission_type: "EMERGENCY",
    title: "Case #10042 — Septic Shock & Acute Hypotension",
    notes: "Nursing Progress Note (Hour 18): Patient in septic shock with refractory hypotension. Norepinephrine titrated to 0.18 mcg/kg/min to maintain MAP > 65. Lactate rising sharply to 4.5 mmol/L. Oliguria noted with urine output < 15 mL/hr over past 4 hours. Critical care team notified.",
    vitals_generator: (h) => {
      const prog = h / 24.0;
      return {
        hour: h,
        heart_rate: Math.round(92 + 25 * prog + (Math.random() * 2 - 1)),
        sbp: Math.round(110 - 28 * prog),
        dbp: Math.round(68 - 18 * prog),
        map: Math.round(82 - 22 * prog),
        resp_rate: Math.round(20 + 8 * prog),
        temperature: parseFloat((38.2 + 0.8 * prog).toFixed(1)),
        spo2: Math.round(94 - 5 * prog),
        gcs: Math.max(7, Math.round(14 - 5 * prog)),
        lactate: parseFloat((1.8 + 2.7 * prog).toFixed(1)),
        creatinine: parseFloat((1.2 + 1.8 * prog).toFixed(2)),
        wbc: parseFloat((14.0 + 8.0 * prog).toFixed(1)),
        platelets: Math.round(180 - 60 * prog),
        glucose: Math.round(150 + 40 * prog),
        bun: Math.round(28 + 22 * prog)
      };
    }
  },
  ards_respiratory: {
    id: 10088,
    stay_id: 30088,
    age: 72,
    gender: "F",
    icu_type: "SICU",
    admission_type: "EMERGENCY",
    title: "Case #10088 — Severe ARDS & Mechanical Ventilation",
    notes: "Physician Assessment (Hour 20): Acute Respiratory Distress Syndrome (ARDS) secondary to aspiration pneumonia. Intubated on PRVC mode, FiO2 80%, PEEP 14 cmH2O. P/F ratio dropped to 110. Arterial blood gas indicates severe mixed metabolic and respiratory acidosis.",
    vitals_generator: (h) => {
      const prog = h / 24.0;
      return {
        hour: h,
        heart_rate: Math.round(88 + 18 * prog),
        sbp: Math.round(125 - 15 * prog),
        dbp: Math.round(75 - 10 * prog),
        map: Math.round(91 - 12 * prog),
        resp_rate: Math.round(24 + 10 * prog),
        temperature: parseFloat((37.8 + 0.5 * prog).toFixed(1)),
        spo2: Math.round(92 - 6 * prog),
        gcs: Math.max(8, Math.round(12 - 4 * prog)),
        lactate: parseFloat((1.5 + 1.8 * prog).toFixed(1)),
        creatinine: parseFloat((1.0 + 0.8 * prog).toFixed(2)),
        wbc: parseFloat((16.0 + 6.0 * prog).toFixed(1)),
        platelets: Math.round(210 - 30 * prog),
        glucose: Math.round(140 + 30 * prog),
        bun: Math.round(22 + 14 * prog)
      };
    }
  },
  postop_stable: {
    id: 10018,
    stay_id: 30018,
    age: 58,
    gender: "M",
    icu_type: "CVICU",
    admission_type: "ELECTIVE",
    title: "Case #10018 — Post-Op Observation (Recovering)",
    notes: "Nursing Progress Note (Hour 22): Patient recovering well following elective CABG. Hemodynamically stable off all vasoactive drips. Vitals: HR 74 in normal sinus rhythm, MAP 85 mmHg, SpO2 99% on 2L nasal cannula. Clear breath sounds, alert x 4. Preparing for floor transfer.",
    vitals_generator: (h) => {
      const prog = h / 24.0;
      return {
        hour: h,
        heart_rate: Math.round(82 - 8 * prog),
        sbp: Math.round(124 + 4 * prog),
        dbp: Math.round(76 + 2 * prog),
        map: Math.round(88 + 2 * prog),
        resp_rate: Math.round(16 - 2 * prog),
        temperature: parseFloat((37.0 + 0.1 * prog).toFixed(1)),
        spo2: Math.round(97 + 2 * prog),
        gcs: 15,
        lactate: parseFloat((1.4 - 0.5 * prog).toFixed(1)),
        creatinine: 0.9,
        wbc: 7.8,
        platelets: 240,
        glucose: 110,
        bun: 14
      };
    }
  }
};

// Ward Patient Grid
const WARD_BEDS = [
  { bed: "Bed 01", id: 10042, age: 68, risk: 0.84, status: "High Risk", unit: "MICU", hr: 118, map: 58, lactate: 4.5, caseKey: "sepsis_shock" },
  { bed: "Bed 02", id: 10088, age: 72, risk: 0.76, status: "High Risk", unit: "SICU", hr: 104, map: 72, lactate: 3.1, caseKey: "ards_respiratory" },
  { bed: "Bed 03", id: 10018, age: 58, risk: 0.08, status: "Low Risk", unit: "CVICU", hr: 74, map: 88, lactate: 0.9, caseKey: "postop_stable" },
  { bed: "Bed 04", id: 10095, age: 64, risk: 0.32, status: "Moderate", unit: "MICU", hr: 88, map: 78, lactate: 1.8, caseKey: "sepsis_shock" },
  { bed: "Bed 05", id: 10104, age: 80, risk: 0.62, status: "High Risk", unit: "CCU", hr: 96, map: 64, lactate: 2.7, caseKey: "ards_respiratory" },
  { bed: "Bed 06", id: 10112, age: 51, risk: 0.12, status: "Low Risk", unit: "TSICU", hr: 78, map: 84, lactate: 1.1, caseKey: "postop_stable" }
];

document.addEventListener("DOMContentLoaded", () => {
  initScrollSpy();
  loadPatientCase("sepsis_shock");
  renderWardBeds();
  loadEvaluationBenchmark();
});

// ScrollSpy to update active navbar pill smoothly as user scrolls
function initScrollSpy() {
  const sections = ["portal-home", "simulation-section", "patient-assessment", "explainability-studio", "research-benchmark", "ward-telemetry"];
  
  window.addEventListener("scroll", () => {
    const scrollPos = window.scrollY + 140;
    
    sections.forEach((secId) => {
      const el = document.getElementById(secId);
      const navBtn = document.getElementById(`nav-${secId}`);
      if (el && navBtn) {
        const top = el.offsetTop;
        const height = el.offsetHeight;
        if (scrollPos >= top && scrollPos < top + height) {
          document.querySelectorAll(".nav-tab-btn").forEach((b) => b.classList.remove("active"));
          navBtn.classList.add("active");
        }
      }
    });
  });
}

// Load Preset Patient Case
function loadPatientCase(caseKey) {
  currentPatientCase = caseKey;
  const preset = CLINICAL_PRESETS[caseKey];
  if (!preset) return;

  // Update Preset Button Active Styles
  document.querySelectorAll(".btn-preset").forEach((b) => b.classList.remove("active"));
  const activeBtnId = caseKey === "sepsis_shock" ? "preset-sepsis" : (caseKey === "ards_respiratory" ? "preset-ards" : "preset-postop");
  const activeBtn = document.getElementById(activeBtnId);
  if (activeBtn) activeBtn.classList.add("active");

  // Update Demographic Header
  document.getElementById("pt-id-display").innerText = `#${preset.id}`;
  document.getElementById("pt-demog-display").innerText = `${preset.age} yrs / ${preset.gender} • ${preset.icu_type}`;
  document.getElementById("pt-notes-input").value = preset.notes;

  // Generate 24 Hourly Observations
  const observations = [];
  for (let h = 0; h < 24; h++) {
    observations.push(preset.vitals_generator(h));
  }
  window._currentObservations = observations;

  // Sync Live Sliders with Hour 23 values
  const lastObs = observations[23];
  document.getElementById("slider-map").value = lastObs.map;
  document.getElementById("slider-map-val").innerText = `${lastObs.map} mmHg`;
  document.getElementById("slider-lactate").value = lastObs.lactate;
  document.getElementById("slider-lactate-val").innerText = `${lastObs.lactate} mmol/L`;
  document.getElementById("slider-hr").value = lastObs.heart_rate;
  document.getElementById("slider-hr-val").innerText = `${lastObs.heart_rate} bpm`;
  document.getElementById("slider-spo2").value = lastObs.spo2;
  document.getElementById("slider-spo2-val").innerText = `${lastObs.spo2}%`;

  // Sync Live ECG Header
  document.getElementById("live-ecg-hr").innerText = lastObs.heart_rate;
  document.getElementById("live-ecg-map").innerText = lastObs.map;
  document.getElementById("live-ecg-spo2").innerText = `${lastObs.spo2}%`;

  // Render 24h Chart
  renderVitalsTrajectoryChart(observations);

  // Run Real Prediction
  analyzePatientRecord();
}

// Real-Time Slider Change Listener
function onVitalSliderChange(param, value) {
  const valNum = parseFloat(value);
  const labelMap = {
    map: `${valNum} mmHg`,
    lactate: `${valNum.toFixed(1)} mmol/L`,
    heart_rate: `${valNum} bpm`,
    spo2: `${valNum}%`
  };
  const labelIdMap = {
    map: "slider-map-val",
    lactate: "slider-lactate-val",
    heart_rate: "slider-hr-val",
    spo2: "slider-spo2-val"
  };

  if (labelIdMap[param]) {
    document.getElementById(labelIdMap[param]).innerText = labelMap[param];
  }

  // Update in-memory 24th hour observation
  if (window._currentObservations && window._currentObservations.length === 24) {
    window._currentObservations[23][param] = valNum;
    // Update chart
    renderVitalsTrajectoryChart(window._currentObservations);
    // Instant re-analysis
    analyzePatientRecord();
  }
}

// Render 24-Hour Vitals Trajectory
function renderVitalsTrajectoryChart(observations) {
  const ctx = document.getElementById("vitalsTrajectoryCanvas");
  if (!ctx || typeof Chart === "undefined") return;

  if (activeCharts["vitalsTrajectory"]) {
    try { activeCharts["vitalsTrajectory"].destroy(); } catch(e) {}
  }

  const hours = observations.map((o) => `Hr ${o.hour}`);
  const mapVals = observations.map((o) => o.map);
  const hrVals = observations.map((o) => o.heart_rate);
  const lactVals = observations.map((o) => o.lactate);

  activeCharts["vitalsTrajectory"] = new Chart(ctx, {
    type: "line",
    data: {
      labels: hours,
      datasets: [
        {
          label: "MAP (mmHg)",
          data: mapVals,
          borderColor: "#3ECFB2",
          backgroundColor: "rgba(62, 207, 178, 0.12)",
          tension: 0.3,
          borderWidth: 2.5,
          pointRadius: 3,
          yAxisID: "y"
        },
        {
          label: "Heart Rate (bpm)",
          data: hrVals,
          borderColor: "#A78BFA",
          backgroundColor: "transparent",
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3,
          yAxisID: "y"
        },
        {
          label: "Lactate (mmol/L)",
          data: lactVals,
          borderColor: "#FF5F56",
          backgroundColor: "transparent",
          borderDash: [5, 5],
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3,
          yAxisID: "y1"
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { labels: { color: "#8B9AAD", font: { family: "Plus Jakarta Sans", size: 12, weight: 600 } } },
        tooltip: { backgroundColor: "#0F2B46", borderColor: "rgba(62, 207, 178, 0.3)", borderWidth: 1 }
      },
      scales: {
        x: { grid: { color: "rgba(255, 255, 255, 0.06)" }, ticks: { color: "#8B9AAD" } },
        y: {
          grid: { color: "rgba(255, 255, 255, 0.06)" },
          ticks: { color: "#8B9AAD" },
          title: { display: true, text: "Vitals (mmHg / bpm)", color: "#8B9AAD" }
        },
        y1: {
          position: "right",
          grid: { drawOnChartArea: false },
          ticks: { color: "#EF4444" },
          title: { display: true, text: "Lactate (mmol/L)", color: "#EF4444" }
        }
      }
    }
  });
}

// Call FastAPI /predict/multimodal and /explain
async function analyzePatientRecord() {
  const preset = CLINICAL_PRESETS[currentPatientCase];
  const observations = window._currentObservations || [];
  const noteText = document.getElementById("pt-notes-input").value;

  const payload = {
    demographics: {
      subject_id: preset.id,
      stay_id: preset.stay_id,
      age: preset.age,
      gender: preset.gender,
      icu_type: preset.icu_type,
      admission_type: preset.admission_type
    },
    timeseries: observations,
    clinical_notes: noteText,
    num_mc_samples: 30
  };

  try {
    const res = await fetch("/predict/multimodal", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      const predData = await res.json();
      renderPredictionResults(predData);
    }

    const expRes = await fetch("/explain", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (expRes.ok) {
      const expData = await expRes.json();
      renderExplainability(expData);
    }
  } catch (err) {
    console.warn("API inference fallback:", err);
  }
}

// Update UI with Prediction Outputs
function renderPredictionResults(data) {
  const risk = data.calibrated_risk;
  const rawRisk = data.mortality_48h_risk;
  const unc = data.epistemic_uncertainty;

  // Risk Score Value
  document.getElementById("risk-val-display").innerText = `${(risk * 100).toFixed(1)}%`;
  document.getElementById("raw-risk-sub").innerText = `Calibrated: ${(risk * 100).toFixed(1)}% | Raw Logit: ${(rawRisk * 100).toFixed(1)}%`;

  // Risk Badge
  const badgeEl = document.getElementById("risk-badge-box");
  if (risk < 0.20) {
    badgeEl.className = "risk-level-badge badge-low";
    badgeEl.innerText = "LOW RISK (< 20%)";
  } else if (risk < 0.50) {
    badgeEl.className = "risk-level-badge badge-moderate";
    badgeEl.innerText = "MODERATE RISK (20% - 50%)";
  } else {
    badgeEl.className = "risk-level-badge badge-high";
    badgeEl.innerText = "HIGH RISK (≥ 50%)";
  }

  // Uncertainty
  document.getElementById("uncertainty-val-display").innerText = `σ = ${unc.toFixed(3)} (Confidence: ${(data.confidence_score * 100).toFixed(1)}%)`;
  document.getElementById("uncertainty-bar-fill").style.width = `${Math.min(100, unc * 500)}%`;

  // Modality Weights
  const structW = data.modality_weights.structured_physiology;
  const textW = data.modality_weights.unstructured_clinical_text;
  document.getElementById("struct-w-label").innerText = `${(structW * 100).toFixed(1)}%`;
  document.getElementById("text-w-label").innerText = `${(textW * 100).toFixed(1)}%`;
  document.getElementById("meter-fill-struct").style.width = `${structW * 100}%`;
  document.getElementById("meter-fill-text").style.width = `${textW * 100}%`;

  // Clinical Scores
  document.getElementById("sofa-proxy-val").innerText = data.sofa_proxy_score;
  document.getElementById("saps-proxy-val").innerText = data.saps_ii_proxy_score;

  // Trajectory Chart
  renderRiskTrajectoryChart(data.risk_trajectory);
}

// Render 6h, 12h, 24h, 48h Trajectory
function renderRiskTrajectoryChart(traj) {
  const ctx = document.getElementById("riskTrajectoryCanvas");
  if (!ctx || typeof Chart === "undefined") return;

  if (activeCharts["riskTrajectory"]) {
    try { activeCharts["riskTrajectory"].destroy(); } catch(e) {}
  }

  const horizons = ["6 Hours", "12 Hours", "24 Hours", "48 Hours"];
  const values = [traj.prob_6h * 100, traj.prob_12h * 100, traj.prob_24h * 100, traj.prob_48h * 100];

  activeCharts["riskTrajectory"] = new Chart(ctx, {
    type: "line",
    data: {
      labels: horizons,
      datasets: [
        {
          label: "Predicted Deterioration Probability (%)",
          data: values,
          borderColor: values[3] >= 50 ? "#FF5F56" : "#3ECFB2",
          backgroundColor: values[3] >= 50 ? "rgba(255, 95, 86, 0.15)" : "rgba(62, 207, 178, 0.15)",
          fill: true,
          tension: 0.35,
          borderWidth: 3,
          pointRadius: 6,
          pointHoverRadius: 8
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#0F2B46",
          borderColor: "rgba(62, 207, 178, 0.3)",
          borderWidth: 1,
          callbacks: {
            label: (item) => `Mortality Risk: ${item.raw.toFixed(1)}%`
          }
        }
      },
      scales: {
        x: { grid: { color: "rgba(255, 255, 255, 0.06)" }, ticks: { color: "#8B9AAD", font: { family: "Plus Jakarta Sans" } } },
        y: {
          min: 0,
          max: 100,
          grid: { color: "rgba(255, 255, 255, 0.06)" },
          ticks: { color: "#8B9AAD", callback: (val) => `${val}%`, font: { family: "Space Mono" } }
        }
      }
    }
  });
}

// Render SHAP & Text Attribution
function renderExplainability(exp) {
  // Highlighted Clinical Text
  const textContainer = document.getElementById("highlighted-note-box");
  if (textContainer) {
    textContainer.innerHTML = exp.highlighted_notes_html;
  }

  // SHAP Waterfall Bar Chart
  const ctx = document.getElementById("shapFeatureCanvas");
  if (!ctx || typeof Chart === "undefined") return;

  if (activeCharts["shapBars"]) {
    try { activeCharts["shapBars"].destroy(); } catch(e) {}
  }

  const topPos = exp.top_positive_features || [];
  const topNeg = exp.top_negative_features || [];
  const combined = [...topPos, ...topNeg].slice(0, 8);

  const labels = combined.map((f) => f[0]);
  const values = combined.map((f) => f[1]);
  const colors = values.map((v) => (v >= 0 ? "#FF5F56" : "#3ECFB2"));

  activeCharts["shapBars"] = new Chart(ctx, {
    type: "bar",
    data: {
      labels: labels,
      datasets: [
        {
          label: "SHAP Impact on 48h Risk",
          data: values,
          backgroundColor: colors,
          borderRadius: 6
        }
      ]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: { backgroundColor: "#0F2B46", borderColor: "rgba(62, 207, 178, 0.3)", borderWidth: 1 }
      },
      scales: {
        x: { grid: { color: "rgba(255, 255, 255, 0.06)" }, ticks: { color: "#8B9AAD" } },
        y: { grid: { display: false }, ticks: { color: "#F8FAFC", font: { family: "Plus Jakarta Sans", weight: "600" } } }
      }
    }
  });
}

// Ward Patient Telemetry Grid
function renderWardBeds() {
  const container = document.getElementById("ward-beds-container");
  if (!container) return;

  container.innerHTML = WARD_BEDS.map((b) => {
    let badgeClass = "badge-low";
    if (b.status === "Moderate") badgeClass = "badge-moderate";
    if (b.status === "High Risk") badgeClass = "badge-high";

    return `
      <div class="bed-card" onclick="loadPatientCase('${b.caseKey}'); document.getElementById('patient-assessment').scrollIntoView({behavior: 'smooth'});">
        <div class="bed-card-header">
          <div class="bed-number">${b.bed}</div>
          <div class="risk-level-badge ${badgeClass}" style="margin: 0; padding: 3px 10px; font-size: 11px;">${b.status}</div>
        </div>
        <div style="font-size: 13px; color: #fff; font-weight: 700;">Patient #${b.id} (${b.age}y • ${b.unit})</div>
        <div style="font-size: 24px; font-weight: 800; color: ${b.risk >= 0.5 ? "#F87171" : "#38BDF8"}; margin: 6px 0; font-family: 'JetBrains Mono';">
          ${(b.risk * 100).toFixed(0)}% <span style="font-size: 11px; font-weight: 500; color: #94A3B8;">48h Risk</span>
        </div>
        <div class="bed-vitals-row">
          <span>HR: <strong style="color: #fff;">${b.hr}</strong> bpm</span>
          <span>MAP: <strong style="color: #fff;">${b.map}</strong> mmHg</span>
          <span>Lact: <strong style="color: #fff;">${b.lactate}</strong></span>
        </div>
      </div>
    `;
  }).join("");
}

// Fetch and Populate Empirical Research Benchmark from /metrics
async function loadEvaluationBenchmark() {
  const defaultAblation = [
    { model_name: "Multimodal (Gated Fusion)", modality_type: "Multimodal Fusion", ci: { auroc: { ci_str: "0.952 [0.908, 0.989]" }, auprc: { ci_str: "0.793 [0.588, 0.951]" } }, metrics: { f1: 0.846, brier_score: 0.083, ece: 0.077 } },
    { model_name: "Multimodal (Cross-Attention)", modality_type: "Multimodal Fusion", ci: { auroc: { ci_str: "0.952 [0.907, 0.986]" }, auprc: { ci_str: "0.825 [0.619, 0.957]" } }, metrics: { f1: 0.846, brier_score: 0.076, ece: 0.061 } },
    { model_name: "Multimodal (Intermediate Concat)", modality_type: "Multimodal Fusion", ci: { auroc: { ci_str: "0.949 [0.910, 0.979]" }, auprc: { ci_str: "0.832 [0.682, 0.945]" } }, metrics: { f1: 0.846, brier_score: 0.076, ece: 0.059 } },
    { model_name: "SOFA Proxy Score", modality_type: "Clinical Baseline", ci: { auroc: { ci_str: "0.944 [0.901, 0.983]" }, auprc: { ci_str: "0.774 [0.578, 0.915]" } }, metrics: { f1: 0.438, brier_score: 0.093, ece: 0.120 } },
    { model_name: "Temporal GRU", modality_type: "Structured Temporal", ci: { auroc: { ci_str: "0.940 [0.895, 0.982]" }, auprc: { ci_str: "0.769 [0.568, 0.936]" } }, metrics: { f1: 0.846, brier_score: 0.078, ece: 0.065 } },
    { model_name: "Logistic Regression", modality_type: "Structured Tabular", ci: { auroc: { ci_str: "0.937 [0.884, 0.981]" }, auprc: { ci_str: "0.768 [0.563, 0.929]" } }, metrics: { f1: 0.682, brier_score: 0.104, ece: 0.108 } },
    { model_name: "SAPS II Proxy Score", modality_type: "Clinical Baseline", ci: { auroc: { ci_str: "0.934 [0.877, 0.977]" }, auprc: { ci_str: "0.788 [0.609, 0.932]" } }, metrics: { f1: 0.444, brier_score: 0.101, ece: 0.200 } },
    { model_name: "XGBoost", modality_type: "Structured Tabular", ci: { auroc: { ci_str: "0.933 [0.881, 0.981]" }, auprc: { ci_str: "0.739 [0.527, 0.921]" } }, metrics: { f1: 0.846, brier_score: 0.086, ece: 0.092 } },
    { model_name: "ClinicalBERT NLP", modality_type: "Text Only", ci: { auroc: { ci_str: "0.934 [0.878, 0.972]" }, auprc: { ci_str: "0.755 [0.537, 0.904]" } }, metrics: { f1: 0.846, brier_score: 0.075, ece: 0.059 } }
  ];

  const tbody = document.getElementById("ablation-table-body");
  if (!tbody) return;

  function renderRows(rows) {
    tbody.innerHTML = rows.map((row) => {
      const isTop = row.modality_type.includes("Multimodal");
      return `
        <tr class="${isTop ? "highlight-row" : ""}">
          <td><strong>${row.model_name}</strong></td>
          <td><span style="font-size: 12px; color: #94A3B8;">${row.modality_type}</span></td>
          <td><strong style="color: #38BDF8;">${row.ci.auroc.ci_str}</strong></td>
          <td>${row.ci.auprc.ci_str}</td>
          <td>${row.metrics.f1.toFixed(3)}</td>
          <td>${row.metrics.brier_score.toFixed(3)}</td>
          <td>${row.metrics.ece.toFixed(3)}</td>
        </tr>
      `;
    }).join("");
  }

  // Initial render with validated empirical values
  renderRows(defaultAblation);

  try {
    const res = await fetch("/metrics");
    const data = await res.json();
    if (data.ablation_results) {
      const sorted = Object.values(data.ablation_results).sort((a, b) => b.metrics.auroc - a.metrics.auroc);
      renderRows(sorted);
    }
  } catch (err) {
    console.warn("Using offline benchmark:", err);
  }
}
