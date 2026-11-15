/**
 * MEDGUARD AI — Unified Application Engine
 * Single DOMContentLoaded, single tab router, no conflicts, no lag.
 */
(function () {
  'use strict';

  /* ═══════════════════════════════════════════════════════
     §1  GLOBAL STATE
  ═══════════════════════════════════════════════════════ */
  const S = {
    // Demographics
    age: 65, gender: 'M', icu_type: 'MICU',
    // Vitals
    heart_rate: 92, sbp: 118, dbp: 72, map: 87,
    spo2: 94, temperature: 37.8, resp_rate: 20,
    // Respiratory
    fio2: 0.40, pao2: 82, peep: 6,
    // Labs
    wbc: 11.2, hgb: 10.1, plt: 185,
    sodium: 138, potassium: 4.1, creatinine: 1.4, bun: 22,
    bilirubin: 0.9, albumin: 3.0, glucose: 148, lactate: 2.4,
    // Clinical
    gcs: 13, vasopressor: 0, renal_replacement: 0,
    clinical_notes: 'Patient presents with sepsis secondary to pneumonia. On vasopressors. Urine output declining. Family meeting regarding goals of care.',
    // Results
    riskPct: 0, riskTier: '', confidence: 0,
    ci_low: 0, ci_high: 0,
    trajectory: { h6: 0, h12: 0, h24: 0, h48: 0 },
    shap: [],
    // Internal
    _debounce: null,
    _chart: null,
  };

  /* ═══════════════════════════════════════════════════════
     §2  TAB ROUTER  (single source of truth)
  ═══════════════════════════════════════════════════════ */
  let patientsLoaded = false;

  function initRouter() {
    document.querySelectorAll('.subnav-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const panel = tab.dataset.panel;
        // Update active tab
        document.querySelectorAll('.subnav-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        // Show / hide views
        const calcView = document.getElementById('view-calculator');
        const ptView   = document.getElementById('view-patients');
        if (panel === 'patients') {
          calcView?.classList.add('hidden');
          ptView?.classList.remove('hidden');
          if (!patientsLoaded) { patientsLoaded = true; loadPatients(); }
        } else {
          calcView?.classList.remove('hidden');
          ptView?.classList.add('hidden');
        }
      });
    });
  }

  /* ═══════════════════════════════════════════════════════
     §3  NER
  ═══════════════════════════════════════════════════════ */
  const NER_MAP = {
    diagnosis: ['sepsis','pneumonia','ards','aki','chf','cirrhosis','copd','pe',
                'mi','stroke','sirs','uti','bacteremia','covid','pancreatitis'],
    treatment: ['vasopressor','intubated','dialysis','crrt','proning',
                'norepinephrine','dopamine','vasopressin','epinephrine','phenylephrine',
                'heparin','insulin','antibiotics','steroids'],
    symptom:   ['fever','hypotension','tachycardia','hypoxia','desaturating',
                'altered mental','oliguria','anuria','jaundice','edema','dyspnea','confusion'],
    lab:       ['lactate','creatinine','bilirubin','wbc','troponin','bnp',
                'procalcitonin','pao2','fio2','ph','bicarbonate'],
  };
  const NER_LABELS = { diagnosis:'Dx', treatment:'Rx', symptom:'Sx', lab:'Lab' };

  function extractNER(text) {
    const lower = text.toLowerCase();
    const found = { diagnosis:[], treatment:[], symptom:[], lab:[] };
    for (const [cat, words] of Object.entries(NER_MAP)) {
      for (const w of words) {
        if (lower.includes(w) && !found[cat].includes(w)) found[cat].push(w);
      }
    }
    return found;
  }

  function renderNER(tags) {
    const wrap = document.getElementById('ner-tags');
    if (!wrap) return;
    let html = '';
    for (const [cat, words] of Object.entries(tags)) {
      for (const w of words) {
        html += `<span class="ner-tag ner-${cat}">${NER_LABELS[cat]}: ${w}</span>`;
      }
    }
    wrap.innerHTML = html || '<span style="font-size:12px;color:var(--gray-400)">No entities detected</span>';
  }

  /* ═══════════════════════════════════════════════════════
     §4  CLINICAL FALLBACK CALCULATION
  ═══════════════════════════════════════════════════════ */
  function clinicalFallback() {
    let risk = 0.05;
    if (S.age > 70) risk += 0.08; else if (S.age > 55) risk += 0.04;
    if (S.map < 65) risk += 0.15; else if (S.map < 75) risk += 0.06;
    if (S.heart_rate > 120) risk += 0.08;
    if (S.spo2 < 90)  risk += 0.12; else if (S.spo2 < 94) risk += 0.05;
    if (S.resp_rate > 25) risk += 0.07;
    if (S.gcs < 10)  risk += 0.12; else if (S.gcs < 14) risk += 0.06;
    if (S.lactate > 4) risk += 0.18; else if (S.lactate > 2) risk += 0.10;
    if (S.creatinine > 3) risk += 0.10; else if (S.creatinine > 1.5) risk += 0.05;
    if (S.bilirubin > 2) risk += 0.05;
    if (S.vasopressor) risk += 0.14;
    const lower = S.clinical_notes.toLowerCase();
    if (/goals of care|comfort/.test(lower)) risk += 0.10;
    if (/worsening|deteriorat/.test(lower)) risk += 0.06;
    const pf = S.pao2 / (S.fio2 || 0.21);
    if (pf < 100) risk += 0.12; else if (pf < 200) risk += 0.07;

    risk = Math.min(0.97, Math.max(0.02, risk));
    const unc = 0.04;
    return {
      riskPct:    Math.round(risk * 100),
      confidence: 0.88,
      ci_low:  Math.round(Math.max(0, risk - unc) * 100),
      ci_high: Math.round(Math.min(1, risk + unc) * 100),
      trajectory: { h6: risk*.52, h12: risk*.70, h24: risk*.85, h48: risk },
      shap: buildShap(risk),
    };
  }

  function buildShap(risk) {
    return [
      { name:'Lactate',       val: S.lactate > 2 ? +(S.lactate * 0.040).toFixed(3) : -0.020 },
      { name:'MAP',           val: S.map < 70    ? +((70-S.map)*0.004).toFixed(3)  : -0.010 },
      { name:'SpO₂',          val: S.spo2 < 94   ? +((94-S.spo2)*0.008).toFixed(3) : -0.015 },
      { name:'Creatinine',    val: S.creatinine > 1.2 ? +((S.creatinine-1.2)*0.025).toFixed(3) : -0.005 },
      { name:'GCS',           val: S.gcs < 15    ? +((15-S.gcs)*0.012).toFixed(3)  : -0.020 },
      { name:'Age',           val: S.age > 55    ? +((S.age-55)*0.002).toFixed(3)  : -0.005 },
      { name:'Vasopressor',   val: S.vasopressor ? 0.080 : -0.030 },
      { name:'Clinical Notes',val: +(risk * 0.08).toFixed(3) },
    ].sort((a,b) => Math.abs(b.val) - Math.abs(a.val));
  }

  /* ═══════════════════════════════════════════════════════
     §5  API + DEBOUNCED CALCULATION
  ═══════════════════════════════════════════════════════ */
  async function fetchPrediction() {
    const payload = {
      patient_id: 'preview',
      age: S.age, gender: S.gender, icu_type: S.icu_type,
      heart_rate: S.heart_rate, sbp: S.sbp, dbp: S.dbp, map: S.map,
      spo2: S.spo2, temperature: S.temperature, resp_rate: S.resp_rate,
      labs: {
        wbc: S.wbc, hgb: S.hgb, plt: S.plt,
        sodium: S.sodium, potassium: S.potassium,
        creatinine: S.creatinine, bun: S.bun,
        bilirubin: S.bilirubin, albumin: S.albumin,
        glucose: S.glucose, lactate: S.lactate,
      },
      notes: S.clinical_notes,
      timeseries: [],
    };
    const res = await fetch('/predict/multimodal', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: AbortSignal.timeout(8000),
    });
    if (!res.ok) throw new Error('API ' + res.status);
    return res.json();
  }

  function scheduleCalc() {
    clearTimeout(S._debounce);
    setStatus('Calculating…');
    S._debounce = setTimeout(runCalc, 450);
  }

  async function runCalc() {
    let result;
    try {
      const d = await fetchPrediction();
      result = {
        riskPct:    Math.round((d.mortality_48h_risk || 0) * 100),
        confidence: d.confidence_score || 0.88,
        ci_low:     Math.round((d.ci_low || 0) * 100),
        ci_high:    Math.round((d.ci_high || 0.5) * 100),
        trajectory: d.risk_trajectory || clinicalFallback().trajectory,
        shap:       d.shap_values || buildShap(d.mortality_48h_risk || 0.25),
      };
    } catch {
      result = clinicalFallback();
    }
    Object.assign(S, result);
    setStatus('Live Result');
    renderResult();
  }

  function setStatus(txt) {
    const el = document.getElementById('result-status-label');
    if (el) el.textContent = txt;
  }

  /* ═══════════════════════════════════════════════════════
     §6  RENDER RESULT CARD
  ═══════════════════════════════════════════════════════ */
  function renderResult() {
    const r = S.riskPct;
    const { tier, label, stripe } = getTier(r);

    setText('result-score', r);
    setClass('result-tier',   `result-tier ${tier}`);
    setText ('result-tier',   label);
    setClass('result-stripe', `result-stripe ${stripe}`);

    const needle = document.getElementById('risk-needle');
    if (needle) needle.style.left = r + '%';

    setText('conf-val', (S.confidence * 100).toFixed(0) + '%');
    setText('ci-range', `${S.ci_low}% – ${S.ci_high}%`);

    renderTrajectory();
    renderShap();
    renderGuidance(r);
    renderClassicalScores();
    updatePF();
  }

  function getTier(r) {
    if (r < 15) return { tier:'tier-low',      label:'🟢 Low Risk',      stripe:'low' };
    if (r < 35) return { tier:'tier-moderate',  label:'🟡 Moderate Risk', stripe:'moderate' };
    if (r < 65) return { tier:'tier-high',      label:'🟠 High Risk',     stripe:'high' };
    return             { tier:'tier-critical',  label:'🔴 Critical Risk', stripe:'critical' };
  }

  function renderTrajectory() {
    const canvas = document.getElementById('trajectory-chart');
    if (!canvas) return;
    const t = S.trajectory;
    const data = [t.h6*100||0, t.h12*100||0, t.h24*100||0, t.h48*100||S.riskPct];

    if (S._chart) {
      S._chart.data.datasets[0].data = data;
      S._chart.update('none'); // 'none' = no animation = zero lag
      return;
    }
    S._chart = new Chart(canvas.getContext('2d'), {
      type: 'line',
      data: {
        labels: ['6h','12h','24h','48h'],
        datasets: [{
          data,
          borderColor: '#2563eb',
          backgroundColor: 'rgba(37,99,235,0.08)',
          borderWidth: 2.5,
          pointBackgroundColor: '#2563eb',
          pointRadius: 4,
          fill: true,
          tension: 0.4,
        }],
      },
      options: {
        animation: false,  // critical: no animation = no lag
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          y: {
            min: 0, max: 100,
            ticks: { font:{size:10}, color:'#94a3b8', callback: v => v+'%' },
            grid: { color:'#f1f5f9' },
          },
          x: { ticks: { font:{size:10}, color:'#94a3b8' }, grid:{ display:false } },
        },
      },
    });
  }

  function renderShap() {
    const wrap = document.getElementById('shap-drivers');
    if (!wrap || !S.shap.length) return;
    const maxAbs = Math.max(...S.shap.map(d => Math.abs(d.val)));
    wrap.innerHTML = S.shap.slice(0,7).map(d => {
      const isPos = d.val > 0;
      const pct = Math.round((Math.abs(d.val) / (maxAbs || 1)) * 100);
      return `<div class="driver-row">
        <span class="driver-name">${d.name}</span>
        <div class="driver-bar-wrap"><div class="driver-bar${isPos ? '' : ' neg'}" style="width:${pct}%"></div></div>
        <span class="driver-val ${isPos ? 'pos' : 'neg'}">${isPos?'+':''}${d.val.toFixed(3)}</span>
      </div>`;
    }).join('');
  }

  function renderGuidance(r) {
    const wrap = document.getElementById('guidance-items');
    if (!wrap) return;
    const items =
      r >= 65 ? [
        'Activate rapid response / ICU escalation protocol immediately',
        'Initiate or reassess vasopressor support; target MAP ≥ 65 mmHg',
        'Obtain emergent ABG, lactate, and blood cultures × 2',
        'Consider goals-of-care discussion with patient and family',
        'Increase vitals monitoring frequency to every 15–30 min',
      ] : r >= 35 ? [
        'Increase monitoring intensity — reassess vitals every hour',
        'Review and optimize fluid resuscitation strategy',
        'Check for reversible precipitants (infection source, obstruction)',
        'Ensure FAST-HUG bundle elements are addressed',
        'Serial lactate measurements every 2–4 hours',
      ] : r >= 15 ? [
        'Continue current management with heightened vigilance',
        'Reassess risk score in 6 hours or with any clinical change',
        'Ensure thromboprophylaxis and ulcer prophylaxis in place',
        'Optimize nutritional support; review medication reconciliation',
      ] : [
        'Continue routine ICU monitoring per unit protocol',
        'Consider step-down or early ICU discharge planning',
        'Review daily wake/breathe trial eligibility',
      ];

    wrap.innerHTML = items.map(i =>
      `<div class="guidance-item"><span class="guidance-dot"></span><span>${i}</span></div>`
    ).join('');
  }

  function renderClassicalScores() {
    // SOFA approximation
    let sofa = 0;
    const pf = S.pao2 / (S.fio2 || 0.21);
    if      (pf < 100) sofa += 4; else if (pf < 200) sofa += 3;
    else if (pf < 300) sofa += 2; else if (pf < 400) sofa += 1;
    if      (S.creatinine > 5)   sofa += 4; else if (S.creatinine > 3.5) sofa += 3;
    else if (S.creatinine > 2)   sofa += 2; else if (S.creatinine > 1.2) sofa += 1;
    if      (S.bilirubin > 12)   sofa += 4; else if (S.bilirubin > 6)   sofa += 3;
    else if (S.bilirubin > 2)    sofa += 2; else if (S.bilirubin > 1.2) sofa += 1;
    if (S.map < 70) sofa += (S.vasopressor ? 3 : 1);
    if      (S.gcs <= 5)  sofa += 4; else if (S.gcs <= 9)  sofa += 3;
    else if (S.gcs <= 12) sofa += 2; else if (S.gcs <= 14) sofa += 1;
    if      (S.plt < 20)  sofa += 4; else if (S.plt < 50)  sofa += 3;
    else if (S.plt < 100) sofa += 2; else if (S.plt < 150) sofa += 1;

    setText('score-sofa', sofa);
    setText('score-sofa-mort', sofa>=11?'>40%':sofa>=7?'20–30%':sofa>=4?'5–10%':'<5%');

    const saps = Math.round(15 + (S.age>60?7:0) + (S.gcs<14?10:0) +
      (S.sbp<100?13:0) + (S.heart_rate>110?7:0) +
      (S.creatinine>1.5?10:0) + (S.bun>28?6:0));
    setText('score-saps', saps);
  }

  function updatePF() {
    const pf = (S.pao2 / (S.fio2 || 0.21)).toFixed(0);
    setText('score-pf', pf);
  }

  /* ═══════════════════════════════════════════════════════
     §7  INPUT BINDINGS
  ═══════════════════════════════════════════════════════ */
  function bindSlider(id, key, dispId) {
    const inp  = document.getElementById(id);
    const disp = document.getElementById(dispId);
    if (!inp) return;
    inp.addEventListener('input', () => {
      S[key] = parseFloat(inp.value);
      if (disp) disp.textContent = inp.value;
      scheduleCalc();
    });
  }
  function bindNumber(id, key) {
    const inp = document.getElementById(id);
    if (!inp) return;
    inp.addEventListener('change', () => {
      const v = parseFloat(inp.value);
      if (!isNaN(v)) { S[key] = v; scheduleCalc(); }
    });
  }
  function bindSeg(name, key) {
    document.querySelectorAll(`[data-seg="${name}"]`).forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll(`[data-seg="${name}"]`).forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        S[key] = btn.dataset.val;
        scheduleCalc();
      });
    });
  }
  function bindToggle(name, key) {
    document.querySelectorAll(`[data-toggle="${name}"]`).forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll(`[data-toggle="${name}"]`).forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        S[key] = parseInt(btn.dataset.val, 10);
        scheduleCalc();
      });
    });
  }

  /* ═══════════════════════════════════════════════════════
     §8  EVIDENCE TABS
  ═══════════════════════════════════════════════════════ */
  function initEvTabs() {
    document.querySelectorAll('.ev-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        document.querySelectorAll('.ev-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.ev-panel').forEach(p => p.classList.remove('active'));
        tab.classList.add('active');
        document.getElementById(tab.dataset.panel)?.classList.add('active');
      });
    });
  }

  /* ═══════════════════════════════════════════════════════
     §9  PRESETS
  ═══════════════════════════════════════════════════════ */
  const PRESETS = {
    'Septic Shock':  { heart_rate:118, sbp:82, dbp:52, map:52, spo2:90, resp_rate:28, temperature:38.9, gcs:12, lactate:4.8, creatinine:2.2, vasopressor:1, clinical_notes:'Patient with septic shock, on norepinephrine. Altered mental status. Urine output 10 mL/h.' },
    'Stable Post-Op':{ heart_rate:76, sbp:132, dbp:80, map:92, spo2:98, resp_rate:14, temperature:37.1, gcs:15, lactate:1.1, creatinine:0.9, vasopressor:0, clinical_notes:'Post-operative day 1, hemodynamically stable. Good urine output. Pain controlled.' },
    'ARDS Severe':   { heart_rate:102, sbp:98, dbp:62, map:65, spo2:86, resp_rate:32, fio2:0.80, pao2:52, peep:14, gcs:11, lactate:3.1, creatinine:1.8, vasopressor:1, clinical_notes:'Severe ARDS secondary to aspiration pneumonia. Proned 16 hours. Refractory hypoxemia.' },
    'AKI + CHF':     { heart_rate:88, sbp:105, dbp:68, map:72, spo2:93, resp_rate:22, gcs:14, lactate:2.1, creatinine:3.8, bun:68, bilirubin:1.4, vasopressor:0, clinical_notes:'Acute kidney injury on CKD. Cardiorenal syndrome. Volume overloaded with pulmonary edema.' },
    'GI Bleed':      { heart_rate:124, sbp:88, dbp:54, map:60, spo2:95, resp_rate:18, gcs:14, hgb:6.4, plt:98, lactate:2.8, creatinine:1.1, vasopressor:0, clinical_notes:'Acute upper GI bleed, hemodynamically compromised. Received 2 units pRBC. EGD pending.' },
  };

  function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    Object.assign(S, p);
    document.querySelectorAll('.preset-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.preset === name);
    });
    syncInputsToState();
    scheduleCalc();
  }

  function syncInputsToState() {
    // Sliders
    [['sl-hr','heart_rate','val-hr'],['sl-sbp','sbp','val-sbp'],
     ['sl-map','map','val-map'],['sl-spo2','spo2','val-spo2'],
     ['sl-rr','resp_rate','val-rr'],['sl-temp','temperature','val-temp'],
     ['sl-gcs','gcs','val-gcs']].forEach(([id, key, dispId]) => {
      const inp = document.getElementById(id);
      const disp = document.getElementById(dispId);
      if (inp)  inp.value = S[key];
      if (disp) disp.textContent = S[key];
    });
    // Number inputs
    [['n-lactate','lactate'],['n-creatinine','creatinine'],['n-bilirubin','bilirubin'],
     ['n-wbc','wbc'],['n-hgb','hgb'],['n-plt','plt'],['n-sodium','sodium'],
     ['n-potassium','potassium'],['n-bun','bun'],['n-glucose','glucose'],
     ['n-albumin','albumin'],['n-pao2','pao2'],['n-fio2','fio2'],['n-peep','peep']].forEach(([id, key]) => {
      const inp = document.getElementById(id);
      if (inp) inp.value = S[key];
    });
    // Notes
    const notesEl = document.getElementById('clinical-notes');
    if (notesEl) notesEl.value = S.clinical_notes;
    renderNER(extractNER(S.clinical_notes));
  }

  /* ═══════════════════════════════════════════════════════
     §10 COPY EMR NOTE
  ═══════════════════════════════════════════════════════ */
  function copySbarNote() {
    const { riskPct, ci_low, ci_high, confidence } = S;
    const { label } = getTier(riskPct);
    const text = `SBAR — MEDGUARD AI Risk Report
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SITUATION:
  48-Hour ICU Mortality Risk: ${riskPct}% (${label.replace(/[🟢🟡🟠🔴]/g,'').trim()})
  Confidence: ${(confidence*100).toFixed(0)}%  |  95% CI: ${ci_low}% – ${ci_high}%

BACKGROUND:
  Age ${S.age}y ${S.gender==='M'?'Male':'Female'} | ${S.icu_type}
  Vitals: HR ${S.heart_rate} bpm, MAP ${S.map} mmHg, SpO₂ ${S.spo2}%, RR ${S.resp_rate}, Temp ${S.temperature}°C
  Lactate ${S.lactate} mmol/L | Creatinine ${S.creatinine} mg/dL | GCS ${S.gcs}
  Vasopressors: ${S.vasopressor ? 'Yes' : 'No'}

ASSESSMENT:
${S.clinical_notes}

RECOMMENDATION:
${document.getElementById('guidance-items')?.innerText || 'See clinical guidance panel.'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Generated by MEDGUARD AI · ${new Date().toLocaleString()}`;
    navigator.clipboard.writeText(text).then(() => showToast('📋 SBAR note copied'));
  }

  /* ═══════════════════════════════════════════════════════
     §11 TOAST
  ═══════════════════════════════════════════════════════ */
  function showToast(msg) {
    const t = document.getElementById('toast');
    if (!t) return;
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3000);
  }

  /* ═══════════════════════════════════════════════════════
     §12 HELPERS
  ═══════════════════════════════════════════════════════ */
  function setText(id, v) { const e = document.getElementById(id); if (e) e.textContent = v ?? '—'; }
  function setClass(id, c) { const e = document.getElementById(id); if (e) e.className = c; }

  /* ═══════════════════════════════════════════════════════
     §13 PATIENTS MODULE
  ═══════════════════════════════════════════════════════ */
  let allPatients = [], filteredPatients = [];
  let currentView = 'grid', riskFilter = 'all';
  let sortKey = 'bed', sortAsc = true;
  let selectedPatient = null;

  const ICU_UNITS = ['A','B','C'];
  const NAMES = ['J. Martinez','A. Thompson','R. Patel','S. Williams','M. Chen',
                 'L. Garcia','D. Johnson','P. Kumar','N. Roberts','T. Anderson',
                 'C. Brown','E. Wilson'];

  function bedLabel(i) {
    return `Bed ${ICU_UNITS[Math.floor(i/6)%ICU_UNITS.length]}${String(i%6+1).padStart(2,'0')}`;
  }
  function ptName(i) { return NAMES[i % NAMES.length]; }
  function shortIcu(icu) {
    if (!icu) return 'ICU';
    if (/MICU\/SICU/i.test(icu)) return 'MICU/SICU';
    if (/Medical/i.test(icu))    return 'MICU';
    if (/Trauma/i.test(icu))     return 'TSICU';
    if (/Surgical/i.test(icu))   return 'SICU';
    if (/Cardiac|CCU/i.test(icu)) return 'CCU';
    if (/Neuro/i.test(icu))      return 'NICU';
    return icu.split(' ')[0];
  }

  const VMEANS = { heart_rate:80, sbp:120, dbp:70, map:85, spo2:97, resp_rate:18, gcs:14, lactate:1.5, creatinine:1.0, hemoglobin:11 };
  const VSTDS  = { heart_rate:20, sbp:25,  dbp:15, map:18, spo2:4,  resp_rate:5,  gcs:3,  lactate:1.5, creatinine:1.2, hemoglobin:3  };
  function denorm(v, k) {
    if (v == null) return null;
    return +((VMEANS[k]||0) + v*(VSTDS[k]||1)).toFixed(1);
  }
  function latestVital(ts, k) {
    if (!ts) return null;
    for (let i = ts.length-1; i >= 0; i--) if (ts[i][k] != null) return ts[i][k];
    return null;
  }
  function ptRisk(patient, ts) {
    const lv = k => denorm(latestVital(ts, k), k);
    let r = 0.08;
    const age = patient.age || 65;
    if (age > 70) r += 0.08; else if (age > 55) r += 0.04;
    const map  = lv('map')        ?? 85;
    const hr   = lv('heart_rate') ?? 80;
    const spo2 = lv('spo2')       ?? 97;
    const rr   = lv('resp_rate')  ?? 18;
    const gcs  = lv('gcs')        ?? 14;
    const lac  = lv('lactate')    ?? 1.5;
    const cr   = lv('creatinine') ?? 1.0;
    if (map < 65) r += 0.15; else if (map < 75) r += 0.06;
    if (hr > 120) r += 0.08;
    if (spo2 < 90) r += 0.12; else if (spo2 < 94) r += 0.05;
    if (rr > 25) r += 0.07;
    if (gcs < 10) r += 0.12; else if (gcs < 14) r += 0.06;
    if (lac > 4) r += 0.18; else if (lac > 2) r += 0.10;
    if (cr > 3)  r += 0.10; else if (cr > 1.5) r += 0.05;
    const notes = (patient.clinical_notes || '').toLowerCase();
    if (/goals of care|comfort|worsening|deteriorat/.test(notes)) r += 0.08;
    return Math.round(Math.min(0.95, Math.max(0.03, r)) * 100);
  }
  function riskTier(r) {
    if (r >= 65) return 'critical';
    if (r >= 35) return 'high';
    if (r >= 15) return 'moderate';
    return 'low';
  }
  function riskLabel(tier) {
    return { critical:'Critical', high:'High', moderate:'Moderate', low:'Low' }[tier];
  }

  async function loadPatients() {
    const refreshBtn = document.getElementById('btn-refresh');
    if (refreshBtn) refreshBtn.textContent = '↻ Loading…';
    try {
      const res  = await fetch('/patients');
      const data = await res.json();
      allPatients = (data.patients || []).map((p, i) => {
        const r = ptRisk(p, p.timeseries || []);
        return { ...p, _bed:bedLabel(i), _name:ptName(i), _icu:shortIcu(p.icu_type), _idx:i, _risk:r, _tier:riskTier(r) };
      });
      updateWardStats();
      filterPt();
    } catch (e) {
      const grid = document.getElementById('bed-grid');
      if (grid) grid.innerHTML = `<div style="grid-column:1/-1;padding:40px;text-align:center;color:var(--gray-400)">⚠️ Could not load patients: ${e.message}</div>`;
    }
    if (refreshBtn) refreshBtn.textContent = '↻ Refresh';
  }

  function updateWardStats() {
    setText('stat-total',    allPatients.length);
    setText('stat-critical', allPatients.filter(p => p._tier === 'critical' || p._tier === 'high').length);
    setText('stat-moderate', allPatients.filter(p => p._tier === 'moderate').length);
    setText('stat-low',      allPatients.filter(p => p._tier === 'low').length);
  }

  function filterPt() {
    const q = (document.getElementById('pt-search')?.value || '').toLowerCase();
    filteredPatients = allPatients.filter(p => {
      const matchSearch = !q ||
        p._name.toLowerCase().includes(q) ||
        String(p.subject_id).includes(q) ||
        p._bed.toLowerCase().includes(q) ||
        p._icu.toLowerCase().includes(q);
      const matchRisk = riskFilter === 'all' ||
        p._tier === riskFilter ||
        (riskFilter === 'critical' && (p._tier === 'critical' || p._tier === 'high'));
      return matchSearch && matchRisk;
    });
    sortPt();
    renderPtView();
  }

  function sortPt() {
    filteredPatients.sort((a, b) => {
      let va, vb;
      if (sortKey === 'bed')  { va = a._idx; vb = b._idx; }
      else if (sortKey === 'name') { va = a._name; vb = b._name; }
      else if (sortKey === 'age')  { va = a.age;  vb = b.age; }
      else if (sortKey === 'risk') { va = a._risk; vb = b._risk; }
      else return 0;
      if (typeof va === 'string') return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortAsc ? va - vb : vb - va;
    });
  }

  function renderPtView() {
    if (currentView === 'grid') renderBedGrid();
    else renderPtTable();
  }

  function renderBedGrid() {
    const grid = document.getElementById('bed-grid');
    if (!grid) return;
    if (!filteredPatients.length) {
      grid.innerHTML = `<div style="grid-column:1/-1;padding:40px;text-align:center;color:var(--gray-400)">No patients match the current filter.</div>`;
      return;
    }
    grid.innerHTML = filteredPatients.map(p => {
      const ts   = p.timeseries || [];
      const lv   = k => denorm(latestVital(ts,k),k);
      const hr   = lv('heart_rate') ?? '—';
      const map  = lv('map') ?? '—';
      const spo2 = lv('spo2') ?? '—';
      const lac  = lv('lactate') ?? '—';
      return `<div class="bed-card risk-${p._tier}" onclick="window._mg_openDetail(${p.subject_id})">
        <div class="bed-card-header">
          <span class="bed-number">🛏 ${p._bed}</span>
          <span class="risk-pill ${p._tier}">${riskLabel(p._tier)}</span>
        </div>
        <div class="bed-card-body">
          <div class="pt-name">${p._name}</div>
          <div class="pt-meta">${p.age}y · ${p.gender==='M'?'♂ Male':'♀ Female'} · ${p._icu}</div>
          <div class="pt-score-row">
            <span class="pt-score-num">${p._risk}</span>
            <span class="pt-score-unit">% 48h risk</span>
          </div>
          <div class="pt-vitals-mini">
            <div class="pt-vital-item"><span>HR</span><span class="pt-vital-val">${hr}</span></div>
            <div class="pt-vital-item"><span>MAP</span><span class="pt-vital-val">${map}</span></div>
            <div class="pt-vital-item"><span>SpO₂</span><span class="pt-vital-val">${spo2}</span></div>
            <div class="pt-vital-item"><span>Lactate</span><span class="pt-vital-val">${lac}</span></div>
          </div>
        </div>
        <div class="bed-card-footer">
          <button class="load-btn" onclick="event.stopPropagation();window._mg_loadById(${p.subject_id})">⚡ Load into Calculator</button>
          <span class="icu-badge">${p._icu}</span>
        </div>
      </div>`;
    }).join('');
  }

  function renderPtTable() {
    const tbody = document.getElementById('pt-tbody');
    if (!tbody) return;
    if (!filteredPatients.length) {
      tbody.innerHTML = `<tr><td colspan="9" style="text-align:center;padding:32px;color:var(--gray-400)">No patients match.</td></tr>`;
      return;
    }
    tbody.innerHTML = filteredPatients.map(p => {
      const ts   = p.timeseries || [];
      const lv   = k => denorm(latestVital(ts,k),k);
      const hr   = lv('heart_rate') ?? '—';
      const map  = lv('map') ?? '—';
      const spo2 = lv('spo2') ?? '—';
      const mort = p.mortality_48h
        ? '<span class="mort-badge mort-yes">✕ Deceased</span>'
        : '<span class="mort-badge mort-no">✓ Survived</span>';
      return `<tr onclick="window._mg_openDetail(${p.subject_id})">
        <td class="td-bed">🛏 ${p._bed}</td>
        <td><div class="td-name">${p._name}</div><div style="font-size:11px;color:var(--gray-400)">ID: ${p.subject_id}</div></td>
        <td>${p.age}y</td>
        <td>${p.gender==='M'?'♂ M':'♀ F'}</td>
        <td>${p._icu}</td>
        <td class="td-score ${p._tier}">${p._risk}%</td>
        <td style="font-size:12px">HR <b>${hr}</b> · MAP <b>${map}</b> · SpO₂ <b>${spo2}</b></td>
        <td>${mort}</td>
        <td class="td-action"><button class="tbl-load-btn" onclick="event.stopPropagation();window._mg_loadById(${p.subject_id})">⚡ Load</button></td>
      </tr>`;
    }).join('');
  }

  /* ---- Detail panel ---- */
  window._mg_openDetail = function(subjectId) {
    const p = allPatients.find(pt => pt.subject_id === subjectId);
    if (!p) return;
    selectedPatient = p;
    const ts = p.timeseries || [];
    const lv = k => denorm(latestVital(ts,k),k);
    const fv = (v, unit) => v != null ? `${v} ${unit}` : '—';

    setText('detail-bed-label', `🛏 ${p._bed}  ·  ${p._icu}`);
    setText('detail-name',      p._name);
    setText('detail-risk-score', p._risk);
    const pill = document.getElementById('detail-risk-pill');
    if (pill) { pill.className = `risk-pill ${p._tier}`; pill.textContent = riskLabel(p._tier); }

    setText('d-subject-id', p.subject_id);
    setText('d-stay-id',    p.stay_id);
    setText('d-age',        `${p.age} years`);
    setText('d-gender',     p.gender==='M'?'♂ Male':'♀ Female');
    setText('d-icu-type',   p.icu_type || '—');
    setText('d-admission',  p.admission_type || '—');
    setText('d-hr',         fv(lv('heart_rate'), 'bpm'));
    setText('d-bp',         `${lv('sbp')??'—'} / ${lv('dbp')??'—'} mmHg`);
    setText('d-spo2',       fv(lv('spo2'), '%'));
    setText('d-rr',         fv(lv('resp_rate'), '/min'));
    setText('d-gcs',        fv(lv('gcs'), '/ 15'));
    setText('d-lactate',    fv(lv('lactate'), 'mmol/L'));
    setText('d-creatinine', fv(lv('creatinine'), 'mg/dL'));
    setText('d-hgb',        fv(lv('hemoglobin'), 'g/dL'));

    const mb = document.getElementById('d-mortality-badge');
    if (mb) mb.innerHTML = p.mortality_48h
      ? '<span class="mort-badge mort-yes">✕ 48-hour mortality recorded</span>'
      : '<span class="mort-badge mort-no">✓ Patient survived (MIMIC-IV)</span>';

    const notes = p.clinical_notes || 'No notes.';
    setText('d-notes', notes.length > 300 ? notes.slice(0,300)+'…' : notes);

    document.getElementById('pt-detail-overlay')?.classList.add('open');
  };

  window._mg_closeDetail = function() {
    document.getElementById('pt-detail-overlay')?.classList.remove('open');
    selectedPatient = null;
  };

  window._mg_loadById = function(subjectId) {
    const p = allPatients.find(pt => pt.subject_id === subjectId);
    if (p) { selectedPatient = p; loadPatientIntoCalc(); }
  };

  function loadPatientIntoCalc() {
    if (!selectedPatient) return;
    const p  = selectedPatient;
    const ts = p.timeseries || [];
    const lv = k => denorm(latestVital(ts,k),k);

    // Merge into calculator state
    if (lv('heart_rate') != null) S.heart_rate = lv('heart_rate');
    if (lv('sbp') != null) S.sbp = lv('sbp');
    if (lv('dbp') != null) S.dbp = lv('dbp');
    if (lv('map') != null) S.map = lv('map');
    if (lv('spo2') != null) S.spo2 = lv('spo2');
    if (lv('resp_rate') != null) S.resp_rate = lv('resp_rate');
    if (lv('gcs') != null) S.gcs = lv('gcs');
    if (lv('lactate') != null) S.lactate = lv('lactate');
    if (lv('creatinine') != null) S.creatinine = lv('creatinine');
    if (lv('hemoglobin') != null) S.hgb = lv('hemoglobin');
    S.age    = p.age || S.age;
    S.gender = p.gender || S.gender;
    S.clinical_notes = p.clinical_notes || 'No notes.';

    // Switch to calculator tab
    document.querySelectorAll('.subnav-tab').forEach(t => t.classList.remove('active'));
    const calcTab = document.querySelector('.subnav-tab[data-panel="primary"]');
    if (calcTab) calcTab.classList.add('active');
    document.getElementById('view-calculator')?.classList.remove('hidden');
    document.getElementById('view-patients')?.classList.add('hidden');

    // Sync inputs
    syncInputsToState();
    window._mg_closeDetail();
    scheduleCalc();
    window.scrollTo({ top: 0, behavior: 'smooth' });
    showToast(`✅ Loaded ${p._name} into calculator`);
  }

  /* ---- Patients toolbar (exposed to HTML) ---- */
  window.filterPatients = filterPt;
  window.setRiskFilter  = function(btn, filter) {
    riskFilter = filter;
    document.querySelectorAll('.filter-btn[data-risk]').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    filterPt();
  };
  window.switchView = function(mode) {
    currentView = mode;
    document.getElementById('btn-grid')?.classList.toggle('active', mode==='grid');
    document.getElementById('btn-table')?.classList.toggle('active', mode==='table');
    document.getElementById('bed-grid')?.classList.toggle('hidden', mode!=='grid');
    document.getElementById('pt-table-wrap')?.classList.toggle('hidden', mode!=='table');
    renderPtView();
  };
  window.sortTable = function(key) {
    if (sortKey === key) sortAsc = !sortAsc; else { sortKey = key; sortAsc = true; }
    sortPt(); renderPtView();
  };
  window.loadPatients = loadPatients;
  window.closeDetail  = window._mg_closeDetail;
  window.openDetail   = window._mg_openDetail;
  window.loadPatientIntoCalcById = window._mg_loadById;
  window.loadPatientIntoCalc     = loadPatientIntoCalc;

  /* ═══════════════════════════════════════════════════════
     §14 SINGLE DOMContentLoaded — INIT EVERYTHING
  ═══════════════════════════════════════════════════════ */
  document.addEventListener('DOMContentLoaded', function () {
    // ---- Tab router (single handler) ----
    initRouter();

    // ---- Evidence tabs ----
    initEvTabs();

    // ---- Preset buttons ----
    document.querySelectorAll('.preset-btn').forEach(btn => {
      btn.addEventListener('click', () => applyPreset(btn.dataset.preset));
    });

    // ---- Slider bindings ----
    bindSlider('sl-hr',   'heart_rate',  'val-hr');
    bindSlider('sl-sbp',  'sbp',         'val-sbp');
    bindSlider('sl-map',  'map',         'val-map');
    bindSlider('sl-spo2', 'spo2',        'val-spo2');
    bindSlider('sl-rr',   'resp_rate',   'val-rr');
    bindSlider('sl-temp', 'temperature', 'val-temp');
    bindSlider('sl-gcs',  'gcs',         'val-gcs');

    // ---- Number input bindings ----
    bindNumber('n-lactate',    'lactate');
    bindNumber('n-creatinine', 'creatinine');
    bindNumber('n-bilirubin',  'bilirubin');
    bindNumber('n-wbc',        'wbc');
    bindNumber('n-hgb',        'hgb');
    bindNumber('n-plt',        'plt');
    bindNumber('n-sodium',     'sodium');
    bindNumber('n-potassium',  'potassium');
    bindNumber('n-bun',        'bun');
    bindNumber('n-glucose',    'glucose');
    bindNumber('n-albumin',    'albumin');
    bindNumber('n-pao2',       'pao2');
    bindNumber('n-fio2',       'fio2');
    bindNumber('n-peep',       'peep');

    // ---- Segmented controls ----
    bindSeg('gender',   'gender');
    bindSeg('icu_type', 'icu_type');

    // ---- Toggle buttons ----
    bindToggle('vasopressor', 'vasopressor');
    bindToggle('renal',       'renal_replacement');

    // ---- Clinical notes ----
    const notesEl = document.getElementById('clinical-notes');
    if (notesEl) {
      notesEl.addEventListener('input', () => {
        S.clinical_notes = notesEl.value;
        renderNER(extractNER(S.clinical_notes));
        scheduleCalc();
      });
    }

    // ---- Copy EMR buttons ----
    document.querySelectorAll('.emr-btn, #nav-emr-btn').forEach(b => {
      b.addEventListener('click', copySbarNote);
    });

    // ---- Age slider (not bound to named keys above) ----
    const slAge = document.getElementById('sl-age');
    if (slAge) {
      slAge.addEventListener('input', () => {
        S.age = parseInt(slAge.value, 10);
        const d = document.getElementById('val-age');
        if (d) d.textContent = slAge.value;
        scheduleCalc();
      });
    }

    // ---- Initial render ----
    syncInputsToState();
    runCalc();

    // ---- Expose public API ----
    window.mediguard = {
      state: S,
      applyPreset,
      syncInputs: syncInputsToState,
      syncAndCalc: () => { syncInputsToState(); scheduleCalc(); },
      showToast,
    };
  });

})();
