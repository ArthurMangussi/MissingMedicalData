'use strict';

const CFG = window.MEDINPAINTER;

// ── State ────────────────────────────────────────────────────────────────────
const state = {
  rating:     0,
  lastResult: null,
  imgData:    {},   // { original, corrupted, mask, imputed } as base64
  locked:     false,
};

const RATING_LABELS = ['', 'Poor', 'Fair', 'Good', 'Very Good', 'Excellent'];

// ── DOM shortcuts ─────────────────────────────────────────────────────────────
const $ = id => document.getElementById(id);
const datasetSel    = $('dataset-select');
const mechanismSel  = $('mechanism-select');
const algorithmSel  = $('algorithm-select');
const paramsDiv     = $('params-container');
const datasetDesc   = $('dataset-desc');
const mechanismDesc = $('mechanism-desc');
const algorithmDesc = $('algorithm-desc');
const seedInput     = $('seed-input');
const prevSeedBtn   = $('prev-seed');
const nextSeedBtn   = $('next-seed');
const seedLabel     = $('seed-label');
const generateBtn   = $('generate-btn');
const metricsBar    = $('metrics-bar');
const starRating    = $('star-rating');
const ratingLabel   = $('rating-label');
const notesArea     = $('feedback-notes');
const saveBtn       = $('save-feedback-btn');
const feedbackToast = $('feedback-toast');
const historyEl     = $('feedback-history');
const historyBadge  = $('history-badge');
const exportBtn     = $('export-btn');
const countBadge    = $('feedback-count-badge');
const algoBadge     = $('algo-badge');

const stars = [...starRating.querySelectorAll('.star')];

// ── Init ─────────────────────────────────────────────────────────────────────
function init() {
  syncDatasetDesc();
  syncMechanismUI();
  syncAlgorithmDesc();
  loadHistory();

  datasetSel.addEventListener('change',   syncDatasetDesc);
  mechanismSel.addEventListener('change', syncMechanismUI);
  algorithmSel.addEventListener('change', syncAlgorithmDesc);

  seedInput.addEventListener('input', () => {
    if (+seedInput.value < 0) seedInput.value = 0;
  });
  prevSeedBtn.addEventListener('click', () => {
    seedInput.value = Math.max(0, +seedInput.value - 1);
  });
  nextSeedBtn.addEventListener('click', () => {
    seedInput.value = +seedInput.value + 1;
  });

  generateBtn.addEventListener('click', generate);
  saveBtn.addEventListener('click', saveFeedback);
  exportBtn.addEventListener('click', () => {
    window.location.href = '/api/feedback/export';
  });

  // Star rating
  stars.forEach(star => {
    star.addEventListener('mouseenter', () => highlightStars(+star.dataset.value));
    star.addEventListener('click',      () => selectRating(+star.dataset.value));
  });
  starRating.addEventListener('mouseleave', () => highlightStars(state.rating));

  // Image download buttons
  document.querySelectorAll('.card-download').forEach(btn => {
    btn.addEventListener('click', () => downloadImage(btn.dataset.card));
  });
}

// ── Sync UI descriptions ──────────────────────────────────────────────────────
function syncDatasetDesc() {
  const d = CFG.datasets.find(x => x.value === datasetSel.value);
  datasetDesc.textContent = d?.description ?? '';
}

function syncMechanismUI() {
  const opt  = mechanismSel.options[mechanismSel.selectedIndex];
  const mech = CFG.mechanisms.find(m => m.value === opt.value);
  mechanismDesc.textContent = opt.dataset.desc ?? '';
  renderParams(mech?.params ?? []);
}

function syncAlgorithmDesc() {
  const opt  = algorithmSel.options[algorithmSel.selectedIndex];
  const algo = CFG.algorithms.find(a => a.value === opt.value);
  algorithmDesc.textContent = algo?.description ?? '';
}

// ── Parameter sliders ─────────────────────────────────────────────────────────
const paramValues = {};

function renderParams(params) {
  paramsDiv.innerHTML = '';
  Object.keys(paramValues).forEach(k => delete paramValues[k]);

  params.forEach(p => {
    paramValues[p.id] = p.default;

    const row = document.createElement('div');
    row.className = 'param-row';
    row.innerHTML = `
      <div class="param-header">
        <span class="param-label">${p.label}</span>
        <span class="param-value" id="pv-${p.id}">${p.default}</span>
      </div>
      <input type="range" class="param-slider" id="ps-${p.id}"
        min="${p.min}" max="${p.max}" step="${p.step}" value="${p.default}">
    `;
    paramsDiv.appendChild(row);

    const slider  = row.querySelector(`#ps-${p.id}`);
    const display = row.querySelector(`#pv-${p.id}`);
    slider.addEventListener('input', () => {
      paramValues[p.id] = +slider.value;
      display.textContent = slider.value;
    });
  });
}

// ── Generate ──────────────────────────────────────────────────────────────────
function resetBtn() {
  const txt = generateBtn.querySelector('.btn-text');
  const spn = generateBtn.querySelector('.btn-spinner');
  if (txt) txt.hidden = false;
  if (spn) spn.hidden = true;
  generateBtn.disabled = false;
  state.locked = false;
}

async function generate() {
  if (state.locked) return;
  state.locked = true;

  const txt = generateBtn.querySelector('.btn-text');
  const spn = generateBtn.querySelector('.btn-spinner');
  if (txt) txt.hidden = true;
  if (spn) spn.hidden = false;
  generateBtn.disabled = true;

  try {
    const res = await fetch('/api/process', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        dataset:   datasetSel.value,
        mechanism: mechanismSel.value,
        algorithm: algorithmSel.value,
        params:    paramValues,
        seed:      +seedInput.value,
      }),
    });

    const data = await res.json().catch(() => null);

    if (!res.ok || !data || data.error) {
      console.error('[MedInpainter] Server error:', data?.error ?? res.status);
      return;
    }

    state.lastResult = data;
    state.imgData = { original: data.original, corrupted: data.corrupted, mask: data.mask, imputed: data.imputed };
    renderImages(data);
    renderMetrics(data);
    updateSeedLabel(data);
    activateFeedback();
    updateAlgoBadge();
  } catch (err) {
    console.error('[MedInpainter] Generate error:', err);
  } finally {
    resetBtn();
  }
}

function setCardImage(cardId, b64) {
  const body = document.querySelector(`#${cardId} .card-body`);
  body.innerHTML = `<img src="data:image/png;base64,${b64}" alt="${cardId}">`;
  document.querySelector(`#${cardId} .card-download`).hidden = false;
}

function renderImages(data) {
  setCardImage('card-original',  data.original);
  setCardImage('card-corrupted', data.corrupted);
  setCardImage('card-mask',      data.mask);
  setCardImage('card-imputed',   data.imputed);
}

function renderMetrics(data) {
  metricsBar.hidden = false;
  $('m-missing').textContent = `${data.missing_pct}%`;
  $('m-psnr').textContent    = data.metrics?.psnr  != null ? data.metrics.psnr  : '—';
  $('m-ssim').textContent    = data.metrics?.ssim  != null ? data.metrics.ssim  : '—';
  $('m-mae').textContent     = data.metrics?.mae   != null ? data.metrics.mae   : '—';
}

function updateSeedLabel(data) {
  if (!seedLabel) return;
  if (data.n_images) {
    const shown = (data.image_idx ?? +seedInput.value) + 1;
    seedLabel.textContent = `Image ${shown} / ${data.n_images}`;
  } else {
    seedLabel.textContent = 'Synthetic (dataset unavailable)';
  }
}

function updateAlgoBadge() {
  const algo = CFG.algorithms.find(a => a.value === algorithmSel.value);
  algoBadge.textContent = algo?.label ?? '';
}

function downloadImage(cardKey) {
  const b64 = state.imgData[cardKey];
  if (!b64) return;
  const a = document.createElement('a');
  a.href     = `data:image/png;base64,${b64}`;
  a.download = `medinpainter_${cardKey}.png`;
  a.click();
}

// ── Star rating ───────────────────────────────────────────────────────────────
function highlightStars(upTo) {
  stars.forEach(s => s.classList.toggle('lit', +s.dataset.value <= upTo && upTo > 0));
}

function selectRating(value) {
  state.rating = value;
  stars.forEach(s => s.classList.toggle('lit', +s.dataset.value <= value));
  ratingLabel.textContent = RATING_LABELS[value] ?? '';
  updateSaveBtn();
}

function activateFeedback() {
  state.rating = 0;
  stars.forEach(s => { s.classList.remove('lit'); s.disabled = false; });
  ratingLabel.textContent = 'Select a rating';
  notesArea.value         = '';
  notesArea.disabled      = false;
  feedbackToast.hidden    = true;
  updateSaveBtn();
}

function updateSaveBtn() {
  saveBtn.disabled = !(state.rating > 0 && state.lastResult);
}

// ── Save feedback ─────────────────────────────────────────────────────────────
async function saveFeedback() {
  if (!state.rating || !state.lastResult) return;

  await fetch('/api/feedback', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      dataset:   datasetSel.value,
      mechanism: mechanismSel.value,
      algorithm: algorithmSel.value,
      rating:    state.rating,
      notes:     notesArea.value.trim(),
      metrics:   state.lastResult.metrics ?? {},
      seed:      +seedInput.value,
    }),
  });

  feedbackToast.hidden = false;
  setTimeout(() => { feedbackToast.hidden = true; }, 3500);

  // Disable stars / textarea after save to prevent double-submit
  stars.forEach(s => s.disabled = true);
  notesArea.disabled  = true;
  saveBtn.disabled    = true;

  loadHistory();
}

// ── Feedback history ──────────────────────────────────────────────────────────
async function loadHistory() {
  const res  = await fetch('/api/feedback');
  const data = await res.json();

  const count = data.length;
  historyBadge.textContent     = count;
  countBadge.textContent       = `${count} rating${count !== 1 ? 's' : ''}`;
  exportBtn.hidden              = count === 0;

  const recent = data.slice().reverse().slice(0, 10);
  renderHistory(recent);
}

function renderHistory(entries) {
  if (!entries.length) {
    historyEl.innerHTML = '<div class="empty-state-small">No feedback saved yet</div>';
    return;
  }

  historyEl.innerHTML = entries.map(e => {
    const ds   = CFG.datasets.find(d => d.value === e.dataset)?.label  ?? e.dataset;
    const algo = CFG.algorithms.find(a => a.value === e.algorithm)?.label ?? e.algorithm;
    const mech = CFG.mechanisms.find(m => m.value === e.mechanism)?.label ?? e.mechanism;
    const filled = '★'.repeat(e.rating);
    const empty  = '☆'.repeat(5 - e.rating);
    const metrics = e.metrics && Object.keys(e.metrics).length
      ? `PSNR ${e.metrics.psnr ?? '—'} · SSIM ${e.metrics.ssim ?? '—'}`
      : '';
    const notes = e.notes
      ? `<div class="history-notes">&ldquo;${escHtml(e.notes)}&rdquo;</div>`
      : '';

    return `
      <div class="history-item" data-id="${e.id}">
        <button class="history-delete" data-id="${e.id}" title="Delete">&#10005;</button>
        <div class="history-stars">${filled}${empty}</div>
        <div class="history-meta">
          <strong>${escHtml(ds)}</strong> &middot; ${escHtml(algo)}<br>
          <span>${escHtml(mech)}</span><br>
          <span>${e.ts ?? ''}</span>
        </div>
        ${metrics ? `<div class="history-metrics">${metrics}</div>` : ''}
        ${notes}
      </div>
    `;
  }).join('');

  historyEl.querySelectorAll('.history-delete').forEach(btn => {
    btn.addEventListener('click', async () => {
      await fetch(`/api/feedback/${btn.dataset.id}`, { method: 'DELETE' });
      loadHistory();
    });
  });
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ── Start ─────────────────────────────────────────────────────────────────────
init();
