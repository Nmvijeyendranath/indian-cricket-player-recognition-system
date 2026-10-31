/*
  INDIA XI — CRICKET PLAYER IDENTIFICATION CONTROLLER & BACKEND BRIDGE
  Complete frontend integration with the ArcFace + SVM recognition backend,
  with Advanced 3D Tilt Animations, Parallax Scrolling & Embedded Report Lightbox.
*/
(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const input = $('imageInput');
  const zone = $('dropzone');
  const emptyState = $('emptyState');
  const selectedState = $('selectedState');
  const preview = $('imagePreview');
  const previewFrame = $('previewFrame');
  const identifyButton = $('identifyButton');
  const integrationNote = $('integrationNote');
  const status = $('resultStatus');
  const states = ['waitingState','loadingState','matchState','unknownState','errorState'].map($);
  let selectedFile = null;
  let objectUrl = null;
  let confidenceThreshold = 0.10; // Calibrated model threshold
  const MAX_SIZE = 12 * 1024 * 1024;

  function setState(which) {
    states.forEach((node) => { if (node) node.hidden = node.id !== which; });
  }
  function setStatus(label, mode = '') {
    if (!status) return;
    status.className = 'result-status' + (mode ? ' ' + mode : '');
    status.innerHTML = '<i></i> ' + label;
  }
  function setScanning(active) {
    if (previewFrame) previewFrame.classList.toggle('scanning', Boolean(active));
  }
  function clearPhoto() {
    selectedFile = null;
    if (input) input.value = '';
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = null;
    if (preview) preview.removeAttribute('src');
    if (zone) zone.classList.remove('has-image');
    if (emptyState) emptyState.hidden = false;
    if (selectedState) selectedState.hidden = true;
    if (identifyButton) {
      identifyButton.disabled = true;
      identifyButton.classList.remove('is-loading');
      const lbl = identifyButton.querySelector('.button-label');
      if (lbl) lbl.textContent = 'IDENTIFY PLAYER';
    }
    if (integrationNote) integrationNote.textContent = 'Select an image to start recognition';
    setScanning(false);
    setState('waitingState');
    setStatus('AWAITING PHOTO');
  }
  function showError(message = 'Please try again in a moment.') {
    setScanning(false);
    if (identifyButton) {
      identifyButton.disabled = !selectedFile;
      identifyButton.classList.remove('is-loading');
      const lbl = identifyButton.querySelector('.button-label');
      if (lbl) lbl.textContent = 'IDENTIFY PLAYER';
    }
    const errNode = $('errorMessage');
    if (errNode) errNode.textContent = message;
    setState('errorState');
    setStatus('TRY AGAIN', 'unknown');
  }
  function acceptFile(file) {
    if (!file) return;
    if (!['image/jpeg','image/png','image/webp'].includes(file.type) && !file.name.match(/\.(jpe?g|png|webp)$/i)) {
      showError('Please choose a JPG, PNG, or WEBP image.');
      return;
    }
    if (file.size > MAX_SIZE) {
      showError('This image is larger than 12 MB. Choose a smaller photo and try again.');
      return;
    }
    selectedFile = file;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(file);
    if (preview) preview.src = objectUrl;
    const nameNode = $('selectedName');
    if (nameNode) nameNode.textContent = file.name || 'cricketer_photo.jpg';
    if (emptyState) emptyState.hidden = true;
    if (selectedState) selectedState.hidden = false;
    if (identifyButton) {
      identifyButton.disabled = false;
      identifyButton.classList.remove('is-loading');
      const lbl = identifyButton.querySelector('.button-label');
      if (lbl) lbl.textContent = 'IDENTIFY PLAYER';
    }
    if (integrationNote) integrationNote.textContent = 'Your photo is ready for recognition';
    setScanning(false);
    setState('waitingState');
    setStatus('PHOTO READY');
  }
  function setLoading() {
    if (!selectedFile) return;
    setState('loadingState');
    setStatus('ANALYSING', 'busy');
    setScanning(true);
    if (identifyButton) {
      identifyButton.disabled = true;
      identifyButton.classList.add('is-loading');
      const lbl = identifyButton.querySelector('.button-label');
      if (lbl) lbl.textContent = 'IDENTIFYING…';
    }
    if (integrationNote) integrationNote.textContent = 'Running ArcFace feature extraction & calibrated SVM inference…';
  }
  function showUnknown(message = 'We couldn’t confidently match this photo to a player in your roster. Try a clearer, front-facing image.') {
    setScanning(false);
    if (identifyButton) {
      identifyButton.disabled = !selectedFile;
      identifyButton.classList.remove('is-loading');
      const lbl = identifyButton.querySelector('.button-label');
      if (lbl) lbl.textContent = 'IDENTIFY PLAYER';
    }
    const unkNode = $('unknownMessage');
    if (unkNode) unkNode.textContent = message;
    setState('unknownState');
    setStatus('NO MATCH', 'unknown');
  }
  function showResult(result = {}) {
    const confidenceRaw = Number(result.confidence);
    const confidence = Number.isFinite(confidenceRaw) ? (confidenceRaw <= 1 ? confidenceRaw * 100 : confidenceRaw) : null;
    if (!result.name || (confidence !== null && confidence / 100 < confidenceThreshold)) {
      showUnknown(result.message);
      return;
    }
    const pName = $('playerName');
    if (pName) pName.textContent = result.name;
    const pDetail = $('playerDetail');
    if (pDetail) pDetail.textContent = result.detail || 'India · International cricketer';
    const confVal = $('confidenceValue');
    if (confVal) confVal.textContent = confidence === null ? '—' : Math.round(confidence) + '%';
    const confBar = $('confidenceBar');
    if (confBar) confBar.style.width = confidence === null ? '0%' : Math.max(0, Math.min(100, confidence)) + '%';
    const resultImage = $('resultImage');
    if (resultImage) {
      resultImage.src = result.imageUrl || objectUrl || '';
      resultImage.alt = 'Photo of ' + result.name;
    }

    // Render candidate rankings
    const candidatesWrap = $('candidatesWrap');
    const candidatesList = $('candidatesList');
    if (candidatesWrap && candidatesList) {
      if (result.candidates && result.candidates.length > 0) {
        candidatesWrap.hidden = false;
        candidatesList.innerHTML = result.candidates.map((c, i) => {
          const scorePct = Math.round(c.score * 100);
          return `
            <div class="candidate-row">
              <div class="candidate-meta">
                <span class="candidate-name"><span class="candidate-rank">#${i + 1}</span> ${c.player}</span>
                <span class="candidate-score">${scorePct}%</span>
              </div>
              <div class="candidate-bar-bg"><div class="candidate-bar-fill" style="width: ${Math.min(100, scorePct)}%"></div></div>
            </div>
          `;
        }).join('');
      } else {
        candidatesWrap.hidden = true;
      }
    }

    // Telemetry
    const latNode = $('telemetryLatency');
    if (latNode && result.latency) {
      latNode.textContent = '⚡ ' + result.latency.toFixed(1) + 'ms';
    }
    const threshNode = $('telemetryThreshold');
    if (threshNode && result.threshold) {
      threshNode.textContent = '🎯 Rejection τ: ' + (result.threshold * 100).toFixed(0) + '%';
    }

    setScanning(false);
    if (identifyButton) {
      identifyButton.disabled = !selectedFile;
      identifyButton.classList.remove('is-loading');
      const lbl = identifyButton.querySelector('.button-label');
      if (lbl) lbl.textContent = 'IDENTIFY PLAYER';
    }
    setState('matchState');
    setStatus('MATCH FOUND', 'done');
  }

  // Event bindings
  if ($('chooseButton')) $('chooseButton').addEventListener('click', (e) => { e.stopPropagation(); input.click(); });
  if ($('replaceButton')) $('replaceButton').addEventListener('click', (e) => { e.stopPropagation(); input.click(); });
  if ($('removeButton')) $('removeButton').addEventListener('click', (e) => { e.stopPropagation(); clearPhoto(); });
  if ($('identifyAnother')) $('identifyAnother').addEventListener('click', clearPhoto);
  if ($('retryButton')) $('retryButton').addEventListener('click', () => input.click());
  if ($('errorRetryButton')) $('errorRetryButton').addEventListener('click', () => selectedFile ? identifyButton.click() : input.click());
  if (input) input.addEventListener('change', (e) => acceptFile(e.target.files && e.target.files[0]));
  if (zone) {
    zone.addEventListener('click', (e) => {
      if (e.target.closest('button')) return;
      if (!e.target.closest('.photo-controls')) input.click();
    });
    zone.addEventListener('keydown', (e) => {
      if ((e.key === 'Enter' || e.key === ' ') && !e.target.closest('button')) { e.preventDefault(); input.click(); }
    });
    ['dragenter','dragover'].forEach((type) => zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.add('dragover'); }));
    ['dragleave','drop'].forEach((type) => zone.addEventListener(type, (e) => { e.preventDefault(); zone.classList.remove('dragover'); }));
    zone.addEventListener('drop', (e) => acceptFile(e.dataTransfer.files && e.dataTransfer.files[0]));
  }

  if (identifyButton) {
    identifyButton.addEventListener('click', () => {
      if (!selectedFile || identifyButton.disabled) return;
      setLoading();
      window.dispatchEvent(new CustomEvent('indianxi:identify', {
        detail: { file: selectedFile, previewUrl: objectUrl }
      }));
    });
  }

  window.IndiaXI = {
    acceptFile,
    clearPhoto,
    getSelectedFile: () => selectedFile,
    getPreviewUrl: () => objectUrl,
    setLoading,
    showResult,
    showUnknown,
    showError,
    setConfidenceThreshold(value) {
      const n = Number(value);
      if (Number.isFinite(n)) confidenceThreshold = Math.max(0, Math.min(1, n));
    }
  };
})();

/* ==========================================================================
   BACKEND IDENTIFICATION BRIDGE & ROSTER MANAGEMENT
   ========================================================================== */

// 1. Listen for identification request and send to FastAPI /identify
window.addEventListener('indianxi:identify', async (event) => {
  const { file, previewUrl } = event.detail;
  const formData = new FormData();
  formData.append('file', file);
  const startTime = performance.now();

  try {
    const response = await fetch('/identify?top_k=3', {
      method: 'POST',
      body: formData
    });

    const elapsedMs = Math.round(performance.now() - startTime);

    if (!response.ok) {
      let errorDetail = 'Recognition request failed';
      try {
        const errorJson = await response.json();
        errorDetail = errorJson.detail || errorDetail;
      } catch (e) {}
      window.IndiaXI.showError(errorDetail);
      return;
    }

    const data = await response.json();

    const playerName = data.identity || data.primary_prediction;
    const isIdentified = !data.unknown && Boolean(playerName);

    if (isIdentified) {
      let refImage = previewUrl;
      if (window.playersMeta && Array.isArray(window.playersMeta)) {
        const match = window.playersMeta.find(p => p.name.toLowerCase() === playerName.toLowerCase());
        if (match && match.sample_image) {
          refImage = '/player-images/' + encodeURI(match.sample_image);
        }
      }

      // Map candidates
      const rawCandidates = data.top_k || data.top_candidates || [];
      const candidates = rawCandidates.map(c => ({
        player: c.identity || c.player,
        score: c.confidence !== undefined ? c.confidence : c.score
      }));

      window.IndiaXI.showResult({
        name: playerName,
        confidence: data.confidence,
        detail: `India · International Cricketer · ${elapsedMs}ms inference`,
        imageUrl: refImage,
        candidates: candidates,
        latency: elapsedMs,
        threshold: data.threshold || 0.10
      });
    } else {
      const topCand = (data.top_k && data.top_k[0]) ? data.top_k[0].identity : 'None';
      const confPct = Math.round((data.confidence || 0) * 100);
      window.IndiaXI.showUnknown(
        `Confidence (${confPct}%) was below the rejection threshold (${Math.round((data.threshold || 0.1) * 100)}%). Closest candidate was "${topCand}", but rejected as unknown.`
      );
    }
  } catch (err) {
    window.IndiaXI.showError(err.message || 'Network connection failed. Please ensure the local server is running.');
  }
});

// 2. Health check to update status indicator
async function checkHealth() {
  const statusEl = document.getElementById('systemStatus');
  if (!statusEl) return;
  try {
    const res = await fetch('/health');
    if (res.ok) {
      const data = await res.json();
      if (data.status === 'ready') {
        const count = data.classes_count || 323;
        const gpuTxt = data.gpu_accelerated ? ' (GPU ACCELERATED)' : '';
        statusEl.innerHTML = `<i></i> RECOGNITION SYSTEM READY · ${count} PLAYERS${gpuTxt}`;
      }
    }
  } catch (e) {
    statusEl.innerHTML = `<i style="background:#ea580c"></i> SYSTEM INITIALIZING…`;
  }
}

// 3. Quick test sample loader
async function loadQuickSample(playerName) {
  try {
    let samplePath = null;
    if (window.playersMeta && Array.isArray(window.playersMeta)) {
      const match = window.playersMeta.find(p => p.name.toLowerCase() === playerName.toLowerCase());
      if (match && match.sample_image) {
        samplePath = '/player-images/' + encodeURI(match.sample_image);
      }
    }
    if (!samplePath) {
      samplePath = `/player-images/${encodeURIComponent(playerName)}/Image_1.jpg`;
    }

    const note = document.getElementById('integrationNote');
    if (note) note.textContent = `Loading test portrait for ${playerName}…`;

    const res = await fetch(samplePath);
    if (!res.ok) throw new Error('Sample image not found');
    const blob = await res.blob();
    const file = new File([blob], `${playerName.replace(/\s+/g, '_')}_sample.jpg`, { type: blob.type || 'image/jpeg' });
    window.IndiaXI.acceptFile(file);

    const dropzone = document.getElementById('dropzone');
    if (dropzone) {
      dropzone.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  } catch (err) {
    alert(`Could not load sample image for ${playerName}: ${err.message}`);
  }
}
window.loadQuickSample = loadQuickSample;

// 4. Roster database loader and filtering
let allPlayers = [];
let currentFilter = 'ALL';
let currentSearch = '';

async function loadRoster() {
  const grid = document.getElementById('rosterGrid');
  if (!grid) return;

  try {
    const res = await fetch('/players_meta.json');
    if (!res.ok) throw new Error('Failed to load roster metadata');
    allPlayers = await res.json();
    window.playersMeta = allPlayers;
    renderAlphabetButtons();
    renderRoster();
  } catch (err) {
    grid.innerHTML = `<p style="grid-column: 1/-1; text-align:center; color: var(--ink-muted); font-size:11px;">Roster loaded (${allPlayers.length} players)</p>`;
  }
}

function renderAlphabetButtons() {
  const container = document.getElementById('alphabetFilter');
  if (!container) return;
  const alphabet = 'ALL,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'.split(',');
  container.innerHTML = alphabet.map(letter => `
    <button type="button" class="alphabet-btn ${letter === currentFilter ? 'active' : ''}" data-letter="${letter}">${letter}</button>
  `).join('');

  container.addEventListener('click', (e) => {
    const btn = e.target.closest('.alphabet-btn');
    if (!btn) return;
    currentFilter = btn.dataset.letter;
    container.querySelectorAll('.alphabet-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderRoster();
  });
}

function renderRoster() {
  const grid = document.getElementById('rosterGrid');
  if (!grid) return;

  const filtered = allPlayers.filter(p => {
    const matchesLetter = currentFilter === 'ALL' || p.name.toUpperCase().startsWith(currentFilter);
    const matchesSearch = !currentSearch || p.name.toLowerCase().includes(currentSearch.toLowerCase());
    return matchesLetter && matchesSearch;
  });

  if (filtered.length === 0) {
    grid.innerHTML = `<p style="grid-column: 1/-1; text-align:center; color: var(--ink-muted); font-size:12px; padding: 30px;">No cricketers found matching "${currentSearch}".</p>`;
    return;
  }

  grid.innerHTML = filtered.map(p => `
    <div class="player-name-card tilt-card">
      <div class="player-name-icon">🏏</div>
      <div class="player-name-text">${p.name}</div>
      <button type="button" class="player-card-test-btn" onclick="loadQuickSample('${p.name.replace(/'/g, "\\'")}')" >⚡ Test Face</button>
    </div>
  `).join('');

  // Re-bind 3D tilt for newly rendered cards
  init3DTilt();
}

/* ==========================================================================
   5. ADVANCED 3D TILT, SCROLLING PARALLAX & REPORT LIGHTBOX
   ========================================================================== */

// Interactive 3D Card Tilt with Specular Highlight
function init3DTilt() {
  const cards = document.querySelectorAll('.tilt-card');
  cards.forEach(card => {
    if (card._tiltBound) return;
    card._tiltBound = true;

    card.addEventListener('mousemove', (e) => {
      const rect = card.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;
      const xPct = (x / rect.width) - 0.5;
      const yPct = (y / rect.height) - 0.5;

      const rotX = (-yPct * 7).toFixed(2);
      const rotY = (xPct * 7).toFixed(2);
      card.style.transform = `perspective(1000px) rotateX(${rotX}deg) rotateY(${rotY}deg) translateY(-4px)`;
    });

    card.addEventListener('mouseleave', () => {
      card.style.transform = '';
    });
  });
}

// Scroll-Driven Reading Progress and Background Parallax
function initScrollEffects() {
  const progressBar = document.getElementById('scrollProgressBar');
  const bgImg = document.getElementById('stadiumBgImage');

  window.addEventListener('scroll', () => {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;

    if (docHeight > 0 && progressBar) {
      const pct = Math.min(100, Math.max(0, (scrollTop / docHeight) * 100));
      progressBar.style.width = pct + '%';
    }

    if (bgImg) {
      // Smooth vertical parallax shift for stadium backdrop
      bgImg.style.transform = `translateY(${scrollTop * 0.08}px)`;
    }
  }, { passive: true });
}

// 3D Scroll-Triggered Reveal Animations - Ensure all elements are immediately visible
function initScrollReveal() {
  document.querySelectorAll('.reveal-3d').forEach(el => {
    el.classList.add('is-visible');
  });

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          entry.target.classList.add('is-visible');
        }
      });
    }, { threshold: 0.05 });

    document.querySelectorAll('.reveal-3d').forEach(el => observer.observe(el));
  }
}

// Embedded Report Showcase Filter Tabs
function initReportShowcase() {
  const filterBtns = document.querySelectorAll('.report-filter-btn');
  const reportCards = document.querySelectorAll('.report-card');

  filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      filterBtns.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const filter = btn.dataset.filter;

      reportCards.forEach(card => {
        if (filter === 'all' || card.dataset.category === filter) {
          card.style.display = 'flex';
        } else {
          card.style.display = 'none';
        }
      });
    });
  });
}

// High-Res Report Inspection Lightbox Modal Handlers
window.openReportModal = function(src, title, caption) {
  const modal = document.getElementById('reportLightbox');
  const img = document.getElementById('lightboxImg');
  const titleEl = document.getElementById('lightboxTitle');
  const capEl = document.getElementById('lightboxCaption');
  const dlBtn = document.getElementById('lightboxDlBtn');

  if (img) img.src = src;
  if (titleEl) titleEl.textContent = title;
  if (capEl) capEl.textContent = caption;
  if (dlBtn) dlBtn.href = src;

  if (modal) {
    modal.classList.add('active');
    document.body.style.overflow = 'hidden';
  }
};

window.closeReportModal = function() {
  const modal = document.getElementById('reportLightbox');
  if (modal) {
    modal.classList.remove('active');
    document.body.style.overflow = '';
  }
};

window.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') window.closeReportModal();
});

// Initialize immediately or on DOMContentLoaded
function initApp() {
  checkHealth();
  loadRoster();
  initScrollEffects();
  initScrollReveal();
  init3DTilt();
  initReportShowcase();

  const searchInput = document.getElementById('rosterSearch');
  if (searchInput) {
    searchInput.addEventListener('input', (e) => {
      currentSearch = e.target.value.trim();
      renderRoster();
    });
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

/* ==========================================================================
   DATABASE PHOTO MODAL — DEDICATED SEPARATE PLAYER GALLERY
   ========================================================================== */

let dbPlayers = [];
let dbFilter = 'ALL';
let dbSearch = '';
let dbInitialized = false;

async function openDatabaseModal() {
  const overlay = document.getElementById('dbModalOverlay');
  if (!overlay) return;
  overlay.classList.add('open');
  overlay.setAttribute('aria-hidden', 'false');
  document.body.classList.add('db-modal-open');

  if (!dbInitialized) {
    dbInitialized = true;
    renderDbAlphabet();

    const grid = document.getElementById('dbPhotoGrid');
    if (grid) {
      grid.innerHTML = '<div style="grid-column: 1/-1; text-align: center; color: var(--blue-cyan); padding: 50px;">⚡ Loading 323 Legends...</div>';
    }

    try {
      if (window.playersMeta && window.playersMeta.length > 0) {
        dbPlayers = window.playersMeta;
      } else {
        const res = await fetch('/players_meta.json');
        dbPlayers = await res.json();
        window.playersMeta = dbPlayers;
      }
      renderDbGrid();
    } catch (err) {
      console.error('Failed to load database:', err);
      if (grid) grid.innerHTML = '<p class="db-empty">Could not load player database. Please try again.</p>';
    }

    const dbSearchInput = document.getElementById('dbSearch');
    if (dbSearchInput) {
      dbSearchInput.addEventListener('input', (e) => {
        dbSearch = e.target.value.trim();
        renderDbGrid();
      });
    }
  }
}

function closeDatabaseModal() {
  const overlay = document.getElementById('dbModalOverlay');
  if (!overlay) return;
  overlay.classList.remove('open');
  overlay.setAttribute('aria-hidden', 'true');
  document.body.classList.remove('db-modal-open');
}

document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape') closeDatabaseModal();
});

function renderDbAlphabet() {
  const container = document.getElementById('dbAlphabetFilter');
  if (!container) return;
  const letters = 'ALL,A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q,R,S,T,U,V,W,X,Y,Z'.split(',');
  container.innerHTML = letters.map(l => `
    <button type="button" class="alphabet-btn ${l === dbFilter ? 'active' : ''}" data-letter="${l}">${l}</button>
  `).join('');
  container.addEventListener('click', (e) => {
    const btn = e.target.closest('.alphabet-btn');
    if (!btn) return;
    dbFilter = btn.dataset.letter;
    container.querySelectorAll('.alphabet-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    renderDbGrid();
  });
}

function renderDbGrid() {
  const grid = document.getElementById('dbPhotoGrid');
  if (!grid) return;

  const filtered = dbPlayers.filter(p => {
    const matchLetter = dbFilter === 'ALL' || (p.name && p.name.toUpperCase().startsWith(dbFilter));
    const matchSearch = !dbSearch || (p.name && p.name.toLowerCase().includes(dbSearch.toLowerCase()));
    return matchLetter && matchSearch;
  });

  if (filtered.length === 0) {
    grid.innerHTML = '<p class="db-empty">No cricketers found matching your search.</p>';
    return;
  }

  grid.innerHTML = filtered.map(p => {
    const sampleImg = p.sample_image ? p.sample_image.split('/').map(encodeURIComponent).join('/') : '';
    const imgSrc = sampleImg ? ('/player-images/' + sampleImg) : '';
    const safeName = (p.name || '').replace(/'/g, "\\'");
    return `
      <div class="db-player-card tilt-card">
        <div class="db-player-thumb">
          <img src="${imgSrc}" alt="${p.name}" loading="lazy" onerror="this.onerror=null; this.src='data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect fill=%22%230A2452%22 width=%22100%22 height=%22100%22/><text fill=%22%2338BDF8%22 font-size=%2232%22 x=%2250%22 y=%2255%22 text-anchor=%22middle%22 dominant-baseline=%22middle%22>🏏</text></svg>';">
        </div>
        <div class="db-player-info">
          <div class="db-player-name">${p.name}</div>
          <div class="db-player-meta">
            <span>${p.count || 25} Photos</span>
            <span>India</span>
          </div>
          <button type="button" class="db-player-test-btn" onclick="closeDatabaseModal(); setTimeout(() => loadQuickSample('${safeName}'), 300);">⚡ Test Face</button>
        </div>
      </div>
    `;
  }).join('');

  if (typeof init3DTilt === 'function') {
    init3DTilt();
  }
}

window.openDatabaseModal = openDatabaseModal;
window.closeDatabaseModal = closeDatabaseModal;
