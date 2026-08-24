const THRESHOLD = 0.5;
const state = { image: null, imageData: null, predictions: null, status: null };
const elements = {
  emptyState: document.getElementById('emptyState'),
  workspace: document.getElementById('workspace'),
  originalCanvas: document.getElementById('originalCanvas'),
  heatmapCanvas: document.getElementById('heatmapCanvas'),
  diseaseList: document.getElementById('diseaseList'),
  primaryDiagnosis: document.getElementById('primaryDiagnosis'),
  primaryConf: document.getElementById('primaryConf'),
  gauges: document.getElementById('gauges'),
  headerUploadBtn: document.getElementById('headerUploadBtn'),
  systemStatus: document.getElementById('systemStatus'),
  modelSummary: document.getElementById('modelSummary'),
  studyName: document.getElementById('studyName'),
  studyDate: document.getElementById('studyDate'),
};

function setStatus(message, ready = false) {
  elements.systemStatus.textContent = message;
  elements.systemStatus.className = ready
    ? 'bg-emerald-950 border border-emerald-500 text-emerald-400 rounded-2xl p-4 text-sm font-medium'
    : 'bg-amber-950 border border-amber-500 text-amber-300 rounded-2xl p-4 text-sm font-medium';
}

async function checkModel() {
  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    const data = await response.json();
    state.status = data;
    if (!response.ok || !data.ready) throw new Error(data.error || 'Model is unavailable');
    elements.modelSummary.textContent = `${(data.parameters / 1e6).toFixed(1)} M parameters • ${data.device}`;
    elements.modelSummary.className = 'text-xs text-emerald-400';
    setStatus('Real pruned checkpoint loaded • Ready for research inference', true);
  } catch (error) {
    elements.modelSummary.textContent = 'Checkpoint required';
    setStatus(error.message);
  }
}

async function processImage(image, imageData) {
  elements.emptyState.classList.add('hidden');
  elements.workspace.classList.remove('hidden');
  elements.headerUploadBtn.classList.remove('hidden');
  displayOriginalImage(image);
  setStatus('Running pruned ViT inference…');
  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData }),
    });
    const results = await response.json();
    if (!response.ok) throw new Error(results.error || `Inference failed (${response.status})`);
    state.predictions = results;
    displayHeatmap(image, results.saliency);
    updateUI(results);
    document.getElementById('modeBtn').textContent = `Pruned • ${results.inference_time_ms} ms`;
    setStatus(`Inference complete • ${results.inference_time_ms} ms • ${results.device}`, true);
  } catch (error) {
    state.predictions = null;
    elements.primaryDiagnosis.textContent = 'Inference unavailable';
    elements.primaryConf.textContent = '—';
    elements.gauges.innerHTML = '';
    elements.diseaseList.innerHTML = `<div class="text-amber-300 p-4 border border-amber-700 rounded-xl">${escapeHtml(error.message)}</div>`;
    setStatus(error.message);
  }
}

function displayOriginalImage(image) {
  const c = elements.originalCanvas;
  c.width = image.naturalWidth;
  c.height = image.naturalHeight;
  c.getContext('2d').drawImage(image, 0, 0);
}

function displayHeatmap(image, heatmap) {
  const c = elements.heatmapCanvas;
  const ctx = c.getContext('2d');
  const grid = document.createElement('canvas');
  grid.width = heatmap[0].length;
  grid.height = heatmap.length;
  const gridCtx = grid.getContext('2d');
  const pixels = gridCtx.createImageData(grid.width, grid.height);
  heatmap.forEach((row, y) => row.forEach((value, x) => {
    const i = (y * grid.width + x) * 4;
    pixels.data[i] = 255;
    pixels.data[i + 1] = Math.round(220 * (1 - value));
    pixels.data[i + 2] = 0;
    pixels.data[i + 3] = value < 0.15 ? 0 : Math.round(190 * value);
  }));
  gridCtx.putImageData(pixels, 0, 0);
  c.width = image.naturalWidth;
  c.height = image.naturalHeight;
  ctx.drawImage(image, 0, 0);
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(grid, 0, 0, c.width, c.height);
}

function updateUI(results) {
  const order = results.probabilities.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
  const top = order[0];
  elements.primaryDiagnosis.textContent = results.labels[top.i];
  elements.primaryConf.textContent = (top.p * 100).toFixed(1);
  elements.gauges.innerHTML = order.slice(0, 4).map(({ p, i }) => `
    <div class="text-center">
      <div class="gauge w-16 h-16 rounded-full mx-auto flex items-center justify-center text-xs font-bold" style="--percent:${p * 100}%">${Math.round(p * 100)}%</div>
      <div class="text-[10px] mt-2 text-slate-400">${escapeHtml(results.labels[i])}</div>
    </div>`).join('');
  elements.diseaseList.innerHTML = order.map(({ p, i }) => `
    <div class="flex justify-between items-center py-3 border-b border-slate-700 ${p >= THRESHOLD ? 'text-emerald-400' : ''}">
      <span>${escapeHtml(results.labels[i])}</span><span class="font-mono">${(p * 100).toFixed(1)}%</span>
    </div>`).join('');
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
}

function exportReport() {
  if (!state.predictions) return setStatus('Run a successful inference before exporting.');
  const report = {
    model: state.status?.model || 'ThorPruneViT',
    timestamp: new Date().toISOString(),
    research_use_only: true,
    inference_time_ms: state.predictions.inference_time_ms,
    device: state.predictions.device,
    findings: state.predictions.labels.map((finding, i) => ({ finding, probability: state.predictions.probabilities[i] })),
  };
  const url = URL.createObjectURL(new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = url;
  link.download = `ThorPruneViT-Research-${Date.now()}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

const fileInput = document.createElement('input');
fileInput.type = 'file';
fileInput.accept = 'image/jpeg,image/png,image/webp';
document.getElementById('uploadArea').addEventListener('click', () => fileInput.click());
elements.headerUploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', event => {
  const file = event.target.files[0];
  if (!file) return;
  if (file.size > 15 * 1024 * 1024) return setStatus('Image exceeds the 15 MB limit.');
  const reader = new FileReader();
  reader.onload = loadEvent => {
    const image = new Image();
    image.onload = () => {
      state.image = image;
      state.imageData = loadEvent.target.result;
      elements.studyName.textContent = file.name;
      elements.studyDate.textContent = new Date().toLocaleDateString();
      processImage(image, state.imageData);
    };
    image.onerror = () => setStatus('The selected file is not a readable image.');
    image.src = loadEvent.target.result;
  };
  reader.readAsDataURL(file);
});

window.exportReport = exportReport;
checkModel();
