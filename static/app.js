const PREDICT_INTERVAL_MS = 200;

const drawCanvas = document.getElementById("draw-canvas");
const drawCtx = drawCanvas.getContext("2d", { willReadFrequently: true });
const previewCanvas = document.getElementById("preview-canvas");
const previewCtx = previewCanvas.getContext("2d", { willReadFrequently: true });
const fc1Canvas = document.getElementById("fc1-canvas");
const fc1Ctx = fc1Canvas.getContext("2d");

const probBarsContainer = document.getElementById("prob-bars");
const predictedDigitEl = document.getElementById("predicted-digit");
const confidenceEl = document.getElementById("prediction-confidence");
const drawStatusEl = document.getElementById("draw-status");
const modelStatusEl = document.getElementById("model-status");

const conv1Grid = document.getElementById("conv1-grid");
const conv2Grid = document.getElementById("conv2-grid");
const conv1FiltersGrid = document.getElementById("conv1-filters");
const conv2FiltersGrid = document.getElementById("conv2-filters");
const weightsContainer = document.getElementById("weights-container");

const offscreen = document.createElement("canvas");
offscreen.width = 28;
offscreen.height = 28;
const offscreenCtx = offscreen.getContext("2d", { willReadFrequently: true });

const probBarEls = [];

let drawing = false;
let lastX = 0;
let lastY = 0;
let livePredictTimer = null;
let predictInFlight = false;
let statusPollTimer = null;
let previousStatusState = null;

function initCanvas() {
  drawCtx.fillStyle = "#000";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";
  drawCtx.strokeStyle = "#fff";
  drawCtx.lineWidth = 18;

  previewCtx.fillStyle = "#000";
  previewCtx.fillRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewCtx.imageSmoothingEnabled = false;
}

function initProbBars() {
  for (let digit = 0; digit < 10; digit += 1) {
    const row = document.createElement("div");
    row.className = "prob-row";

    const label = document.createElement("span");
    label.textContent = String(digit);

    const track = document.createElement("div");
    track.className = "bar-track";

    const fill = document.createElement("div");
    fill.className = "bar-fill";
    track.appendChild(fill);

    const value = document.createElement("span");
    value.textContent = "0.00";

    row.appendChild(label);
    row.appendChild(track);
    row.appendChild(value);
    probBarsContainer.appendChild(row);

    probBarEls.push({ fill, value, label });
  }
}

function getPointerPosition(event) {
  const rect = drawCanvas.getBoundingClientRect();
  const scaleX = drawCanvas.width / rect.width;
  const scaleY = drawCanvas.height / rect.height;
  return {
    x: (event.clientX - rect.left) * scaleX,
    y: (event.clientY - rect.top) * scaleY,
  };
}

function startDrawing(event) {
  event.preventDefault();
  drawCanvas.setPointerCapture(event.pointerId);
  const { x, y } = getPointerPosition(event);
  drawing = true;
  lastX = x;
  lastY = y;
  drawCtx.beginPath();
  drawCtx.moveTo(x, y);
  startLivePrediction();
}

function draw(event) {
  if (!drawing) {
    return;
  }

  event.preventDefault();
  const { x, y } = getPointerPosition(event);
  drawCtx.lineTo(x, y);
  drawCtx.stroke();
  lastX = x;
  lastY = y;
}

function stopDrawing(event) {
  if (!drawing) {
    return;
  }

  event.preventDefault();
  drawing = false;
  drawCtx.beginPath();
  drawCtx.moveTo(lastX, lastY);
  stopLivePrediction();
  requestPrediction({ refreshVisualizations: true });
}

function startLivePrediction() {
  if (livePredictTimer !== null) {
    return;
  }
  livePredictTimer = window.setInterval(() => {
    requestPrediction();
  }, PREDICT_INTERVAL_MS);
}

function stopLivePrediction() {
  if (livePredictTimer === null) {
    return;
  }
  window.clearInterval(livePredictTimer);
  livePredictTimer = null;
}

function clearCanvas() {
  stopLivePrediction();
  drawCtx.fillStyle = "#000";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
  predictedDigitEl.textContent = "-";
  confidenceEl.textContent = "confidence: -";
  drawStatusEl.textContent = "Canvas cleared.";
  renderProbabilities(new Array(10).fill(0));
  canvasToMatrix();
}

function randomCanvas() {
  clearCanvas();
  drawCtx.strokeStyle = "#fff";
  drawCtx.lineWidth = 14 + Math.random() * 8;

  for (let i = 0; i < 4; i += 1) {
    const x1 = 35 + Math.random() * 210;
    const y1 = 35 + Math.random() * 210;
    const x2 = 35 + Math.random() * 210;
    const y2 = 35 + Math.random() * 210;

    drawCtx.beginPath();
    drawCtx.moveTo(x1, y1);
    drawCtx.lineTo(x2, y2);
    drawCtx.stroke();
  }

  requestPrediction({ refreshVisualizations: true });
}

function canvasToMatrix() {
  offscreenCtx.fillStyle = "#000";
  offscreenCtx.fillRect(0, 0, 28, 28);
  offscreenCtx.drawImage(drawCanvas, 0, 0, 28, 28);

  const imageData = offscreenCtx.getImageData(0, 0, 28, 28).data;
  const matrix = [];

  for (let y = 0; y < 28; y += 1) {
    const row = [];
    for (let x = 0; x < 28; x += 1) {
      const idx = (y * 28 + x) * 4;
      const value = imageData[idx] / 255;
      row.push(Number(value.toFixed(4)));
    }
    matrix.push(row);
  }

  previewCtx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  previewCtx.imageSmoothingEnabled = false;
  previewCtx.drawImage(offscreen, 0, 0, previewCanvas.width, previewCanvas.height);

  return matrix;
}

function renderProbabilities(probabilities) {
  const values = probabilities || new Array(10).fill(0);
  for (let i = 0; i < 10; i += 1) {
    const probability = Number(values[i] || 0);
    const percent = Math.max(0, Math.min(100, probability * 100));
    probBarEls[i].fill.style.maxWidth = `${percent.toFixed(2)}%`;
    probBarEls[i].value.textContent = probability.toFixed(2);
    probBarEls[i].label.style.fontWeight = "500";
  }

  const maxIndex = values.indexOf(Math.max(...values));
  if (maxIndex >= 0) {
    probBarEls[maxIndex].label.style.fontWeight = "700";
  }
}

async function fetchJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (body && body.detail) {
        detail = body.detail;
      }
    } catch (error) {
      // Keep default detail when response is not JSON.
    }
    throw new Error(detail);
  }

  return response.json();
}

async function requestPrediction(options = {}) {
  const { refreshVisualizations = false } = options;

  if (predictInFlight) {
    return;
  }

  predictInFlight = true;
  const image = canvasToMatrix();

  try {
    const data = await fetchJson("/api/predict", {
      method: "POST",
      body: JSON.stringify({ image }),
    });

    predictedDigitEl.textContent = String(data.predicted_class);
    confidenceEl.textContent = `confidence: ${(data.confidence * 100).toFixed(1)}%`;
    drawStatusEl.textContent = `Predicted in ${data.latency_ms.toFixed(1)} ms`;
    renderProbabilities(data.probabilities);

    if (refreshVisualizations) {
      refreshActivations();
    }
  } catch (error) {
    drawStatusEl.textContent = `Prediction unavailable: ${error.message}`;
  } finally {
    predictInFlight = false;
  }
}

function drawHeatmap(canvas, matrix, minValue, maxValue) {
  const ctx = canvas.getContext("2d");
  const rows = matrix.length;
  const cols = matrix[0].length;
  const cellWidth = canvas.width / cols;
  const cellHeight = canvas.height / rows;

  const range = maxValue - minValue || 1;

  for (let y = 0; y < rows; y += 1) {
    for (let x = 0; x < cols; x += 1) {
      const value = matrix[y][x];
      const normalized = (value - minValue) / range;
      const clamped = Math.max(0, Math.min(1, normalized));
      const shade = Math.round(clamped * 255);
      ctx.fillStyle = `rgb(${shade}, ${shade}, ${shade})`;
      ctx.fillRect(x * cellWidth, y * cellHeight, cellWidth, cellHeight);
    }
  }
}

function renderFeatureGrid(container, payload, options = {}) {
  const { squeezeFirstChannel = false, canvasSize = 48 } = options;
  container.innerHTML = "";

  const maps = payload.data;
  const shape = payload.shape || [];

  if (!Array.isArray(maps)) {
    return;
  }

  for (let i = 0; i < maps.length; i += 1) {
    let matrix = maps[i];

    if (shape.length === 4 && squeezeFirstChannel && Array.isArray(matrix) && Array.isArray(matrix[0])) {
      matrix = matrix[0];
    }

    const canvas = document.createElement("canvas");
    canvas.width = canvasSize;
    canvas.height = canvasSize;
    container.appendChild(canvas);

    drawHeatmap(canvas, matrix, payload.min, payload.max);
  }
}

function renderVectorBars(payload) {
  fc1Ctx.clearRect(0, 0, fc1Canvas.width, fc1Canvas.height);
  fc1Ctx.fillStyle = "#f9f9f9";
  fc1Ctx.fillRect(0, 0, fc1Canvas.width, fc1Canvas.height);

  const values = payload.data;
  if (!Array.isArray(values) || values.length === 0) {
    return;
  }

  const min = payload.min;
  const max = payload.max;
  const baseline = fc1Canvas.height - 12;
  const width = fc1Canvas.width / values.length;
  const range = max - min || 1;

  fc1Ctx.fillStyle = "#1a7d64";
  for (let i = 0; i < values.length; i += 1) {
    const normalized = (values[i] - min) / range;
    const height = Math.max(2, normalized * (fc1Canvas.height - 22));
    fc1Ctx.fillRect(i * width, baseline - height, Math.max(1, width - 1), height);
  }
}

async function refreshActivations() {
  const image = canvasToMatrix();
  try {
    const data = await fetchJson("/api/activations", {
      method: "POST",
      body: JSON.stringify({ image }),
    });

    renderFeatureGrid(conv1Grid, data.conv1, { canvasSize: 52 });
    renderFeatureGrid(conv2Grid, data.conv2, { canvasSize: 52 });
    renderVectorBars(data.fc1);
  } catch (error) {
    drawStatusEl.textContent = `Activation view unavailable: ${error.message}`;
  }
}

async function refreshFilters() {
  try {
    const data = await fetchJson("/api/filters", { method: "GET" });
    renderFeatureGrid(conv1FiltersGrid, data.conv1, {
      squeezeFirstChannel: true,
      canvasSize: 52,
    });

    renderFeatureGrid(conv2FiltersGrid, {
      data: data.conv2_summary.data,
      shape: data.conv2_summary.reduced_shape,
      min: data.conv2_summary.min,
      max: data.conv2_summary.max,
    });
  } catch (error) {
    drawStatusEl.textContent = `Filter view unavailable: ${error.message}`;
  }
}

function drawHistogram(canvas, counts) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#fcfcfc";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  const maxCount = Math.max(...counts, 1);
  const barWidth = canvas.width / counts.length;

  ctx.fillStyle = "#2f876f";
  for (let i = 0; i < counts.length; i += 1) {
    const height = (counts[i] / maxCount) * (canvas.height - 8);
    ctx.fillRect(i * barWidth, canvas.height - height, Math.max(1, barWidth - 1), height);
  }
}

async function refreshWeights() {
  try {
    const data = await fetchJson("/api/weights", { method: "GET" });
    weightsContainer.innerHTML = "";

    for (const layer of data.layers) {
      const card = document.createElement("div");
      card.className = "weight-card";

      const title = document.createElement("strong");
      title.textContent = layer.name;

      const meta = document.createElement("div");
      meta.className = "weight-meta";
      meta.textContent = `count=${layer.count} mean=${layer.mean.toFixed(4)} std=${layer.std.toFixed(4)}`;

      const canvas = document.createElement("canvas");
      canvas.width = 420;
      canvas.height = 90;

      card.appendChild(title);
      card.appendChild(meta);
      card.appendChild(canvas);
      weightsContainer.appendChild(card);

      drawHistogram(canvas, layer.hist.counts);
    }
  } catch (error) {
    drawStatusEl.textContent = `Weight view unavailable: ${error.message}`;
  }
}

async function fetchModelStatus() {
  try {
    const data = await fetchJson("/api/model/status", { method: "GET" });
    modelStatusEl.textContent = JSON.stringify(data, null, 2);

    if (previousStatusState === "training" && data.state === "ready") {
      drawStatusEl.textContent = "Training complete. Model updated.";
      refreshFilters();
      refreshWeights();
      requestPrediction();
    }

    previousStatusState = data.state;
  } catch (error) {
    modelStatusEl.textContent = `Failed to load model status: ${error.message}`;
  }
}

async function startTraining(event) {
  event.preventDefault();

  const epochs = Number(document.getElementById("epochs").value);
  const batchSize = Number(document.getElementById("batch-size").value);
  const lr = Number(document.getElementById("lr").value);
  const forceRestart = document.getElementById("force-restart").checked;

  try {
    const result = await fetchJson("/api/model/train", {
      method: "POST",
      body: JSON.stringify({
        epochs,
        batch_size: batchSize,
        lr,
        force_restart: forceRestart,
      }),
    });

    drawStatusEl.textContent = `Training started: ${result.job_id}`;
    fetchModelStatus();
  } catch (error) {
    drawStatusEl.textContent = `Training request failed: ${error.message}`;
  }
}

function installEventHandlers() {
  drawCanvas.addEventListener("pointerdown", startDrawing);
  drawCanvas.addEventListener("pointermove", draw);
  drawCanvas.addEventListener("pointerup", stopDrawing);
  drawCanvas.addEventListener("pointerleave", stopDrawing);
  drawCanvas.addEventListener("pointercancel", stopDrawing);

  document.getElementById("clear-canvas").addEventListener("click", clearCanvas);
  document.getElementById("random-canvas").addEventListener("click", randomCanvas);
  document.getElementById("predict-now").addEventListener("click", () => {
    requestPrediction({ refreshVisualizations: true });
  });

  document.getElementById("refresh-activations").addEventListener("click", refreshActivations);
  document.getElementById("refresh-filters").addEventListener("click", refreshFilters);
  document.getElementById("refresh-weights").addEventListener("click", refreshWeights);
  document.getElementById("refresh-status").addEventListener("click", fetchModelStatus);

  document.getElementById("training-form").addEventListener("submit", startTraining);
}

function startStatusPolling() {
  fetchModelStatus();
  if (statusPollTimer !== null) {
    clearInterval(statusPollTimer);
  }
  statusPollTimer = setInterval(fetchModelStatus, 3000);
}

function bootstrap() {
  initCanvas();
  initProbBars();
  installEventHandlers();
  canvasToMatrix();
  startStatusPolling();
  refreshFilters();
  refreshWeights();
}

bootstrap();
