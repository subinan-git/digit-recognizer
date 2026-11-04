// 손글씨 숫자 인식기 (웹 버전)
// 1) 모델: model.json의 가중치로 신경망 계산을 직접 수행
// 2) 그리기: 마우스/터치로 캔버스에 숫자를 그림
// 3) 전처리: 그림을 MNIST와 같은 28x28 형식으로 변환
// 4) 화면: 예측 결과와 확률 막대 표시

const BRUSH_WIDTH = 20;
const MNIST_SIZE = 28;
const DIGIT_BOX = 20; // MNIST 숫자는 28x28 안의 20x20 영역에 맞춰져 있음

const drawCanvas = document.getElementById("draw-canvas");
const drawCtx = drawCanvas.getContext("2d");
const previewCanvas = document.getElementById("preview-canvas");
const previewCtx = previewCanvas.getContext("2d");
const resultDigit = document.getElementById("result-digit");
const statusText = document.getElementById("status");
const probBars = document.getElementById("prob-bars");

// ===== 1. 모델 =====

let model = null;

async function loadModel() {
  const response = await fetch("model.json");
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`);
  }
  model = await response.json();
}

// 완전연결층: output[j] = biases[j] + sum(input[i] * weights[i][j])
function dense(input, weights, biases) {
  const output = biases.slice();
  for (let i = 0; i < input.length; i++) {
    const value = input[i];
    if (value === 0) continue; // 빈 픽셀은 결과에 영향이 없으므로 건너뜀
    const row = weights[i];
    for (let j = 0; j < output.length; j++) {
      output[j] += value * row[j];
    }
  }
  return output;
}

function relu(values) {
  return values.map((v) => Math.max(0, v));
}

// 출력값을 합이 1인 확률로 변환 (큰 수 방지를 위해 최댓값을 빼고 계산)
function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

// 784개 픽셀 값 -> 숫자 0~9 각각의 확률 10개
function predict(pixels) {
  let x = pixels;
  model.layers.forEach((layer, index) => {
    x = dense(x, layer.weights, layer.biases);
    const isLastLayer = index === model.layers.length - 1;
    if (!isLastLayer) {
      x = relu(x);
    }
  });
  return softmax(x);
}

// ===== 2. 그리기 =====

let lastPoint = null;

function clearCanvas() {
  drawCtx.fillStyle = "black";
  drawCtx.fillRect(0, 0, drawCanvas.width, drawCanvas.height);
}

// 화면에 표시되는 캔버스 크기와 실제 픽셀 크기가 다를 수 있어 좌표를 환산
function getPoint(event) {
  const rect = drawCanvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) * drawCanvas.width) / rect.width,
    y: ((event.clientY - rect.top) * drawCanvas.height) / rect.height,
  };
}

function drawTo(point) {
  drawCtx.strokeStyle = "white";
  drawCtx.fillStyle = "white";
  drawCtx.lineWidth = BRUSH_WIDTH;
  drawCtx.lineCap = "round";
  drawCtx.lineJoin = "round";

  if (lastPoint) {
    drawCtx.beginPath();
    drawCtx.moveTo(lastPoint.x, lastPoint.y);
    drawCtx.lineTo(point.x, point.y);
    drawCtx.stroke();
  } else {
    // 클릭만 했을 때도 점이 찍히도록
    drawCtx.beginPath();
    drawCtx.arc(point.x, point.y, BRUSH_WIDTH / 2, 0, Math.PI * 2);
    drawCtx.fill();
  }
  lastPoint = point;
}

// Pointer 이벤트는 마우스, 터치, 펜을 한 번에 처리함
drawCanvas.addEventListener("pointerdown", (event) => {
  drawCanvas.setPointerCapture(event.pointerId);
  lastPoint = null;
  drawTo(getPoint(event));
});

drawCanvas.addEventListener("pointermove", (event) => {
  if (lastPoint) {
    drawTo(getPoint(event));
  }
});

drawCanvas.addEventListener("pointerup", () => {
  lastPoint = null;
  runPrediction();
});

drawCanvas.addEventListener("pointercancel", () => {
  lastPoint = null;
});

// ===== 3. 전처리 =====

function createCanvas(width, height) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  return canvas;
}

// 획이 있는 영역(바운딩 박스)을 찾음. 아무것도 없으면 null
function findBoundingBox(canvas) {
  const { width, height } = canvas;
  const data = canvas.getContext("2d").getImageData(0, 0, width, height).data;
  let left = width, right = -1, top = height, bottom = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (data[(y * width + x) * 4] > 0) { // 빨강 채널 (흑백이라 R=G=B)
        left = Math.min(left, x);
        right = Math.max(right, x);
        top = Math.min(top, y);
        bottom = Math.max(bottom, y);
      }
    }
  }
  if (right < 0) return null;
  return { x: left, y: top, width: right - left + 1, height: bottom - top + 1 };
}

// 한 번에 크게 줄이면 획이 깨지므로 절반씩 여러 번 나눠 줄임
function resize(source, box, targetWidth, targetHeight) {
  let current = createCanvas(box.width, box.height);
  current.getContext("2d").drawImage(
    source, box.x, box.y, box.width, box.height, 0, 0, box.width, box.height
  );
  while (current.width > targetWidth * 2 || current.height > targetHeight * 2) {
    const next = createCanvas(
      Math.max(targetWidth, Math.ceil(current.width / 2)),
      Math.max(targetHeight, Math.ceil(current.height / 2))
    );
    next.getContext("2d").drawImage(current, 0, 0, next.width, next.height);
    current = next;
  }
  const result = createCanvas(targetWidth, targetHeight);
  result.getContext("2d").drawImage(current, 0, 0, targetWidth, targetHeight);
  return result;
}

// 밝기의 무게중심 (픽셀 좌표 기준)
function centerOfMass(canvas) {
  const { width, height } = canvas;
  const data = canvas.getContext("2d").getImageData(0, 0, width, height).data;
  let total = 0, sumX = 0, sumY = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const value = data[(y * width + x) * 4];
      total += value;
      sumX += x * value;
      sumY += y * value;
    }
  }
  return { x: sumX / total, y: sumY / total };
}

// 그린 그림 -> MNIST 형식의 784개 값(0~1). 파이썬 버전의 preprocess()와 같은 순서
function preprocess(canvas) {
  const box = findBoundingBox(canvas);
  if (!box) return null;

  // 비율을 유지하면서 긴 변이 20픽셀이 되도록 축소
  const scale = DIGIT_BOX / Math.max(box.width, box.height);
  const digit = resize(
    canvas, box,
    Math.max(1, Math.round(box.width * scale)),
    Math.max(1, Math.round(box.height * scale))
  );

  // 무게중심이 28x28 프레임의 중앙(14, 14)에 오도록 배치
  const center = centerOfMass(digit);
  previewCtx.fillStyle = "black";
  previewCtx.fillRect(0, 0, MNIST_SIZE, MNIST_SIZE);
  previewCtx.drawImage(digit, MNIST_SIZE / 2 - center.x, MNIST_SIZE / 2 - center.y);

  const data = previewCtx.getImageData(0, 0, MNIST_SIZE, MNIST_SIZE).data;
  const pixels = new Array(MNIST_SIZE * MNIST_SIZE);
  for (let i = 0; i < pixels.length; i++) {
    pixels[i] = data[i * 4] / 255;
  }
  return pixels;
}

// ===== 4. 화면 =====

// 숫자 0~9의 확률 막대를 처음 한 번 만들어 둠
function createBars() {
  for (let digit = 0; digit < 10; digit++) {
    const item = document.createElement("li");
    item.innerHTML = `
      <span>${digit}</span>
      <div class="bar-track"><div class="bar-fill"></div></div>
      <span class="bar-value">0%</span>`;
    probBars.appendChild(item);
  }
}

function showProbabilities(probs) {
  const best = probs.indexOf(Math.max(...probs));
  probBars.querySelectorAll("li").forEach((item, digit) => {
    const percent = probs[digit] * 100;
    item.querySelector(".bar-fill").style.width = `${percent}%`;
    item.querySelector(".bar-value").textContent = `${percent.toFixed(0)}%`;
    item.classList.toggle("best", digit === best && percent > 0);
  });
}

function runPrediction() {
  if (!model) return;
  const pixels = preprocess(drawCanvas);
  if (!pixels) return;

  const probs = predict(pixels);
  const best = probs.indexOf(Math.max(...probs));
  resultDigit.textContent = best;
  statusText.textContent = `Confidence: ${(probs[best] * 100).toFixed(1)}%`;
  showProbabilities(probs);
}

function reset() {
  clearCanvas();
  previewCtx.fillStyle = "black";
  previewCtx.fillRect(0, 0, MNIST_SIZE, MNIST_SIZE);
  resultDigit.textContent = "?";
  statusText.textContent = model ? "Draw a digit" : "Loading model...";
  showProbabilities(new Array(10).fill(0));
}

document.getElementById("clear-button").addEventListener("click", reset);

// ===== 시작 =====

createBars();
reset();
loadModel()
  .then(() => {
    statusText.textContent = "Draw a digit";
  })
  .catch((error) => {
    statusText.textContent = `Failed to load model (${error.message})`;
  });
