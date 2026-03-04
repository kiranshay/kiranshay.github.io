/**
 * About Page Interactive Visualizations
 * - Decision Boundary Painter
 * - Monte Carlo Pi Estimator
 */

// ============================================
// DECISION BOUNDARY PAINTER
// ============================================

class DecisionBoundaryGame {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;

    this.ctx = this.canvas.getContext('2d');
    this.setupCanvas();

    // Data points: {x, y, label}
    this.points = [];
    this.currentClass = 0; // 0 = blue, 1 = orange

    // Simple neural network (2 inputs -> 8 hidden -> 1 output)
    this.initNetwork();

    // Training state
    this.isTraining = false;
    this.animationId = null;
    this.epoch = 0;
    this.learningRate = 0.3;

    // Setup
    this.setupControls();
    this.setupCanvasInteraction();
    this.render();
  }

  setupCanvas() {
    const container = this.canvas.parentElement;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    this.canvas.width = Math.min(400, rect.width - 40) * dpr;
    this.canvas.height = 360 * dpr;
    this.canvas.style.width = `${Math.min(400, rect.width - 40)}px`;
    this.canvas.style.height = '360px';
    this.ctx.scale(dpr, dpr);

    this.width = Math.min(400, rect.width - 40);
    this.height = 360;
  }

  initNetwork() {
    // Simple 2-layer network: 2 -> 4 -> 1
    // Smaller network = smoother decision boundaries
    this.hiddenSize = 4;
    this.w1 = this.randomMatrix(2, this.hiddenSize);
    this.b1 = this.zeroArray(this.hiddenSize);
    this.w2 = this.randomMatrix(this.hiddenSize, 1);
    this.b2 = this.zeroArray(1);
  }

  zeroMatrix(rows, cols) {
    const m = [];
    for (let i = 0; i < rows; i++) {
      m[i] = Array(cols).fill(0);
    }
    return m;
  }

  randomMatrix(rows, cols) {
    // He initialization: weights ~ N(0, sqrt(2/fan_in))
    const scale = Math.sqrt(2 / rows);
    const m = [];
    for (let i = 0; i < rows; i++) {
      m[i] = [];
      for (let j = 0; j < cols; j++) {
        // Box-Muller transform for normal distribution
        const u1 = Math.random();
        const u2 = Math.random();
        const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        m[i][j] = normal * scale;
      }
    }
    return m;
  }

  zeroArray(n) {
    return Array(n).fill(0);
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  sigmoidDerivative(x) {
    return x * (1 - x);
  }

  relu(x) {
    return Math.max(0, x);
  }

  reluDerivative(x) {
    return x > 0 ? 1 : 0;
  }

  forward(input) {
    // Hidden layer with ReLU
    this.hidden = [];
    for (let j = 0; j < this.hiddenSize; j++) {
      let sum = this.b1[j];
      for (let i = 0; i < 2; i++) {
        sum += input[i] * this.w1[i][j];
      }
      this.hidden[j] = this.relu(sum);
    }

    // Output layer with sigmoid
    let output = this.b2[0];
    for (let j = 0; j < this.hiddenSize; j++) {
      output += this.hidden[j] * this.w2[j][0];
    }
    this.output = this.sigmoid(output);

    return this.output;
  }

  backward(input, target) {
    // Output error
    const outputError = target - this.output;
    const outputDelta = outputError * this.sigmoidDerivative(this.output);

    // Hidden error
    const hiddenDelta = [];
    for (let j = 0; j < this.hiddenSize; j++) {
      const error = outputDelta * this.w2[j][0];
      hiddenDelta[j] = error * this.reluDerivative(this.hidden[j]);
    }

    // Update weights
    for (let j = 0; j < this.hiddenSize; j++) {
      this.w2[j][0] += this.learningRate * outputDelta * this.hidden[j];
    }
    this.b2[0] += this.learningRate * outputDelta;

    for (let i = 0; i < 2; i++) {
      for (let j = 0; j < this.hiddenSize; j++) {
        this.w1[i][j] += this.learningRate * hiddenDelta[j] * input[i];
      }
    }
    for (let j = 0; j < this.hiddenSize; j++) {
      this.b1[j] += this.learningRate * hiddenDelta[j];
    }

    return outputError * outputError;
  }

  trainStep() {
    if (this.points.length < 2) return 0;

    let totalLoss = 0;

    // Shuffle and train on all points
    const shuffled = [...this.points].sort(() => Math.random() - 0.5);

    for (const point of shuffled) {
      const input = [point.x, point.y];
      this.forward(input);
      totalLoss += this.backward(input, point.label);
    }

    this.epoch++;
    return totalLoss / this.points.length;
  }

  computeAccuracy() {
    if (this.points.length === 0) return 0;

    let correct = 0;
    for (const point of this.points) {
      const pred = this.forward([point.x, point.y]);
      const predLabel = pred > 0.5 ? 1 : 0;
      if (predLabel === point.label) correct++;
    }
    return correct / this.points.length;
  }

  getPlotDimensions() {
    // Shared calculation for consistency between click handling and rendering
    const padding = 25;
    const plotSize = Math.min(this.width, this.height - 60) - padding * 2;
    const plotX = (this.width - plotSize) / 2;
    const plotY = padding;
    return { padding, plotSize, plotX, plotY };
  }

  setupCanvasInteraction() {
    this.canvas.addEventListener('click', (e) => {
      // Use offsetX/offsetY for accurate canvas-relative coordinates
      // Then scale by the ratio of logical size to displayed size
      const displayWidth = this.canvas.offsetWidth;
      const displayHeight = this.canvas.offsetHeight;

      const clickX = e.offsetX * (this.width / displayWidth);
      const clickY = e.offsetY * (this.height / displayHeight);

      // Convert to normalized coordinates (0-1)
      const { plotSize, plotX, plotY } = this.getPlotDimensions();

      if (clickX >= plotX && clickX <= plotX + plotSize &&
          clickY >= plotY && clickY <= plotY + plotSize) {
        const x = (clickX - plotX) / plotSize;
        const y = 1 - (clickY - plotY) / plotSize;

        this.points.push({ x, y, label: this.currentClass });
        this.updateStats();
        this.render();
      }
    });

    this.canvas.style.cursor = 'crosshair';
  }

  updateStats() {
    const epochEl = document.getElementById('db-epoch');
    const pointsEl = document.getElementById('db-points');
    const accEl = document.getElementById('db-accuracy');

    if (epochEl) epochEl.textContent = this.epoch;
    if (pointsEl) pointsEl.textContent = this.points.length;
    if (accEl) accEl.textContent = `${(this.computeAccuracy() * 100).toFixed(0)}%`;
  }

  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    // Clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    const { plotSize, plotX, plotY } = this.getPlotDimensions();

    // Show grey background when no points (ready state), otherwise show decision boundary
    if (this.points.length === 0) {
      // Neutral grey fill to indicate ready state
      ctx.fillStyle = 'rgba(100, 116, 139, 0.4)';
      ctx.fillRect(plotX, plotY, plotSize, plotSize);
    } else {
      // Draw decision boundary heatmap
      const resolution = 40;
      const cellSize = plotSize / resolution;

      for (let i = 0; i < resolution; i++) {
        for (let j = 0; j < resolution; j++) {
          const x = (i + 0.5) / resolution;
          const y = (j + 0.5) / resolution;
          const pred = this.forward([x, y]);

          // Interpolate between blue and orange based on prediction
          const t = pred;
          // Blue (class 0): #3b82f6, Orange (class 1): #f97316
          const r = Math.floor(59 + t * (249 - 59));
          const g = Math.floor(130 + t * (115 - 130));
          const b = Math.floor(246 + t * (22 - 246));

          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.4)`;
          ctx.fillRect(
            plotX + i * cellSize,
            plotY + (resolution - 1 - j) * cellSize,
            cellSize + 1,
            cellSize + 1
          );
        }
      }

      // Draw decision boundary line (pred = 0.5)
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();

      let firstPoint = true;
      const resolution2 = 40;
      for (let i = 0; i <= resolution2; i++) {
        for (let j = 0; j <= resolution2; j++) {
          const x = i / resolution2;
          const y = j / resolution2;
          const pred = this.forward([x, y]);

          if (Math.abs(pred - 0.5) < 0.05) {
            const sx = plotX + x * plotSize;
            const sy = plotY + (1 - y) * plotSize;

            if (firstPoint) {
              ctx.moveTo(sx, sy);
              firstPoint = false;
            } else {
              ctx.lineTo(sx, sy);
            }
          }
        }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw plot border
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    ctx.strokeRect(plotX, plotY, plotSize, plotSize);

    // Draw data points
    for (const point of this.points) {
      const sx = plotX + point.x * plotSize;
      const sy = plotY + (1 - point.y) * plotSize;

      // Outer ring
      ctx.beginPath();
      ctx.arc(sx, sy, 10, 0, Math.PI * 2);
      ctx.fillStyle = point.label === 0 ? '#3b82f6' : '#f97316';
      ctx.fill();

      // Inner circle
      ctx.beginPath();
      ctx.arc(sx, sy, 6, 0, Math.PI * 2);
      ctx.fillStyle = point.label === 0 ? '#60a5fa' : '#fb923c';
      ctx.fill();

      // Border
      ctx.strokeStyle = '#ffffff';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    // Legend
    const legendY = plotY + plotSize + 25;
    ctx.textAlign = 'left';
    ctx.fillStyle = '#3b82f6';
    ctx.beginPath();
    ctx.arc(plotX + 10, legendY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText('Class A', plotX + 20, legendY + 4);

    ctx.fillStyle = '#f97316';
    ctx.beginPath();
    ctx.arc(plotX + 80, legendY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Class B', plotX + 90, legendY + 4);

    // Instructions
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'center';
    ctx.fillText('Click to add points, then train!', w / 2, h - 12);
  }

  animate() {
    if (!this.isTraining) return;

    // Train multiple steps per frame
    for (let i = 0; i < 5; i++) {
      this.trainStep();
    }
    this.updateStats();
    this.render();

    // Auto-stop when 100% accuracy is achieved (with small tolerance for floating point)
    const accuracy = this.computeAccuracy();
    if (accuracy >= 0.9999 && this.points.length >= 2) {
      this.stop();
      return;
    }

    this.animationId = requestAnimationFrame(() => this.animate());
  }

  start() {
    if (this.isTraining || this.points.length < 2) return;
    this.isTraining = true;
    this.animate();
    this.updateButton();
  }

  stop() {
    this.isTraining = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    this.updateButton();
  }

  toggle() {
    if (this.isTraining) {
      this.stop();
    } else {
      this.start();
    }
  }

  updateButton() {
    const btn = document.getElementById('db-toggle-btn');
    if (btn) {
      btn.innerHTML = this.isTraining
        ? '<i data-lucide="pause"></i> Pause'
        : '<i data-lucide="play"></i> Train';
      if (typeof lucide !== 'undefined') lucide.createIcons();
    }
  }

  setClass(classLabel) {
    this.currentClass = classLabel;
    document.getElementById('class-a-btn')?.classList.toggle('active', classLabel === 0);
    document.getElementById('class-b-btn')?.classList.toggle('active', classLabel === 1);
  }

  reset() {
    this.stop();
    this.points = [];
    this.epoch = 0;
    this.initNetwork();
    this.updateStats();
    this.render();
  }

  setupControls() {
    // Toggle button
    const toggleBtn = document.getElementById('db-toggle-btn');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => this.toggle());
    }

    // Reset button
    const resetBtn = document.getElementById('db-reset-btn');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => this.reset());
    }

    // Class selection buttons
    const classABtn = document.getElementById('class-a-btn');
    if (classABtn) {
      classABtn.addEventListener('click', () => this.setClass(0));
      classABtn.classList.add('active');
    }

    const classBBtn = document.getElementById('class-b-btn');
    if (classBBtn) {
      classBBtn.addEventListener('click', () => this.setClass(1));
    }
  }
}


// ============================================
// MONTE CARLO PI ESTIMATOR
// ============================================

class MonteCarloPiGame {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;

    this.ctx = this.canvas.getContext('2d');
    this.setupCanvas();

    // State
    this.reset();

    // Animation
    this.isRunning = false;
    this.animationId = null;
    this.speed = 10; // points per frame

    // Setup controls
    this.setupControls();

    // Initial render
    this.render();
  }

  setupCanvas() {
    const container = this.canvas.parentElement;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    this.canvas.width = Math.min(400, rect.width - 40) * dpr;
    this.canvas.height = 320 * dpr;
    this.canvas.style.width = `${Math.min(400, rect.width - 40)}px`;
    this.canvas.style.height = '320px';
    this.ctx.scale(dpr, dpr);

    this.width = Math.min(400, rect.width - 40);
    this.height = 320;
  }

  reset() {
    this.points = [];
    this.insideCount = 0;
    this.totalCount = 0;
    this.piEstimate = 0;
    this.piHistory = [];
    this.updateStats();
    this.render();
  }

  addPoint() {
    const x = Math.random() * 2 - 1;
    const y = Math.random() * 2 - 1;
    const inside = (x * x + y * y) <= 1;

    this.points.push({ x, y, inside });
    this.totalCount++;
    if (inside) this.insideCount++;

    // Limit stored points for performance
    if (this.points.length > 5000) {
      this.points.shift();
    }

    this.piEstimate = 4 * this.insideCount / this.totalCount;

    // Track history for convergence plot
    if (this.totalCount % 50 === 0) {
      this.piHistory.push(this.piEstimate);
      if (this.piHistory.length > 100) this.piHistory.shift();
    }

    this.updateStats();
  }

  addPoints(n) {
    for (let i = 0; i < n; i++) {
      this.addPoint();
    }
  }

  updateStats() {
    const countEl = document.getElementById('mc-count');
    const piEl = document.getElementById('mc-pi');
    const errorEl = document.getElementById('mc-error');

    if (countEl) countEl.textContent = this.totalCount.toLocaleString();
    if (piEl) piEl.textContent = this.piEstimate.toFixed(6);
    if (errorEl) {
      const error = Math.abs(this.piEstimate - Math.PI);
      errorEl.textContent = error.toFixed(6);
    }
  }

  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    // Clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    // Draw the square and circle
    const padding = 25;
    const squareSize = Math.min(w - padding * 2, h - 75);
    const cx = w / 2;
    const cy = padding + squareSize / 2;

    // Square background
    ctx.fillStyle = 'rgba(30, 41, 59, 0.8)';
    ctx.fillRect(cx - squareSize/2, cy - squareSize/2, squareSize, squareSize);

    // Square border
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    ctx.strokeRect(cx - squareSize/2, cy - squareSize/2, squareSize, squareSize);

    // Circle
    ctx.beginPath();
    ctx.arc(cx, cy, squareSize/2, 0, Math.PI * 2);
    ctx.strokeStyle = '#667eea';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = 'rgba(102, 126, 234, 0.1)';
    ctx.fill();

    // Draw points
    for (const point of this.points) {
      const sx = cx + point.x * (squareSize / 2);
      const sy = cy - point.y * (squareSize / 2);

      ctx.beginPath();
      ctx.arc(sx, sy, 2, 0, Math.PI * 2);
      ctx.fillStyle = point.inside ? '#22c55e' : '#ef4444';
      ctx.fill();
    }

    // Draw formula and explanation
    const formulaY = cy + squareSize/2 + 22;
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '11px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('π ≈ 4 × (points in circle) / (total points)', cx, formulaY);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter, sans-serif';
    ctx.fillText(`π ≈ 4 × ${this.insideCount} / ${this.totalCount} = ${this.piEstimate.toFixed(4)}`, cx, formulaY + 15);

    // Legend
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';

    ctx.fillStyle = '#22c55e';
    ctx.beginPath();
    ctx.arc(cx - 55, formulaY + 32, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Inside', cx - 47, formulaY + 36);

    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(cx + 15, formulaY + 32, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Outside', cx + 23, formulaY + 36);
  }

  animate() {
    if (!this.isRunning) return;

    this.addPoints(this.speed);
    this.render();

    // Auto-stop when error is at or below 0.002
    const error = Math.abs(this.piEstimate - Math.PI);
    if (this.totalCount > 0 && error <= 0.002) {
      this.stop();
      return;
    }

    this.animationId = requestAnimationFrame(() => this.animate());
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.animate();
    this.updateButton();
  }

  stop() {
    this.isRunning = false;
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
    this.updateButton();
  }

  toggle() {
    if (this.isRunning) {
      this.stop();
    } else {
      this.start();
    }
  }

  updateButton() {
    const btn = document.getElementById('mc-toggle-btn');
    if (btn) {
      btn.innerHTML = this.isRunning
        ? '<i data-lucide="pause"></i> Pause'
        : '<i data-lucide="play"></i> Run';
      if (typeof lucide !== 'undefined') lucide.createIcons();
    }
  }

  setSpeed(value) {
    this.speed = value;
    const labels = ['Slow', 'Medium', 'Fast', 'Very Fast'];
    const speedLabel = document.getElementById('speed-label');
    if (speedLabel) speedLabel.textContent = labels[value - 1] || 'Medium';
  }

  addBatch(n) {
    this.addPoints(n);
    this.render();
  }

  setupControls() {
    // Toggle button
    const toggleBtn = document.getElementById('mc-toggle-btn');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => this.toggle());
    }

    // Add 100 button
    const add100Btn = document.getElementById('mc-add100-btn');
    if (add100Btn) {
      add100Btn.addEventListener('click', () => this.addBatch(100));
    }

    // Add 1000 button
    const add1000Btn = document.getElementById('mc-add1000-btn');
    if (add1000Btn) {
      add1000Btn.addEventListener('click', () => this.addBatch(1000));
    }

    // Reset button
    const resetBtn = document.getElementById('mc-reset-btn');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        this.stop();
        this.reset();
      });
    }

    // Speed slider
    const speedSlider = document.getElementById('speed-slider');
    if (speedSlider) {
      speedSlider.addEventListener('input', (e) => {
        const speeds = [1, 10, 50, 200];
        this.setSpeed(speeds[parseInt(e.target.value) - 1]);
      });
    }
  }
}


// ============================================
// INITIALIZATION
// ============================================

document.addEventListener('DOMContentLoaded', () => {
  // Initialize Decision Boundary Painter
  if (document.getElementById('db-canvas')) {
    window.decisionBoundaryGame = new DecisionBoundaryGame('db-canvas');
  }

  // Initialize Monte Carlo Pi Estimator
  if (document.getElementById('mc-canvas')) {
    window.monteCarloPiGame = new MonteCarloPiGame('mc-canvas');
  }
});
