/**
 * About Page Interactive Visualizations
 * - Gradient Descent Visualizer
 * - Monte Carlo Pi Estimator
 */

// ============================================
// GRADIENT DESCENT VISUALIZER
// ============================================

class GradientDescentGame {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;

    this.ctx = this.canvas.getContext('2d');
    this.setupCanvas();

    // Optimization parameters
    this.learningRate = 0.1;
    this.momentum = 0.0;
    this.landscape = 'bowl'; // 'bowl', 'saddle', 'ravine'

    // Ball state
    this.reset();

    // Animation
    this.isRunning = false;
    this.animationId = null;

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
    this.canvas.height = 350 * dpr;
    this.canvas.style.width = `${Math.min(400, rect.width - 40)}px`;
    this.canvas.style.height = '350px';
    this.ctx.scale(dpr, dpr);

    this.width = Math.min(400, rect.width - 40);
    this.height = 350;
  }

  reset() {
    // Random starting position
    this.x = (Math.random() - 0.5) * 3;
    this.y = (Math.random() - 0.5) * 3;
    this.vx = 0;
    this.vy = 0;
    this.path = [{x: this.x, y: this.y}];
    this.iteration = 0;
    this.loss = this.computeLoss(this.x, this.y);
    this.updateStats();
    this.render();
  }

  computeLoss(x, y) {
    switch (this.landscape) {
      case 'bowl':
        return x * x + y * y;
      case 'saddle':
        return x * x - y * y + 0.5 * (x * x + y * y);
      case 'ravine':
        return 0.5 * x * x + 5 * y * y;
      default:
        return x * x + y * y;
    }
  }

  computeGradient(x, y) {
    switch (this.landscape) {
      case 'bowl':
        return { dx: 2 * x, dy: 2 * y };
      case 'saddle':
        return { dx: 2 * x + x, dy: -2 * y + y };
      case 'ravine':
        return { dx: x, dy: 10 * y };
      default:
        return { dx: 2 * x, dy: 2 * y };
    }
  }

  step() {
    const grad = this.computeGradient(this.x, this.y);

    // Momentum update
    this.vx = this.momentum * this.vx - this.learningRate * grad.dx;
    this.vy = this.momentum * this.vy - this.learningRate * grad.dy;

    // Position update
    this.x += this.vx;
    this.y += this.vy;

    // Clamp to bounds
    this.x = Math.max(-4, Math.min(4, this.x));
    this.y = Math.max(-4, Math.min(4, this.y));

    // Record path
    this.path.push({x: this.x, y: this.y});
    if (this.path.length > 200) this.path.shift();

    this.iteration++;
    this.loss = this.computeLoss(this.x, this.y);
    this.updateStats();
  }

  updateStats() {
    const iterEl = document.getElementById('gd-iteration');
    const lossEl = document.getElementById('gd-loss');
    const posEl = document.getElementById('gd-position');

    if (iterEl) iterEl.textContent = this.iteration;
    if (lossEl) lossEl.textContent = this.loss.toFixed(4);
    if (posEl) posEl.textContent = `(${this.x.toFixed(2)}, ${this.y.toFixed(2)})`;
  }

  worldToScreen(wx, wy) {
    const padding = 30;
    const size = Math.min(this.width, this.height) - padding * 2;
    const cx = this.width / 2;
    const cy = this.height / 2;

    return {
      x: cx + (wx / 4) * (size / 2),
      y: cy - (wy / 4) * (size / 2)
    };
  }

  render() {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;

    // Clear
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, w, h);

    // Draw contours
    this.drawContours();

    // Draw path
    this.drawPath();

    // Draw ball
    this.drawBall();

    // Draw minimum marker
    const min = this.worldToScreen(0, 0);
    ctx.beginPath();
    ctx.arc(min.x, min.y, 6, 0, Math.PI * 2);
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.fillStyle = 'rgba(34, 197, 94, 0.3)';
    ctx.fill();

    // Labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Minimum', min.x, min.y + 20);
  }

  drawContours() {
    const ctx = this.ctx;
    const resolution = 50;
    const padding = 30;
    const size = Math.min(this.width, this.height) - padding * 2;
    const startX = (this.width - size) / 2;
    const startY = (this.height - size) / 2;
    const cellSize = size / resolution;

    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const wx = ((i / resolution) - 0.5) * 8;
        const wy = ((j / resolution) - 0.5) * 8;
        const loss = this.computeLoss(wx, wy);

        // Color based on loss
        const maxLoss = 20;
        const t = Math.min(1, loss / maxLoss);

        // Purple to dark gradient
        const r = Math.floor(30 + t * 70);
        const g = Math.floor(20 + t * 20);
        const b = Math.floor(60 + t * 80);

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(startX + i * cellSize, startY + j * cellSize, cellSize + 1, cellSize + 1);
      }
    }

    // Draw contour lines
    const levels = [0.5, 1, 2, 4, 8, 16];
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.3)';
    ctx.lineWidth = 1;

    for (const level of levels) {
      ctx.beginPath();
      for (let angle = 0; angle <= Math.PI * 2; angle += 0.05) {
        let wx, wy;

        if (this.landscape === 'bowl') {
          const r = Math.sqrt(level);
          wx = r * Math.cos(angle);
          wy = r * Math.sin(angle);
        } else if (this.landscape === 'ravine') {
          const a = Math.sqrt(level / 0.5);
          const b = Math.sqrt(level / 5);
          wx = a * Math.cos(angle);
          wy = b * Math.sin(angle);
        } else {
          // Skip for saddle - more complex
          continue;
        }

        const screen = this.worldToScreen(wx, wy);
        if (angle === 0) {
          ctx.moveTo(screen.x, screen.y);
        } else {
          ctx.lineTo(screen.x, screen.y);
        }
      }
      ctx.closePath();
      ctx.stroke();
    }
  }

  drawPath() {
    if (this.path.length < 2) return;

    const ctx = this.ctx;
    ctx.beginPath();

    const start = this.worldToScreen(this.path[0].x, this.path[0].y);
    ctx.moveTo(start.x, start.y);

    for (let i = 1; i < this.path.length; i++) {
      const p = this.worldToScreen(this.path[i].x, this.path[i].y);
      ctx.lineTo(p.x, p.y);
    }

    ctx.strokeStyle = 'rgba(251, 191, 36, 0.8)';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Draw path points
    for (let i = 0; i < this.path.length; i += 3) {
      const p = this.worldToScreen(this.path[i].x, this.path[i].y);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
      ctx.fillStyle = 'rgba(251, 191, 36, 0.6)';
      ctx.fill();
    }
  }

  drawBall() {
    const ctx = this.ctx;
    const pos = this.worldToScreen(this.x, this.y);

    // Glow
    const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 20);
    gradient.addColorStop(0, 'rgba(251, 191, 36, 0.4)');
    gradient.addColorStop(1, 'rgba(251, 191, 36, 0)');
    ctx.fillStyle = gradient;
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);
    ctx.fill();

    // Ball
    ctx.beginPath();
    ctx.arc(pos.x, pos.y, 8, 0, Math.PI * 2);
    ctx.fillStyle = '#fbbf24';
    ctx.fill();
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  animate() {
    if (!this.isRunning) return;

    this.step();
    this.render();

    // Stop if converged
    if (this.loss < 0.0001) {
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
    const btn = document.getElementById('gd-toggle-btn');
    if (btn) {
      btn.innerHTML = this.isRunning
        ? '<i data-lucide="pause"></i> Pause'
        : '<i data-lucide="play"></i> Run';
      if (typeof lucide !== 'undefined') lucide.createIcons();
    }
  }

  singleStep() {
    this.step();
    this.render();
  }

  setLearningRate(value) {
    this.learningRate = value;
    document.getElementById('lr-value').textContent = value.toFixed(2);
  }

  setMomentum(value) {
    this.momentum = value;
    document.getElementById('momentum-value').textContent = value.toFixed(2);
  }

  setLandscape(type) {
    this.landscape = type;
    this.reset();
  }

  setupControls() {
    // Toggle button
    const toggleBtn = document.getElementById('gd-toggle-btn');
    if (toggleBtn) {
      toggleBtn.addEventListener('click', () => this.toggle());
    }

    // Step button
    const stepBtn = document.getElementById('gd-step-btn');
    if (stepBtn) {
      stepBtn.addEventListener('click', () => this.singleStep());
    }

    // Reset button
    const resetBtn = document.getElementById('gd-reset-btn');
    if (resetBtn) {
      resetBtn.addEventListener('click', () => {
        this.stop();
        this.reset();
      });
    }

    // Learning rate slider
    const lrSlider = document.getElementById('lr-slider');
    if (lrSlider) {
      lrSlider.addEventListener('input', (e) => {
        this.setLearningRate(parseFloat(e.target.value));
      });
    }

    // Momentum slider
    const momentumSlider = document.getElementById('momentum-slider');
    if (momentumSlider) {
      momentumSlider.addEventListener('input', (e) => {
        this.setMomentum(parseFloat(e.target.value));
      });
    }

    // Landscape select
    const landscapeSelect = document.getElementById('landscape-select');
    if (landscapeSelect) {
      landscapeSelect.addEventListener('change', (e) => {
        this.setLandscape(e.target.value);
      });
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
    this.canvas.height = 350 * dpr;
    this.canvas.style.width = `${Math.min(400, rect.width - 40)}px`;
    this.canvas.style.height = '350px';
    this.ctx.scale(dpr, dpr);

    this.width = Math.min(400, rect.width - 40);
    this.height = 350;
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
    const padding = 30;
    const squareSize = Math.min(w - padding * 2, h - 80);
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
    const formulaY = cy + squareSize/2 + 25;
    ctx.fillStyle = '#cbd5e1';
    ctx.font = '12px JetBrains Mono, monospace';
    ctx.textAlign = 'center';
    ctx.fillText('π ≈ 4 × (points in circle) / (total points)', cx, formulaY);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText(`π ≈ 4 × ${this.insideCount} / ${this.totalCount} = ${this.piEstimate.toFixed(4)}`, cx, formulaY + 18);

    // Legend
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'left';

    ctx.fillStyle = '#22c55e';
    ctx.beginPath();
    ctx.arc(cx - 60, formulaY + 38, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Inside', cx - 52, formulaY + 42);

    ctx.fillStyle = '#ef4444';
    ctx.beginPath();
    ctx.arc(cx + 20, formulaY + 38, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#94a3b8';
    ctx.fillText('Outside', cx + 28, formulaY + 42);
  }

  animate() {
    if (!this.isRunning) return;

    this.addPoints(this.speed);
    this.render();

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
  // Initialize Gradient Descent Visualizer
  if (document.getElementById('gd-canvas')) {
    window.gradientDescentGame = new GradientDescentGame('gd-canvas');
  }

  // Initialize Monte Carlo Pi Estimator
  if (document.getElementById('mc-canvas')) {
    window.monteCarloPiGame = new MonteCarloPiGame('mc-canvas');
  }
});
