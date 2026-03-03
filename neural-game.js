// Neural Network Mini-Game
// Interactive XOR learning visualization with decision boundary

class NeuralNetworkGame {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;

    this.ctx = this.canvas.getContext('2d');
    this.epoch = 0;
    this.loss = 1.0;
    this.accuracy = 0;
    this.autoTraining = false;
    this.autoTrainInterval = null;
    this.animationFrame = null;
    this.dataFlowParticles = [];
    this.hoveredNeuron = null;
    this.selectedInput = null;

    // Network architecture: 2 inputs, 4 hidden, 1 output
    this.layers = [2, 4, 1];

    // XOR training data with colors
    this.trainingData = [
      { input: [0, 0], target: 0, color: '#ef4444' },
      { input: [0, 1], target: 1, color: '#22c55e' },
      { input: [1, 0], target: 1, color: '#22c55e' },
      { input: [1, 1], target: 0, color: '#ef4444' }
    ];

    this.initializeWeights();
    this.setupCanvas();
    this.setupEventListeners();
    this.startAnimation();
    this.updatePredictions();
  }

  setupCanvas() {
    // Make canvas responsive
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    const container = this.canvas.parentElement;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // Set display size
    this.canvas.style.width = '100%';
    this.canvas.style.height = '400px';

    // Set actual size in memory
    this.canvas.width = rect.width * dpr;
    this.canvas.height = 400 * dpr;

    // Scale context
    this.ctx.scale(dpr, dpr);

    // Store display dimensions
    this.displayWidth = rect.width;
    this.displayHeight = 400;
  }

  initializeWeights() {
    this.weights = [];
    this.biases = [];

    for (let i = 0; i < this.layers.length - 1; i++) {
      const layerWeights = [];
      const layerBiases = [];

      for (let j = 0; j < this.layers[i + 1]; j++) {
        const neuronWeights = [];
        for (let k = 0; k < this.layers[i]; k++) {
          // Xavier initialization
          const scale = Math.sqrt(2.0 / (this.layers[i] + this.layers[i + 1]));
          neuronWeights.push((Math.random() - 0.5) * 2 * scale);
        }
        layerWeights.push(neuronWeights);
        layerBiases.push((Math.random() - 0.5) * 0.5);
      }

      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
    }

    this.activations = this.layers.map(size => new Array(size).fill(0.5));
    this.predictions = [0.5, 0.5, 0.5, 0.5];
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-Math.max(-500, Math.min(500, x))));
  }

  sigmoidDerivative(x) {
    return x * (1 - x);
  }

  forward(input) {
    this.activations[0] = [...input];

    for (let layer = 1; layer < this.layers.length; layer++) {
      for (let neuron = 0; neuron < this.layers[layer]; neuron++) {
        let sum = this.biases[layer - 1][neuron];
        for (let prevNeuron = 0; prevNeuron < this.layers[layer - 1]; prevNeuron++) {
          sum += this.activations[layer - 1][prevNeuron] * this.weights[layer - 1][neuron][prevNeuron];
        }
        this.activations[layer][neuron] = this.sigmoid(sum);
      }
    }

    return this.activations[this.layers.length - 1][0];
  }

  train() {
    const learningRate = 0.8;
    let totalLoss = 0;
    let correct = 0;

    // Shuffle training data
    const shuffled = [...this.trainingData].sort(() => Math.random() - 0.5);

    for (const data of shuffled) {
      const output = this.forward(data.input);
      const error = data.target - output;
      totalLoss += error * error;

      if ((output > 0.5 && data.target === 1) || (output <= 0.5 && data.target === 0)) {
        correct++;
      }

      // Backpropagation
      const outputDelta = error * this.sigmoidDerivative(output);

      // Hidden to output weights
      for (let h = 0; h < this.layers[1]; h++) {
        this.weights[1][0][h] += learningRate * outputDelta * this.activations[1][h];
      }
      this.biases[1][0] += learningRate * outputDelta;

      // Input to hidden weights
      for (let h = 0; h < this.layers[1]; h++) {
        const hiddenDelta = outputDelta * this.weights[1][0][h] * this.sigmoidDerivative(this.activations[1][h]);
        for (let i = 0; i < this.layers[0]; i++) {
          this.weights[0][h][i] += learningRate * hiddenDelta * this.activations[0][i];
        }
        this.biases[0][h] += learningRate * hiddenDelta;
      }
    }

    this.epoch++;
    this.loss = totalLoss / this.trainingData.length;
    this.accuracy = (correct / this.trainingData.length) * 100;

    this.updatePredictions();
    this.updateDisplay();
    this.spawnDataFlowParticles();
  }

  updatePredictions() {
    this.predictions = this.trainingData.map(data => this.forward(data.input));
  }

  spawnDataFlowParticles() {
    // Create particles flowing through the network
    for (let i = 0; i < 3; i++) {
      setTimeout(() => {
        this.dataFlowParticles.push({
          layer: 0,
          progress: 0,
          neuronFrom: Math.floor(Math.random() * this.layers[0]),
          neuronTo: Math.floor(Math.random() * this.layers[1]),
          color: this.accuracy >= 100 ? '#22c55e' : '#667eea',
          alpha: 1
        });
      }, i * 50);
    }
  }

  reset() {
    this.epoch = 0;
    this.loss = 1.0;
    this.accuracy = 0;
    this.stopAutoTrain();
    this.initializeWeights();
    this.updatePredictions();
    this.updateDisplay();
  }

  toggleAutoTrain() {
    if (this.autoTraining) {
      this.stopAutoTrain();
    } else {
      this.startAutoTrain();
    }
  }

  startAutoTrain() {
    this.autoTraining = true;
    const btn = document.getElementById('auto-train-btn');
    if (btn) {
      btn.innerHTML = '<i data-lucide="pause"></i> Pause';
      btn.classList.add('active');
    }
    if (typeof lucide !== 'undefined') lucide.createIcons();

    this.autoTrainInterval = setInterval(() => {
      this.train();
      if (this.accuracy >= 100 && this.loss < 0.01) {
        this.stopAutoTrain();
        this.celebrateSuccess();
      }
      if (this.epoch >= 2000) {
        this.stopAutoTrain();
      }
    }, 50);
  }

  stopAutoTrain() {
    this.autoTraining = false;
    const btn = document.getElementById('auto-train-btn');
    if (btn) {
      btn.innerHTML = '<i data-lucide="zap"></i> Auto Train';
      btn.classList.remove('active');
    }
    if (typeof lucide !== 'undefined') lucide.createIcons();

    if (this.autoTrainInterval) {
      clearInterval(this.autoTrainInterval);
      this.autoTrainInterval = null;
    }
  }

  celebrateSuccess() {
    // Add celebration particles
    for (let i = 0; i < 20; i++) {
      setTimeout(() => {
        this.dataFlowParticles.push({
          layer: -1, // Special flag for celebration
          x: this.displayWidth / 2,
          y: this.displayHeight / 2,
          vx: (Math.random() - 0.5) * 10,
          vy: (Math.random() - 0.5) * 10,
          color: ['#22c55e', '#667eea', '#f59e0b', '#ec4899'][Math.floor(Math.random() * 4)],
          alpha: 1,
          size: Math.random() * 6 + 2
        });
      }, i * 30);
    }
  }

  updateDisplay() {
    const epochEl = document.getElementById('epoch-count');
    const lossEl = document.getElementById('loss-value');
    const accEl = document.getElementById('accuracy-value');

    if (epochEl) epochEl.textContent = this.epoch;
    if (lossEl) {
      lossEl.textContent = this.loss.toFixed(4);
      lossEl.style.color = this.loss < 0.1 ? '#22c55e' : this.loss < 0.3 ? '#f59e0b' : '#ef4444';
    }
    if (accEl) {
      accEl.textContent = this.accuracy.toFixed(0) + '%';
      accEl.style.color = this.accuracy >= 100 ? '#22c55e' : this.accuracy >= 50 ? '#f59e0b' : '#ef4444';
    }

    // Update prediction indicators
    this.trainingData.forEach((data, i) => {
      const el = document.getElementById(`pred-${i}`);
      if (el) {
        const pred = this.predictions[i];
        const correct = (pred > 0.5 && data.target === 1) || (pred <= 0.5 && data.target === 0);
        el.textContent = pred.toFixed(2);
        el.className = `pred-value ${correct ? 'correct' : 'incorrect'}`;
      }
    });
  }

  getNeuronPosition(layer, neuron) {
    const padding = 100;
    const rightPadding = 220; // Space for XOR table
    const availableWidth = this.displayWidth - padding - rightPadding;
    const layerSpacing = availableWidth / (this.layers.length - 1);
    const x = padding + layer * layerSpacing;

    const neuronSpacing = 70;
    const layerHeight = (this.layers[layer] - 1) * neuronSpacing;
    const startY = (this.displayHeight - layerHeight) / 2;
    const y = startY + neuron * neuronSpacing;

    return { x, y };
  }

  startAnimation() {
    const animate = () => {
      this.draw();
      this.animationFrame = requestAnimationFrame(animate);
    };
    animate();
  }

  draw() {
    const ctx = this.ctx;
    const width = this.displayWidth;
    const height = this.displayHeight;

    // Clear canvas with gradient background
    const bgGradient = ctx.createLinearGradient(0, 0, width, height);
    bgGradient.addColorStop(0, '#0f172a');
    bgGradient.addColorStop(1, '#1e293b');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);

    // Draw grid pattern
    ctx.strokeStyle = 'rgba(71, 85, 105, 0.3)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw connections with gradient based on weight
    for (let layer = 0; layer < this.layers.length - 1; layer++) {
      for (let from = 0; from < this.layers[layer]; from++) {
        for (let to = 0; to < this.layers[layer + 1]; to++) {
          const fromPos = this.getNeuronPosition(layer, from);
          const toPos = this.getNeuronPosition(layer + 1, to);
          const weight = this.weights[layer][to][from];
          const absWeight = Math.abs(weight);
          const alpha = Math.min(absWeight / 2, 0.9);

          // Create gradient for connection
          const gradient = ctx.createLinearGradient(fromPos.x, fromPos.y, toPos.x, toPos.y);
          if (weight > 0) {
            gradient.addColorStop(0, `rgba(74, 222, 128, ${alpha * 0.5})`);
            gradient.addColorStop(1, `rgba(74, 222, 128, ${alpha})`);
          } else {
            gradient.addColorStop(0, `rgba(248, 113, 113, ${alpha * 0.5})`);
            gradient.addColorStop(1, `rgba(248, 113, 113, ${alpha})`);
          }

          ctx.beginPath();
          ctx.moveTo(fromPos.x, fromPos.y);
          ctx.lineTo(toPos.x, toPos.y);
          ctx.strokeStyle = gradient;
          ctx.lineWidth = Math.min(absWeight * 2 + 1, 4);
          ctx.stroke();
        }
      }
    }

    // Update and draw data flow particles
    this.dataFlowParticles = this.dataFlowParticles.filter(p => {
      if (p.layer === -1) {
        // Celebration particle
        p.x += p.vx;
        p.y += p.vy;
        p.vy += 0.2; // gravity
        p.alpha -= 0.02;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color.replace(')', `, ${p.alpha})`).replace('rgb', 'rgba');
        ctx.fill();

        return p.alpha > 0;
      } else {
        // Data flow particle
        p.progress += 0.05;

        if (p.progress >= 1) {
          p.layer++;
          p.progress = 0;
          p.neuronFrom = p.neuronTo;
          p.neuronTo = Math.floor(Math.random() * (this.layers[p.layer + 1] || 1));
        }

        if (p.layer >= this.layers.length - 1) {
          return false;
        }

        const fromPos = this.getNeuronPosition(p.layer, p.neuronFrom);
        const toPos = this.getNeuronPosition(p.layer + 1, p.neuronTo);
        const x = fromPos.x + (toPos.x - fromPos.x) * p.progress;
        const y = fromPos.y + (toPos.y - fromPos.y) * p.progress;

        // Glow effect
        const glow = ctx.createRadialGradient(x, y, 0, x, y, 12);
        glow.addColorStop(0, p.color);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, 12, 0, Math.PI * 2);
        ctx.fill();

        // Core
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();

        return true;
      }
    });

    // Draw neurons
    const labels = ['Input', 'Hidden', 'Output'];
    for (let layer = 0; layer < this.layers.length; layer++) {
      // Draw layer label
      const firstNeuron = this.getNeuronPosition(layer, 0);
      ctx.fillStyle = '#94a3b8';
      ctx.font = 'bold 12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(labels[layer], firstNeuron.x, 25);

      for (let neuron = 0; neuron < this.layers[layer]; neuron++) {
        const pos = this.getNeuronPosition(layer, neuron);
        const activation = this.activations[layer][neuron];
        const isHovered = this.hoveredNeuron?.layer === layer && this.hoveredNeuron?.neuron === neuron;
        const radius = isHovered ? 28 : 24;

        // Outer glow
        const glowRadius = radius + 15;
        const glow = ctx.createRadialGradient(pos.x, pos.y, radius, pos.x, pos.y, glowRadius);
        glow.addColorStop(0, `rgba(102, 126, 234, ${activation * 0.4})`);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, glowRadius, 0, Math.PI * 2);
        ctx.fill();

        // Neuron body
        const bodyGradient = ctx.createRadialGradient(pos.x - 5, pos.y - 5, 0, pos.x, pos.y, radius);
        const intensity = activation;
        bodyGradient.addColorStop(0, `rgb(${120 + intensity * 80}, ${140 + intensity * 60}, ${200 + intensity * 55})`);
        bodyGradient.addColorStop(1, `rgb(${60 + intensity * 40}, ${80 + intensity * 50}, ${140 + intensity * 40})`);

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = bodyGradient;
        ctx.fill();

        // Border
        ctx.strokeStyle = isHovered ? '#a78bfa' : '#667eea';
        ctx.lineWidth = isHovered ? 3 : 2;
        ctx.stroke();

        // Activation value
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 11px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(activation.toFixed(2), pos.x, pos.y);

        // Neuron label (for input/output)
        if (layer === 0) {
          ctx.fillStyle = '#64748b';
          ctx.font = '10px Inter, sans-serif';
          ctx.fillText(neuron === 0 ? 'x₁' : 'x₂', pos.x, pos.y + radius + 15);
        } else if (layer === this.layers.length - 1) {
          ctx.fillStyle = '#64748b';
          ctx.font = '10px Inter, sans-serif';
          ctx.fillText('ŷ', pos.x, pos.y + radius + 15);
        }
      }
    }

    // Draw XOR truth table
    this.drawXORTable(ctx, width, height);

    // Draw decision boundary mini-map
    this.drawDecisionBoundary(ctx, width, height);
  }

  drawXORTable(ctx, width, height) {
    const tableX = width - 190;
    const tableY = 50;
    const cellW = 45;
    const cellH = 32;

    // Table background
    ctx.fillStyle = 'rgba(30, 41, 59, 0.9)';
    ctx.roundRect(tableX - 15, tableY - 35, 180, 175, 12);
    ctx.fill();
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Title
    ctx.fillStyle = '#f1f5f9';
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('XOR Predictions', tableX + 75, tableY - 12);

    // Headers
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'center';
    const headers = ['x₁', 'x₂', 'Target', 'Pred'];
    headers.forEach((h, i) => {
      ctx.fillText(h, tableX + i * cellW + 20, tableY + 10);
    });

    // Data rows
    this.trainingData.forEach((data, row) => {
      const y = tableY + 30 + row * cellH;
      const pred = this.predictions[row];
      const correct = (pred > 0.5 && data.target === 1) || (pred <= 0.5 && data.target === 0);

      // Row highlight
      if (correct) {
        ctx.fillStyle = 'rgba(34, 197, 94, 0.1)';
        ctx.fillRect(tableX - 5, y - 10, 170, cellH);
      }

      // Values
      ctx.fillStyle = '#e2e8f0';
      ctx.font = '12px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(data.input[0].toString(), tableX + 20, y + 5);
      ctx.fillText(data.input[1].toString(), tableX + 20 + cellW, y + 5);

      // Target with color
      ctx.fillStyle = data.target === 1 ? '#4ade80' : '#f87171';
      ctx.fillText(data.target.toString(), tableX + 20 + cellW * 2, y + 5);

      // Prediction
      ctx.fillStyle = correct ? '#4ade80' : '#f87171';
      ctx.fillText(pred.toFixed(2), tableX + 20 + cellW * 3, y + 5);

      // Check mark or X
      ctx.font = '14px sans-serif';
      ctx.fillText(correct ? '✓' : '✗', tableX + 165, y + 5);
    });
  }

  drawDecisionBoundary(ctx, width, height) {
    const mapSize = 100;
    const mapX = width - 190;
    const mapY = height - mapSize - 35;
    const resolution = 20;
    const cellSize = mapSize / resolution;

    // Background
    ctx.fillStyle = 'rgba(30, 41, 59, 0.9)';
    ctx.roundRect(mapX - 15, mapY - 25, 180, mapSize + 45, 12);
    ctx.fill();
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Title
    ctx.fillStyle = '#f1f5f9';
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Decision Boundary', mapX + 75, mapY - 5);

    // Draw decision boundary
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x1 = i / resolution;
        const x2 = j / resolution;
        const pred = this.forward([x1, x2]);

        // Color based on prediction
        const r = Math.floor(248 - pred * 174);
        const g = Math.floor(113 + pred * 109);
        const b = Math.floor(113 + pred * 15);
        ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
        ctx.fillRect(mapX + i * cellSize, mapY + 15 + (resolution - 1 - j) * cellSize, cellSize + 1, cellSize + 1);
      }
    }

    // Draw border
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1;
    ctx.strokeRect(mapX, mapY + 15, mapSize, mapSize);

    // Draw training points
    this.trainingData.forEach(data => {
      const px = mapX + data.input[0] * mapSize;
      const py = mapY + 15 + (1 - data.input[1]) * mapSize;

      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = data.target === 1 ? '#22c55e' : '#ef4444';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('x₁ →', mapX + mapSize / 2, mapY + mapSize + 28);

    ctx.save();
    ctx.translate(mapX - 8, mapY + 15 + mapSize / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('x₂ →', 0, 0);
    ctx.restore();
  }

  setupEventListeners() {
    // Mouse tracking for hover effects
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.displayWidth / rect.width;
      const scaleY = this.displayHeight / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      this.hoveredNeuron = null;
      for (let layer = 0; layer < this.layers.length; layer++) {
        for (let neuron = 0; neuron < this.layers[layer]; neuron++) {
          const pos = this.getNeuronPosition(layer, neuron);
          const dist = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
          if (dist < 30) {
            this.hoveredNeuron = { layer, neuron };
            this.canvas.style.cursor = layer === 1 ? 'pointer' : 'default';
            return;
          }
        }
      }
      this.canvas.style.cursor = 'default';
    });

    // Click to boost hidden neurons
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.displayWidth / rect.width;
      const scaleY = this.displayHeight / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      for (let neuron = 0; neuron < this.layers[1]; neuron++) {
        const pos = this.getNeuronPosition(1, neuron);
        const dist = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
        if (dist < 30) {
          this.boostNeuron(1, neuron);
          break;
        }
      }
    });

    // Button listeners
    const trainBtn = document.getElementById('train-btn');
    const resetBtn = document.getElementById('reset-btn');
    const autoBtn = document.getElementById('auto-train-btn');

    if (trainBtn) trainBtn.addEventListener('click', () => this.train());
    if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    if (autoBtn) autoBtn.addEventListener('click', () => this.toggleAutoTrain());
  }

  boostNeuron(layerIndex, neuronIndex) {
    if (layerIndex === 0 || layerIndex === this.layers.length - 1) return;

    // Randomize weights connected to this neuron
    const layer = layerIndex - 1;
    for (let i = 0; i < this.weights[layer][neuronIndex].length; i++) {
      this.weights[layer][neuronIndex][i] += (Math.random() - 0.5) * 0.5;
    }
    this.biases[layer][neuronIndex] += (Math.random() - 0.5) * 0.2;

    // Spawn particles
    for (let i = 0; i < 5; i++) {
      const pos = this.getNeuronPosition(layerIndex, neuronIndex);
      this.dataFlowParticles.push({
        layer: -1,
        x: pos.x,
        y: pos.y,
        vx: (Math.random() - 0.5) * 6,
        vy: (Math.random() - 0.5) * 6,
        color: '#a78bfa',
        alpha: 1,
        size: Math.random() * 4 + 2
      });
    }

    this.updatePredictions();
    this.updateDisplay();
  }

  destroy() {
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
    }
    if (this.autoTrainInterval) {
      clearInterval(this.autoTrainInterval);
    }
    window.removeEventListener('resize', this.resizeCanvas);
  }
}

// Polyfill for roundRect
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function(x, y, w, h, r) {
    if (w < 2 * r) r = w / 2;
    if (h < 2 * r) r = h / 2;
    this.beginPath();
    this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r);
    this.closePath();
    return this;
  };
}

// Initialize game when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('nn-canvas')) {
    window.neuralGame = new NeuralNetworkGame('nn-canvas');
  }
});
