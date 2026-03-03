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
    this.testPoint = null; // For interactive testing
    this.pulsePhase = 0;

    // Network architecture: 2 inputs, 6 hidden, 1 output
    this.layers = [2, 6, 1];

    // XOR training data
    this.trainingData = [
      { input: [0, 0], target: 0 },
      { input: [0, 1], target: 1 },
      { input: [1, 0], target: 1 },
      { input: [1, 1], target: 0 }
    ];

    this.initializeWeights();
    this.setupCanvas();
    this.setupEventListeners();
    this.startAnimation();
    this.updatePredictions();
  }

  setupCanvas() {
    this.resizeCanvas();
    window.addEventListener('resize', () => this.resizeCanvas());
  }

  resizeCanvas() {
    const container = this.canvas.parentElement;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    this.canvas.style.width = '100%';
    this.canvas.style.height = '420px';

    this.canvas.width = rect.width * dpr;
    this.canvas.height = 420 * dpr;

    this.ctx.scale(dpr, dpr);

    this.displayWidth = rect.width;
    this.displayHeight = 420;
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
    const acts = [input.slice()];

    for (let layer = 1; layer < this.layers.length; layer++) {
      const layerActs = [];
      for (let neuron = 0; neuron < this.layers[layer]; neuron++) {
        let sum = this.biases[layer - 1][neuron];
        for (let prevNeuron = 0; prevNeuron < this.layers[layer - 1]; prevNeuron++) {
          sum += acts[layer - 1][prevNeuron] * this.weights[layer - 1][neuron][prevNeuron];
        }
        layerActs.push(this.sigmoid(sum));
      }
      acts.push(layerActs);
    }

    this.activations = acts;
    return acts[acts.length - 1][0];
  }

  train() {
    const learningRate = 1.0;
    let totalLoss = 0;
    let correct = 0;

    const shuffled = [...this.trainingData].sort(() => Math.random() - 0.5);

    for (const data of shuffled) {
      const output = this.forward(data.input);
      const error = data.target - output;
      totalLoss += error * error;

      if ((output > 0.5 && data.target === 1) || (output <= 0.5 && data.target === 0)) {
        correct++;
      }

      const outputDelta = error * this.sigmoidDerivative(output);

      // Hidden to output
      for (let h = 0; h < this.layers[1]; h++) {
        this.weights[1][0][h] += learningRate * outputDelta * this.activations[1][h];
      }
      this.biases[1][0] += learningRate * outputDelta;

      // Input to hidden
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
    for (let i = 0; i < 4; i++) {
      setTimeout(() => {
        this.dataFlowParticles.push({
          layer: 0,
          progress: 0,
          neuronFrom: Math.floor(Math.random() * this.layers[0]),
          neuronTo: Math.floor(Math.random() * this.layers[1]),
          color: this.accuracy >= 100 ? '#22c55e' : '#8b5cf6',
          size: Math.random() * 3 + 3
        });
      }, i * 40);
    }
  }

  reset() {
    this.epoch = 0;
    this.loss = 1.0;
    this.accuracy = 0;
    this.testPoint = null;
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
    }, 40);
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
    for (let i = 0; i < 30; i++) {
      setTimeout(() => {
        const angle = (Math.PI * 2 * i) / 30;
        this.dataFlowParticles.push({
          layer: -1,
          x: this.displayWidth / 2,
          y: this.displayHeight / 2,
          vx: Math.cos(angle) * (3 + Math.random() * 5),
          vy: Math.sin(angle) * (3 + Math.random() * 5),
          color: ['#22c55e', '#8b5cf6', '#f59e0b', '#ec4899', '#06b6d4'][Math.floor(Math.random() * 5)],
          alpha: 1,
          size: Math.random() * 8 + 3
        });
      }, i * 20);
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
  }

  getNeuronPosition(layer, neuron) {
    const padding = 80;
    const rightPadding = 240;
    const availableWidth = this.displayWidth - padding - rightPadding;
    const layerSpacing = availableWidth / (this.layers.length - 1);
    const x = padding + layer * layerSpacing;

    const neuronSpacing = 55;
    const layerHeight = (this.layers[layer] - 1) * neuronSpacing;
    const startY = (this.displayHeight - layerHeight) / 2;
    const y = startY + neuron * neuronSpacing;

    return { x, y };
  }

  startAnimation() {
    const animate = () => {
      this.pulsePhase += 0.03;
      this.draw();
      this.animationFrame = requestAnimationFrame(animate);
    };
    animate();
  }

  draw() {
    const ctx = this.ctx;
    const width = this.displayWidth;
    const height = this.displayHeight;

    // Dark gradient background
    const bgGradient = ctx.createLinearGradient(0, 0, width, height);
    bgGradient.addColorStop(0, '#0c0f1a');
    bgGradient.addColorStop(1, '#1a1f35');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, width, height);

    // Subtle grid
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.08)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 40) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 40) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Draw connections with animated pulse
    for (let layer = 0; layer < this.layers.length - 1; layer++) {
      for (let from = 0; from < this.layers[layer]; from++) {
        for (let to = 0; to < this.layers[layer + 1]; to++) {
          const fromPos = this.getNeuronPosition(layer, from);
          const toPos = this.getNeuronPosition(layer + 1, to);
          const weight = this.weights[layer][to][from];
          const absWeight = Math.abs(weight);

          // Animated pulse along connection
          const pulsePos = (this.pulsePhase + layer * 0.5 + from * 0.2) % 1;

          // Draw connection line
          ctx.beginPath();
          ctx.moveTo(fromPos.x, fromPos.y);
          ctx.lineTo(toPos.x, toPos.y);

          const alpha = Math.min(absWeight / 1.5, 0.8);
          if (weight > 0) {
            ctx.strokeStyle = `rgba(34, 197, 94, ${alpha})`;
          } else {
            ctx.strokeStyle = `rgba(239, 68, 68, ${alpha})`;
          }
          ctx.lineWidth = Math.min(absWeight * 2.5 + 1, 5);
          ctx.stroke();

          // Draw pulse dot
          if (this.autoTraining || this.epoch > 0) {
            const px = fromPos.x + (toPos.x - fromPos.x) * pulsePos;
            const py = fromPos.y + (toPos.y - fromPos.y) * pulsePos;
            const pulseAlpha = Math.sin(pulsePos * Math.PI) * alpha * 0.8;

            ctx.beginPath();
            ctx.arc(px, py, 3, 0, Math.PI * 2);
            ctx.fillStyle = weight > 0 ? `rgba(74, 222, 128, ${pulseAlpha})` : `rgba(248, 113, 113, ${pulseAlpha})`;
            ctx.fill();
          }
        }
      }
    }

    // Update and draw data flow particles
    this.dataFlowParticles = this.dataFlowParticles.filter(p => {
      if (p.layer === -1) {
        p.x += p.vx;
        p.y += p.vy;
        p.vy += 0.15;
        p.alpha -= 0.015;

        ctx.beginPath();
        ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = p.color + Math.floor(p.alpha * 255).toString(16).padStart(2, '0');
        ctx.fill();

        return p.alpha > 0;
      } else {
        p.progress += 0.04;

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

        // Glowing particle
        const glow = ctx.createRadialGradient(x, y, 0, x, y, 15);
        glow.addColorStop(0, p.color);
        glow.addColorStop(0.5, p.color + '60');
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(x, y, 15, 0, Math.PI * 2);
        ctx.fill();

        ctx.beginPath();
        ctx.arc(x, y, p.size, 0, Math.PI * 2);
        ctx.fillStyle = '#fff';
        ctx.fill();

        return true;
      }
    });

    // Draw neurons
    const layerLabels = ['Input', 'Hidden', 'Output'];
    for (let layer = 0; layer < this.layers.length; layer++) {
      const firstNeuron = this.getNeuronPosition(layer, 0);
      ctx.fillStyle = '#64748b';
      ctx.font = 'bold 11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(layerLabels[layer], firstNeuron.x, 20);

      for (let neuron = 0; neuron < this.layers[layer]; neuron++) {
        const pos = this.getNeuronPosition(layer, neuron);
        const activation = this.activations[layer][neuron];
        const isHovered = this.hoveredNeuron?.layer === layer && this.hoveredNeuron?.neuron === neuron;
        const baseRadius = 22;
        const radius = isHovered ? baseRadius + 4 : baseRadius;

        // Outer glow based on activation
        const glowSize = radius + 20;
        const glow = ctx.createRadialGradient(pos.x, pos.y, radius * 0.5, pos.x, pos.y, glowSize);
        const glowColor = layer === this.layers.length - 1
          ? (activation > 0.5 ? '34, 197, 94' : '239, 68, 68')
          : '139, 92, 246';
        glow.addColorStop(0, `rgba(${glowColor}, ${activation * 0.5})`);
        glow.addColorStop(1, 'transparent');
        ctx.fillStyle = glow;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, glowSize, 0, Math.PI * 2);
        ctx.fill();

        // Neuron body with gradient
        const bodyGradient = ctx.createRadialGradient(pos.x - 5, pos.y - 5, 0, pos.x, pos.y, radius);
        bodyGradient.addColorStop(0, '#4c1d95');
        bodyGradient.addColorStop(0.5, '#5b21b6');
        bodyGradient.addColorStop(1, '#3730a3');

        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        ctx.fillStyle = bodyGradient;
        ctx.fill();

        // Activation indicator ring
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius, 0, Math.PI * 2);
        const ringColor = activation > 0.5 ? '#22c55e' : '#8b5cf6';
        ctx.strokeStyle = isHovered ? '#f59e0b' : ringColor;
        ctx.lineWidth = isHovered ? 4 : 3;
        ctx.stroke();

        // Inner activation fill
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, radius - 4, 0, Math.PI * 2 * activation, false);
        ctx.lineTo(pos.x, pos.y);
        ctx.closePath();
        ctx.fillStyle = `rgba(${activation > 0.5 ? '34, 197, 94' : '139, 92, 246'}, 0.3)`;
        ctx.fill();

        // Activation text - always white on dark background
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px JetBrains Mono, monospace';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(activation.toFixed(2), pos.x, pos.y);

        // Labels
        if (layer === 0) {
          ctx.fillStyle = '#94a3b8';
          ctx.font = '11px Inter, sans-serif';
          ctx.fillText(neuron === 0 ? 'x₁' : 'x₂', pos.x, pos.y + radius + 14);
        } else if (layer === this.layers.length - 1) {
          ctx.fillStyle = '#94a3b8';
          ctx.font = '11px Inter, sans-serif';
          ctx.fillText('ŷ', pos.x, pos.y + radius + 14);
        }
      }
    }

    // Draw XOR truth table
    this.drawXORTable(ctx, width, height);

    // Draw interactive decision boundary
    this.drawDecisionBoundary(ctx, width, height);
  }

  drawXORTable(ctx, width, height) {
    const tableX = width - 210;
    const tableY = 35;
    const cellW = 42;
    const cellH = 34;

    // Panel background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    this.roundRect(ctx, tableX - 12, tableY - 25, 195, 185, 12);
    ctx.fill();
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Title
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('XOR Truth Table', tableX + 85, tableY - 5);

    // Headers
    ctx.fillStyle = '#64748b';
    ctx.font = '11px Inter, sans-serif';
    const headers = ['x₁', 'x₂', 'Target', 'Pred'];
    headers.forEach((h, i) => {
      ctx.fillText(h, tableX + i * cellW + 18, tableY + 20);
    });

    // Data rows
    this.trainingData.forEach((data, row) => {
      const y = tableY + 40 + row * cellH;
      const pred = this.predictions[row];
      const correct = (pred > 0.5 && data.target === 1) || (pred <= 0.5 && data.target === 0);

      // Row background
      ctx.fillStyle = correct ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.1)';
      this.roundRect(ctx, tableX - 5, y - 12, 180, cellH - 2, 6);
      ctx.fill();

      // Values
      ctx.fillStyle = '#e2e8f0';
      ctx.font = '12px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(data.input[0].toString(), tableX + 18, y + 5);
      ctx.fillText(data.input[1].toString(), tableX + 18 + cellW, y + 5);

      // Target
      ctx.fillStyle = data.target === 1 ? '#4ade80' : '#f87171';
      ctx.fillText(data.target.toString(), tableX + 18 + cellW * 2, y + 5);

      // Prediction
      ctx.fillStyle = correct ? '#4ade80' : '#f87171';
      ctx.fillText(pred.toFixed(2), tableX + 18 + cellW * 3, y + 5);

      // Status icon
      ctx.font = 'bold 14px sans-serif';
      ctx.fillText(correct ? '✓' : '✗', tableX + 177, y + 5);
    });
  }

  drawDecisionBoundary(ctx, width, height) {
    const mapSize = 120;
    const mapX = width - 210;
    const mapY = height - mapSize - 55;
    const resolution = 24;
    const cellSize = mapSize / resolution;

    // Panel background
    ctx.fillStyle = 'rgba(15, 23, 42, 0.95)';
    this.roundRect(ctx, mapX - 12, mapY - 30, 195, mapSize + 65, 12);
    ctx.fill();
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Title
    ctx.fillStyle = '#f8fafc';
    ctx.font = 'bold 13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Decision Boundary', mapX + 85, mapY - 10);

    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#64748b';
    ctx.fillText('Click to test any point!', mapX + 85, mapY + 5);

    // Draw decision boundary heatmap
    for (let i = 0; i < resolution; i++) {
      for (let j = 0; j < resolution; j++) {
        const x1 = i / (resolution - 1);
        const x2 = j / (resolution - 1);
        const pred = this.forward([x1, x2]);

        // Smooth color gradient
        const r = Math.floor(248 - pred * 214);
        const g = Math.floor(113 + pred * 109);
        const b = Math.floor(113 - pred * 79);

        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(
          mapX + i * cellSize,
          mapY + 18 + (resolution - 1 - j) * cellSize,
          cellSize + 0.5,
          cellSize + 0.5
        );
      }
    }

    // Border
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 2;
    ctx.strokeRect(mapX, mapY + 18, mapSize, mapSize);

    // Training points
    this.trainingData.forEach(data => {
      const px = mapX + data.input[0] * mapSize;
      const py = mapY + 18 + (1 - data.input[1]) * mapSize;

      // Outer ring
      ctx.beginPath();
      ctx.arc(px, py, 10, 0, Math.PI * 2);
      ctx.fillStyle = data.target === 1 ? 'rgba(34, 197, 94, 0.3)' : 'rgba(239, 68, 68, 0.3)';
      ctx.fill();

      // Point
      ctx.beginPath();
      ctx.arc(px, py, 6, 0, Math.PI * 2);
      ctx.fillStyle = data.target === 1 ? '#22c55e' : '#ef4444';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();
    });

    // Test point if exists
    if (this.testPoint) {
      const px = mapX + this.testPoint.x * mapSize;
      const py = mapY + 18 + (1 - this.testPoint.y) * mapSize;
      const pred = this.forward([this.testPoint.x, this.testPoint.y]);

      // Crosshair
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(px - 15, py);
      ctx.lineTo(px + 15, py);
      ctx.moveTo(px, py - 15);
      ctx.lineTo(px, py + 15);
      ctx.stroke();
      ctx.setLineDash([]);

      // Test point
      ctx.beginPath();
      ctx.arc(px, py, 8, 0, Math.PI * 2);
      ctx.fillStyle = '#f59e0b';
      ctx.fill();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.stroke();

      // Prediction label
      ctx.fillStyle = '#f8fafc';
      ctx.font = 'bold 11px JetBrains Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText(`(${this.testPoint.x.toFixed(2)}, ${this.testPoint.y.toFixed(2)}) → ${pred.toFixed(3)}`, mapX + mapSize / 2, mapY + mapSize + 35);
    }

    // Axis labels
    ctx.fillStyle = '#64748b';
    ctx.font = '10px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('0', mapX - 8, mapY + mapSize + 22);
    ctx.fillText('1', mapX + mapSize + 8, mapY + mapSize + 22);
    ctx.fillText('x₁', mapX + mapSize / 2, mapY + mapSize + 50);
  }

  roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
  }

  setupEventListeners() {
    this.canvas.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.displayWidth / rect.width;
      const scaleY = this.displayHeight / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      // Check decision boundary area
      const mapSize = 120;
      const mapX = this.displayWidth - 210;
      const mapY = this.displayHeight - mapSize - 55;

      if (x >= mapX && x <= mapX + mapSize && y >= mapY + 18 && y <= mapY + 18 + mapSize) {
        this.canvas.style.cursor = 'crosshair';
        return;
      }

      // Check neurons
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

    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const scaleX = this.displayWidth / rect.width;
      const scaleY = this.displayHeight / rect.height;
      const x = (e.clientX - rect.left) * scaleX;
      const y = (e.clientY - rect.top) * scaleY;

      // Check decision boundary click
      const mapSize = 120;
      const mapX = this.displayWidth - 210;
      const mapY = this.displayHeight - mapSize - 55;

      if (x >= mapX && x <= mapX + mapSize && y >= mapY + 18 && y <= mapY + 18 + mapSize) {
        this.testPoint = {
          x: (x - mapX) / mapSize,
          y: 1 - (y - mapY - 18) / mapSize
        };
        return;
      }

      // Check hidden neuron click
      for (let neuron = 0; neuron < this.layers[1]; neuron++) {
        const pos = this.getNeuronPosition(1, neuron);
        const dist = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
        if (dist < 30) {
          this.boostNeuron(1, neuron);
          break;
        }
      }
    });

    const trainBtn = document.getElementById('train-btn');
    const resetBtn = document.getElementById('reset-btn');
    const autoBtn = document.getElementById('auto-train-btn');

    if (trainBtn) trainBtn.addEventListener('click', () => this.train());
    if (resetBtn) resetBtn.addEventListener('click', () => this.reset());
    if (autoBtn) autoBtn.addEventListener('click', () => this.toggleAutoTrain());
  }

  boostNeuron(layerIndex, neuronIndex) {
    if (layerIndex === 0 || layerIndex === this.layers.length - 1) return;

    const layer = layerIndex - 1;
    for (let i = 0; i < this.weights[layer][neuronIndex].length; i++) {
      this.weights[layer][neuronIndex][i] += (Math.random() - 0.5) * 0.8;
    }
    this.biases[layer][neuronIndex] += (Math.random() - 0.5) * 0.3;

    // Burst effect
    const pos = this.getNeuronPosition(layerIndex, neuronIndex);
    for (let i = 0; i < 12; i++) {
      const angle = (Math.PI * 2 * i) / 12;
      this.dataFlowParticles.push({
        layer: -1,
        x: pos.x,
        y: pos.y,
        vx: Math.cos(angle) * 4,
        vy: Math.sin(angle) * 4,
        color: '#f59e0b',
        alpha: 1,
        size: Math.random() * 4 + 2
      });
    }

    this.updatePredictions();
    this.updateDisplay();
  }

  destroy() {
    if (this.animationFrame) cancelAnimationFrame(this.animationFrame);
    if (this.autoTrainInterval) clearInterval(this.autoTrainInterval);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('nn-canvas')) {
    window.neuralGame = new NeuralNetworkGame('nn-canvas');
  }
});
