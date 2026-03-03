// Neural Network Mini-Game
// Interactive XOR learning visualization

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

    // Network architecture: 2 inputs, 4 hidden, 1 output
    this.layers = [2, 4, 1];

    // XOR training data
    this.trainingData = [
      { input: [0, 0], target: 0 },
      { input: [0, 1], target: 1 },
      { input: [1, 0], target: 1 },
      { input: [1, 1], target: 0 }
    ];

    this.initializeWeights();
    this.setupEventListeners();
    this.draw();
  }

  initializeWeights() {
    // Initialize weights with small random values
    this.weights = [];
    this.biases = [];

    for (let i = 0; i < this.layers.length - 1; i++) {
      const layerWeights = [];
      const layerBiases = [];

      for (let j = 0; j < this.layers[i + 1]; j++) {
        const neuronWeights = [];
        for (let k = 0; k < this.layers[i]; k++) {
          neuronWeights.push((Math.random() - 0.5) * 2);
        }
        layerWeights.push(neuronWeights);
        layerBiases.push((Math.random() - 0.5) * 0.5);
      }

      this.weights.push(layerWeights);
      this.biases.push(layerBiases);
    }

    // Store activations for visualization
    this.activations = this.layers.map(size => new Array(size).fill(0.5));
  }

  sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
  }

  sigmoidDerivative(x) {
    return x * (1 - x);
  }

  forward(input) {
    this.activations[0] = input;

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
    const learningRate = 0.5;
    let totalLoss = 0;
    let correct = 0;

    for (const data of this.trainingData) {
      const output = this.forward(data.input);
      const error = data.target - output;
      totalLoss += error * error;

      if ((output > 0.5 && data.target === 1) || (output <= 0.5 && data.target === 0)) {
        correct++;
      }

      // Backpropagation (simplified for 3-layer network)
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

    this.updateDisplay();
    this.draw();
  }

  boostNeuron(layerIndex, neuronIndex) {
    if (layerIndex === 0 || layerIndex === this.layers.length - 1) return;

    // Boost weights connected to this neuron
    const layer = layerIndex - 1;
    for (let i = 0; i < this.weights[layer][neuronIndex].length; i++) {
      this.weights[layer][neuronIndex][i] *= 1.2;
    }

    // Visual feedback
    this.activations[layerIndex][neuronIndex] = Math.min(1, this.activations[layerIndex][neuronIndex] + 0.2);
    this.draw();
  }

  reset() {
    this.epoch = 0;
    this.loss = 1.0;
    this.accuracy = 0;
    this.stopAutoTrain();
    this.initializeWeights();
    this.updateDisplay();
    this.draw();
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
    if (btn) btn.innerHTML = '<i data-lucide="pause"></i> Pause';
    lucide.createIcons();

    this.autoTrainInterval = setInterval(() => {
      this.train();
      if (this.accuracy >= 100 || this.epoch >= 1000) {
        this.stopAutoTrain();
      }
    }, 100);
  }

  stopAutoTrain() {
    this.autoTraining = false;
    const btn = document.getElementById('auto-train-btn');
    if (btn) btn.innerHTML = '<i data-lucide="zap"></i> Auto Train';
    lucide.createIcons();

    if (this.autoTrainInterval) {
      clearInterval(this.autoTrainInterval);
      this.autoTrainInterval = null;
    }
  }

  updateDisplay() {
    const epochEl = document.getElementById('epoch-count');
    const lossEl = document.getElementById('loss-value');
    const accEl = document.getElementById('accuracy-value');

    if (epochEl) epochEl.textContent = this.epoch;
    if (lossEl) lossEl.textContent = this.loss.toFixed(3);
    if (accEl) accEl.textContent = this.accuracy.toFixed(0) + '%';
  }

  getNeuronPosition(layer, neuron) {
    const padding = 80;
    const layerSpacing = (this.canvas.width - 2 * padding) / (this.layers.length - 1);
    const x = padding + layer * layerSpacing;

    const maxNeurons = Math.max(...this.layers);
    const neuronSpacing = (this.canvas.height - 2 * padding) / (maxNeurons + 1);
    const layerHeight = this.layers[layer] * neuronSpacing;
    const startY = (this.canvas.height - layerHeight) / 2 + neuronSpacing / 2;
    const y = startY + neuron * neuronSpacing;

    return { x, y };
  }

  draw() {
    const ctx = this.ctx;
    const width = this.canvas.width;
    const height = this.canvas.height;

    // Clear canvas
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(0, 0, width, height);

    // Draw connections
    for (let layer = 0; layer < this.layers.length - 1; layer++) {
      for (let from = 0; from < this.layers[layer]; from++) {
        for (let to = 0; to < this.layers[layer + 1]; to++) {
          const fromPos = this.getNeuronPosition(layer, from);
          const toPos = this.getNeuronPosition(layer + 1, to);

          const weight = this.weights[layer][to][from];
          const alpha = Math.min(Math.abs(weight) / 2, 1);

          ctx.beginPath();
          ctx.moveTo(fromPos.x, fromPos.y);
          ctx.lineTo(toPos.x, toPos.y);

          if (weight > 0) {
            ctx.strokeStyle = `rgba(74, 222, 128, ${alpha})`;
          } else {
            ctx.strokeStyle = `rgba(248, 113, 113, ${alpha})`;
          }
          ctx.lineWidth = Math.abs(weight) * 1.5 + 0.5;
          ctx.stroke();
        }
      }
    }

    // Draw neurons
    for (let layer = 0; layer < this.layers.length; layer++) {
      for (let neuron = 0; neuron < this.layers[layer]; neuron++) {
        const pos = this.getNeuronPosition(layer, neuron);
        const activation = this.activations[layer][neuron];

        // Neuron glow
        const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, 30);
        gradient.addColorStop(0, `rgba(102, 126, 234, ${activation * 0.3})`);
        gradient.addColorStop(1, 'rgba(102, 126, 234, 0)');
        ctx.fillStyle = gradient;
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 30, 0, Math.PI * 2);
        ctx.fill();

        // Neuron circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, 20, 0, Math.PI * 2);

        const intensity = Math.floor(activation * 255);
        ctx.fillStyle = `rgb(${100 + intensity * 0.4}, ${100 + intensity * 0.6}, ${150 + intensity * 0.4})`;
        ctx.fill();

        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Activation value
        ctx.fillStyle = '#fff';
        ctx.font = '10px JetBrains Mono';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(activation.toFixed(2), pos.x, pos.y);
      }
    }

    // Draw layer labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px Inter';
    ctx.textAlign = 'center';

    const labels = ['Input', 'Hidden', 'Output'];
    for (let layer = 0; layer < this.layers.length; layer++) {
      const pos = this.getNeuronPosition(layer, 0);
      ctx.fillText(labels[layer], pos.x, 30);
    }
  }

  setupEventListeners() {
    // Canvas click for boosting neurons
    this.canvas.addEventListener('click', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      // Check if clicked on a hidden neuron
      for (let neuron = 0; neuron < this.layers[1]; neuron++) {
        const pos = this.getNeuronPosition(1, neuron);
        const dist = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
        if (dist < 25) {
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
}

// Initialize game when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('nn-canvas')) {
    window.neuralGame = new NeuralNetworkGame('nn-canvas');
  }
});
