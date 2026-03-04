/**
 * Interactive Skill Tree Visualization
 * A constellation/tech-tree style visualization of skills and their relationships
 */

class SkillTree {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;

    this.ctx = this.canvas.getContext('2d');
    this.nodes = [];
    this.edges = [];
    this.hoveredNode = null;
    this.selectedNode = null;
    this.animationId = null;
    this.pulsePhase = 0;

    // Colors matching the site's design system
    this.colors = {
      primary: '#3b82f6',      // Blue
      secondary: '#8b5cf6',    // Purple
      accent: '#06b6d4',       // Cyan
      neutral: '#64748b',      // Gray
      background: '#0f172a',
      nodeFill: '#1e293b',
      nodeStroke: '#334155',
      text: '#f8fafc',
      textMuted: '#94a3b8',
      connection: 'rgba(100, 116, 139, 0.3)',
      connectionActive: 'rgba(59, 130, 246, 0.6)',
      glow: 'rgba(59, 130, 246, 0.4)'
    };

    this.initSkillData();
    this.setupCanvas();
    this.bindEvents();
    this.animate();
  }

  initSkillData() {
    // Define skill nodes with positions, categories, and proficiency
    const categories = {
      foundation: { color: this.colors.neutral, label: 'Foundation' },
      ml: { color: this.colors.primary, label: 'Machine Learning' },
      quant: { color: this.colors.accent, label: 'Quantitative' },
      programming: { color: this.colors.secondary, label: 'Programming' },
      domain: { color: '#f59e0b', label: 'Domain Expertise' }
    };

    // Node definitions with relative positions (0-1 range)
    // Using padding: nodes positioned between 0.08-0.92 horizontally, 0.10-0.88 vertically
    this.nodes = [
      // Foundation layer (bottom)
      { id: 'math', name: 'Mathematics', category: 'foundation', x: 0.28, y: 0.82, proficiency: 90,
        description: 'Linear algebra, calculus, discrete math' },
      { id: 'stats', name: 'Statistics', category: 'foundation', x: 0.50, y: 0.86, proficiency: 90,
        description: 'Probability theory, statistical inference' },
      { id: 'programming', name: 'Programming', category: 'foundation', x: 0.72, y: 0.82, proficiency: 95,
        description: 'Core programming concepts and algorithms' },

      // Quantitative layer
      { id: 'prob', name: 'Probability', category: 'quant', x: 0.22, y: 0.64, proficiency: 90,
        description: 'Probability distributions, Bayesian methods' },
      { id: 'linalg', name: 'Linear Algebra', category: 'quant', x: 0.42, y: 0.68, proficiency: 85,
        description: 'Matrix operations, eigenvalues, SVD' },
      { id: 'optimization', name: 'Optimization', category: 'quant', x: 0.58, y: 0.64, proficiency: 80,
        description: 'Convex optimization, gradient methods' },

      // Programming layer
      { id: 'python', name: 'Python', category: 'programming', x: 0.78, y: 0.68, proficiency: 95,
        description: 'Primary language for ML and data science' },
      { id: 'numpy', name: 'NumPy/Pandas', category: 'programming', x: 0.88, y: 0.54, proficiency: 90,
        description: 'Data manipulation and numerical computing' },
      { id: 'sklearn', name: 'Scikit-learn', category: 'programming', x: 0.84, y: 0.40, proficiency: 90,
        description: 'Classical ML algorithms and pipelines' },
      { id: 'pytorch', name: 'PyTorch', category: 'programming', x: 0.90, y: 0.28, proficiency: 75,
        description: 'Deep learning framework' },

      // Stochastic methods
      { id: 'stochastic', name: 'Stochastic Modeling', category: 'quant', x: 0.16, y: 0.46, proficiency: 90,
        description: 'Random processes, simulation methods' },
      { id: 'markov', name: 'Markov Chains', category: 'quant', x: 0.10, y: 0.30, proficiency: 85,
        description: 'State transitions, steady-state analysis' },
      { id: 'monte_carlo', name: 'Monte Carlo', category: 'quant', x: 0.26, y: 0.34, proficiency: 90,
        description: 'Simulation and estimation techniques' },

      // ML Core
      { id: 'ml_fundamentals', name: 'ML Fundamentals', category: 'ml', x: 0.50, y: 0.46, proficiency: 88,
        description: 'Supervised/unsupervised learning, evaluation' },
      { id: 'feature_eng', name: 'Feature Engineering', category: 'ml', x: 0.38, y: 0.34, proficiency: 85,
        description: 'Feature extraction and selection' },
      { id: 'trees', name: 'Tree Methods', category: 'ml', x: 0.58, y: 0.30, proficiency: 90,
        description: 'Decision trees, ensemble methods' },
      { id: 'xgboost', name: 'XGBoost', category: 'ml', x: 0.50, y: 0.18, proficiency: 90,
        description: 'Gradient boosting for tabular data' },
      { id: 'dnn', name: 'Deep Learning', category: 'ml', x: 0.68, y: 0.22, proficiency: 80,
        description: 'Neural network architectures' },
      { id: 'timeseries', name: 'Time-Series ML', category: 'ml', x: 0.34, y: 0.20, proficiency: 85,
        description: 'Temporal pattern recognition' },

      // Domain expertise (top)
      { id: 'neuro', name: 'Neuroscience', category: 'domain', x: 0.18, y: 0.14, proficiency: 85,
        description: 'Neural systems and brain function' },
      { id: 'signal', name: 'Signal Processing', category: 'domain', x: 0.34, y: 0.10, proficiency: 85,
        description: 'Filtering, spectral analysis' },
      { id: 'healthcare_ml', name: 'Healthcare ML', category: 'domain', x: 0.50, y: 0.08, proficiency: 80,
        description: 'Clinical prediction models' },
      { id: 'sports', name: 'Sports Analytics', category: 'domain', x: 0.66, y: 0.10, proficiency: 85,
        description: 'Performance modeling, sabermetrics' },
      { id: 'disease', name: 'Disease Modeling', category: 'domain', x: 0.82, y: 0.14, proficiency: 80,
        description: 'Epidemiological simulations' }
    ];

    // Define edges (skill dependencies/relationships)
    this.edges = [
      // Foundation connections
      { from: 'math', to: 'stats' },
      { from: 'math', to: 'linalg' },
      { from: 'stats', to: 'prob' },
      { from: 'programming', to: 'python' },

      // Quant pathways
      { from: 'prob', to: 'stochastic' },
      { from: 'stochastic', to: 'markov' },
      { from: 'stochastic', to: 'monte_carlo' },
      { from: 'linalg', to: 'optimization' },

      // Programming progression
      { from: 'python', to: 'numpy' },
      { from: 'numpy', to: 'sklearn' },
      { from: 'sklearn', to: 'pytorch' },

      // ML core paths
      { from: 'optimization', to: 'ml_fundamentals' },
      { from: 'prob', to: 'ml_fundamentals' },
      { from: 'ml_fundamentals', to: 'feature_eng' },
      { from: 'ml_fundamentals', to: 'trees' },
      { from: 'trees', to: 'xgboost' },
      { from: 'trees', to: 'dnn' },
      { from: 'sklearn', to: 'trees' },
      { from: 'pytorch', to: 'dnn' },
      { from: 'feature_eng', to: 'timeseries' },
      { from: 'monte_carlo', to: 'timeseries' },

      // Domain connections
      { from: 'markov', to: 'neuro' },
      { from: 'timeseries', to: 'signal' },
      { from: 'signal', to: 'neuro' },
      { from: 'xgboost', to: 'healthcare_ml' },
      { from: 'dnn', to: 'healthcare_ml' },
      { from: 'monte_carlo', to: 'sports' },
      { from: 'markov', to: 'sports' },
      { from: 'stochastic', to: 'disease' },
      { from: 'monte_carlo', to: 'disease' }
    ];

    this.categories = categories;
  }

  setupCanvas() {
    const container = this.canvas.parentElement;
    const rect = container.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    // Set canvas size - even larger for better spacing
    const canvasHeight = 640;
    this.canvas.width = rect.width * dpr;
    this.canvas.height = canvasHeight * dpr;
    this.canvas.style.width = rect.width + 'px';
    this.canvas.style.height = canvasHeight + 'px';

    this.ctx.scale(dpr, dpr);
    this.width = rect.width;
    this.height = canvasHeight;

    // Padding to keep nodes away from edges
    const padX = 55;
    const padY = 50;

    // Convert relative positions to actual pixel positions with padding
    this.nodes.forEach(node => {
      node.px = padX + node.x * (this.width - 2 * padX);
      node.py = padY + node.y * (this.height - 2 * padY);
      node.radius = 24 + (node.proficiency / 100) * 7; // Larger nodes
    });
  }

  bindEvents() {
    this.canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
    this.canvas.addEventListener('click', (e) => this.handleClick(e));
    this.canvas.addEventListener('mouseleave', () => {
      this.hoveredNode = null;
    });

    window.addEventListener('resize', () => {
      this.setupCanvas();
    });
  }

  handleMouseMove(e) {
    const rect = this.canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    this.hoveredNode = null;

    for (const node of this.nodes) {
      const dx = x - node.px;
      const dy = y - node.py;
      const dist = Math.sqrt(dx * dx + dy * dy);

      if (dist < node.radius + 5) {
        this.hoveredNode = node;
        this.canvas.style.cursor = 'pointer';
        break;
      }
    }

    if (!this.hoveredNode) {
      this.canvas.style.cursor = 'default';
    }
  }

  handleClick(e) {
    if (this.hoveredNode) {
      this.selectedNode = this.selectedNode === this.hoveredNode ? null : this.hoveredNode;
      this.updateInfoPanel();
    } else {
      this.selectedNode = null;
      this.updateInfoPanel();
    }
  }

  updateInfoPanel() {
    const panel = document.getElementById('skill-info-panel');
    if (!panel) return;

    if (this.selectedNode) {
      const node = this.selectedNode;
      const cat = this.categories[node.category];

      panel.innerHTML = `
        <div class="skill-info-header" style="border-color: ${cat.color}">
          <span class="skill-info-category" style="color: ${cat.color}">${cat.label}</span>
          <h3 class="skill-info-name">${node.name}</h3>
        </div>
        <p class="skill-info-description">${node.description}</p>
        <div class="skill-info-proficiency">
          <span class="skill-info-label">Proficiency</span>
          <div class="skill-info-bar">
            <div class="skill-info-fill" style="width: ${node.proficiency}%; background: ${cat.color}"></div>
          </div>
          <span class="skill-info-value">${node.proficiency}%</span>
        </div>
        <div class="skill-info-connections">
          <span class="skill-info-label">Connected Skills</span>
          <div class="skill-info-tags">
            ${this.getConnectedSkills(node).map(s =>
              `<span class="skill-tag" style="border-color: ${this.categories[s.category].color}">${s.name}</span>`
            ).join('')}
          </div>
        </div>
      `;
      panel.classList.add('visible');
    } else {
      panel.classList.remove('visible');
    }
  }

  getConnectedSkills(node) {
    const connected = [];
    for (const edge of this.edges) {
      if (edge.from === node.id) {
        const target = this.nodes.find(n => n.id === edge.to);
        if (target) connected.push(target);
      }
      if (edge.to === node.id) {
        const source = this.nodes.find(n => n.id === edge.from);
        if (source) connected.push(source);
      }
    }
    return connected;
  }

  animate() {
    this.pulsePhase += 0.02;
    this.render();
    this.animationId = requestAnimationFrame(() => this.animate());
  }

  render() {
    const ctx = this.ctx;

    // Clear canvas
    ctx.fillStyle = this.colors.background;
    ctx.fillRect(0, 0, this.width, this.height);

    // Draw subtle grid pattern
    this.drawGrid();

    // Draw edges
    this.drawEdges();

    // Draw nodes
    this.drawNodes();

    // Draw legend
    this.drawLegend();
  }

  drawGrid() {
    const ctx = this.ctx;
    ctx.strokeStyle = 'rgba(51, 65, 85, 0.15)';
    ctx.lineWidth = 1;

    const gridSize = 40;

    for (let x = 0; x < this.width; x += gridSize) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, this.height);
      ctx.stroke();
    }

    for (let y = 0; y < this.height; y += gridSize) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(this.width, y);
      ctx.stroke();
    }
  }

  drawEdges() {
    const ctx = this.ctx;

    for (const edge of this.edges) {
      const fromNode = this.nodes.find(n => n.id === edge.from);
      const toNode = this.nodes.find(n => n.id === edge.to);

      if (!fromNode || !toNode) continue;

      // Check if this edge connects to hovered or selected node
      const isActive = (this.hoveredNode && (edge.from === this.hoveredNode.id || edge.to === this.hoveredNode.id)) ||
                       (this.selectedNode && (edge.from === this.selectedNode.id || edge.to === this.selectedNode.id));

      ctx.beginPath();
      ctx.moveTo(fromNode.px, fromNode.py);
      ctx.lineTo(toNode.px, toNode.py);

      if (isActive) {
        ctx.strokeStyle = this.colors.connectionActive;
        ctx.lineWidth = 2;

        // Draw animated particles along active edges
        this.drawEdgeParticle(fromNode, toNode);
      } else {
        ctx.strokeStyle = this.colors.connection;
        ctx.lineWidth = 1;
      }

      ctx.stroke();
    }
  }

  drawEdgeParticle(from, to) {
    const ctx = this.ctx;
    const t = (Math.sin(this.pulsePhase * 2) + 1) / 2;
    const x = from.px + (to.px - from.px) * t;
    const y = from.py + (to.py - from.py) * t;

    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fillStyle = this.colors.primary;
    ctx.fill();
  }

  drawNodes() {
    const ctx = this.ctx;

    // Sort nodes so hovered/selected are drawn last (on top)
    const sortedNodes = [...this.nodes].sort((a, b) => {
      if (a === this.selectedNode) return 1;
      if (b === this.selectedNode) return -1;
      if (a === this.hoveredNode) return 1;
      if (b === this.hoveredNode) return -1;
      return 0;
    });

    for (const node of sortedNodes) {
      const isHovered = node === this.hoveredNode;
      const isSelected = node === this.selectedNode;
      const cat = this.categories[node.category];

      // Glow effect for hovered/selected
      if (isHovered || isSelected) {
        const glowSize = isSelected ? 20 : 15;
        const gradient = ctx.createRadialGradient(
          node.px, node.py, node.radius,
          node.px, node.py, node.radius + glowSize
        );
        gradient.addColorStop(0, cat.color + '60');
        gradient.addColorStop(1, cat.color + '00');

        ctx.beginPath();
        ctx.arc(node.px, node.py, node.radius + glowSize, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();
      }

      // Node outer ring (proficiency indicator)
      const ringWidth = 3;
      ctx.beginPath();
      ctx.arc(node.px, node.py, node.radius, 0, Math.PI * 2);
      ctx.strokeStyle = this.colors.nodeStroke;
      ctx.lineWidth = ringWidth;
      ctx.stroke();

      // Proficiency arc
      const startAngle = -Math.PI / 2;
      const endAngle = startAngle + (node.proficiency / 100) * Math.PI * 2;
      ctx.beginPath();
      ctx.arc(node.px, node.py, node.radius, startAngle, endAngle);
      ctx.strokeStyle = cat.color;
      ctx.lineWidth = ringWidth;
      ctx.stroke();

      // Node fill
      ctx.beginPath();
      ctx.arc(node.px, node.py, node.radius - ringWidth, 0, Math.PI * 2);
      ctx.fillStyle = isHovered || isSelected ? '#1e3a5f' : this.colors.nodeFill;
      ctx.fill();

      // Pulsing effect for hovered nodes
      if (isHovered) {
        const pulseRadius = node.radius + Math.sin(this.pulsePhase * 3) * 3;
        ctx.beginPath();
        ctx.arc(node.px, node.py, pulseRadius, 0, Math.PI * 2);
        ctx.strokeStyle = cat.color + '40';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Node label
      ctx.fillStyle = isHovered || isSelected ? this.colors.text : this.colors.textMuted;
      ctx.font = `${isHovered || isSelected ? '600' : '500'} 11px Inter, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      // Word wrap for long names
      const words = node.name.split(' ');
      if (words.length > 1 && node.name.length > 10) {
        ctx.fillText(words[0], node.px, node.py - 6);
        ctx.fillText(words.slice(1).join(' '), node.px, node.py + 6);
      } else {
        ctx.fillText(node.name, node.px, node.py);
      }
    }
  }

  drawLegend() {
    const ctx = this.ctx;
    const padding = 16;
    const itemHeight = 22;
    const startX = padding;
    const startY = this.height - padding - (Object.keys(this.categories).length * itemHeight);

    ctx.font = '500 11px Inter, sans-serif';

    let y = startY;
    for (const [key, cat] of Object.entries(this.categories)) {
      // Category dot
      ctx.beginPath();
      ctx.arc(startX + 6, y + 6, 5, 0, Math.PI * 2);
      ctx.fillStyle = cat.color;
      ctx.fill();

      // Category label
      ctx.fillStyle = this.colors.textMuted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'middle';
      ctx.fillText(cat.label, startX + 18, y + 6);

      y += itemHeight;
    }
  }

  destroy() {
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
    }
  }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
  const tree = new SkillTree('skill-tree-canvas');
});
