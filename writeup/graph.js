// The computation graph kuangrad builds for one drawing. Each circle is a Neuron, shaded by its value. Between dense
// layers sits the weight matrix as a grid of weight Neurons: row i belongs to input i, column j to output j. Only a few
// rows of each layer are drawn. Labels are HTML overlays typeset with KaTeX and sized in container units, so they scale
// with the SVG; the yellow boxes are overlays in the same coordinate space, so they glide between regions with CSS.
// In the forward and backward modes the graph plays one pass, stage by stage, and reports each stage to onStage.
import { tex, rich } from './format.js';

const W = 720, H = 346, NS = 'http://www.w3.org/2000/svg';
const R = 14, PITCH = 34, GAP = 18, CY = 184, CELL = 11, THIN = 5, FAN_IN = 24, FAN_OUT = 26, BRACKET = 4;
const STAGE = 2000, HOLD = 2800;

// Values are illustrative but shaped like a real pass on a drawn 2: x₇ and x₈ are inked pixels, about half of the
// hidden units are dead, p puts 70% on the label, and the gradient through a dead unit is 0.
const COLUMNS = [
  { id: 'x', letter: 'x', name: 'x', size: '256', slots: [0, 1, null, 7, 8, null, 255], depth: 7, scale: 1,
    values: [0, 0, 1, 1, 0], grads: [0.01, -0.02, 0.03, -0.01, 0.02] },
  { id: 'z1', letter: 'z', name: 'z_1', size: '512', slots: [0, 1, 2, 3, null, 511], depth: 6, scale: 0.8,
    values: [0.42, -0.31, 0.77, -0.18, 0.25], grads: [0.05, 0, 0.09, 0, -0.04] },
  { id: 'a1', letter: 'a', name: 'a_1', size: '512', slots: [0, 1, 2, 3, null, 511], depth: 5, scale: 0.8,
    values: [0.42, 0, 0.77, 0, 0.25], grads: [0.05, -0.11, 0.09, 0.07, -0.04] },
  { id: 'z2', letter: 'z', name: 'z_2', size: '256', slots: [0, 1, 2, null, 254, 255], depth: 4, scale: 0.8,
    values: [-0.22, 0.55, 0.18, 0.9, -0.4], grads: [0, -0.08, 0.2, -0.15, 0] },
  { id: 'a2', letter: 'a', name: 'a_2', size: '256', slots: [0, 1, 2, null, 254, 255], depth: 3, scale: 0.8,
    values: [0, 0.55, 0.18, 0.9, 0], grads: [0.12, -0.08, 0.2, -0.15, 0.06] },
  { id: 'l', letter: 'l', name: 'l', size: '10', slots: [0, 1, 2, null, 8, 9], depth: 2, scale: 2,
    values: [-0.5, 0.1, 1.9, 0.2, -0.3], grads: [0.02, 0.03, -0.3, 0.03, 0.02] },
  { id: 'p', letter: 'p', name: 'p', size: '10', slots: [0, 1, 2, null, 8, 9], depth: 1, scale: 0.7,
    values: [0.02, 0.03, 0.7, 0.03, 0.02], grads: [0, 0, -1.43, 0, 0] },
  { id: 'L', letter: 'L', name: 'L', size: '1', slots: [0], depth: 0, scale: 0.5, values: [0.36], grads: [1] }
];
// LINKS[i] connects column i to column i + 1.
const LINKS = [
  { kind: 'dense', op: 'multi', matrix: { id: 'w1', name: 'W_1', dims: '256 \\times 512', depth: 7 } },
  { kind: 'relu', op: 'relu' },
  { kind: 'dense', op: 'multi', matrix: { id: 'w2', name: 'W_2', dims: '512 \\times 256', depth: 5 } },
  { kind: 'relu', op: 'relu' },
  { kind: 'dense', op: 'multi', matrix: { id: 'w3', name: 'W_3', dims: '256 \\times 10', depth: 3 } },
  { kind: 'all', op: 'sftmx' },
  { kind: 'loss', op: 'inverse_log_max' }
];
const LABEL = 2; // the drawing in this example is a 2
const CHOSEN = { x: 7, z: 3 }; // the weight w₇,₃ drawn as "one edge"

// --- layout ------------------------------------------------------------------------------------------------------
const extent = slots => slots.reduce((sum, slot) => sum + (slot === null ? THIN : CELL), 0);
const centers = (slots, start, full, thin) => {
  let y = start;
  return slots.map(slot => { const size = slot === null ? thin : full; const c = y + size / 2; y += size; return c; });
};
const X = [];
{
  let cursor = 10;
  COLUMNS.forEach((column, i) => {
    column.r = column.id === 'L' ? R + 2 : R;
    X[i] = cursor + column.r;
    cursor = X[i] + column.r;
    const link = LINKS[i];
    if (!link) return;
    link.from = i;
    if (link.kind === 'dense') {
      const m = link.matrix;
      m.w = extent(COLUMNS[i + 1].slots);
      m.h = extent(column.slots);
      m.x = cursor + FAN_IN;
      m.y = CY - m.h / 2;
      m.rows = centers(column.slots, m.y, CELL, THIN);
      m.cols = centers(COLUMNS[i + 1].slots, m.x, CELL, THIN);
      cursor = m.x + m.w + FAN_OUT;
    } else cursor += link.kind === 'relu' ? 30 : 40;
  });
}
COLUMNS.forEach(column => { column.ys = centers(column.slots, CY - column.slots.reduce((s, slot) => s + (slot === null ? GAP : PITCH), 0) / 2, PITCH, GAP); });
const MATRICES = LINKS.filter(link => link.matrix).map(link => link.matrix);
LINKS.forEach((link, i) => { if (link.matrix) link.matrix.link = i; });

const TOP = 40, BOTTOM = 314;
const span = (from, to) => ({ x: X[from] - 24, y: TOP, w: X[to] - X[from] + 48, h: BOTTOM - TOP });
const matrixBox = m => ({ x: m.x - 12, y: m.y - 30, w: m.w + 24, h: m.h + 50 });
const nodeY = (c, slot) => COLUMNS[c].ys[COLUMNS[c].slots.indexOf(slot)];

export const REGIONS = {
  all: { x: 4, y: 8, w: 712, h: 332 },
  x: span(0, 0),
  neuron: { x: X[0] - R - 6, y: nodeY(0, CHOSEN.x) - R - 6, w: 2 * R + 12, h: 2 * R + 12 },
  edge: (() => {
    const m = LINKS[0].matrix, y1 = nodeY(0, CHOSEN.x), y2 = nodeY(1, CHOSEN.z);
    const top = Math.min(y1, y2, m.y) - R - 6, bottom = Math.max(y1, y2, m.y + m.h) + R + 6;
    return { x: X[0] - R - 6, y: top, w: X[1] - X[0] + 2 * R + 12, h: bottom - top };
  })(),
  layer1: span(0, 1), relu1: span(1, 2), layer2: span(2, 3), relu2: span(3, 4), layer3: span(4, 5),
  softmax: span(5, 6), loss: span(6, 7),
  w1: matrixBox(MATRICES[0]), w2: matrixBox(MATRICES[1]), w3: matrixBox(MATRICES[2])
};

// --- captions ----------------------------------------------------------------------------------------------------
// One span per caption: the caption is a flex box, and loose text next to math would lose its spaces.
const line = html => `<span>${html}</span>`;
const KEY = {
  values: '<span><i class="sw sw-pos"></i>positive <i class="sw sw-neg"></i>negative <i class="sw sw-zero"></i>zero</span>',
  cache: '<i class="sw sw-cache"></i>',
  grad: '<i class="sw sw-grad"></i>'
};
const CAPTIONS = {
  '': `${KEY.values}<span class="sep">·</span>${line(rich('one matrix cell per weight Neuron'))}`,
  edge: line(rich('one weight: $x_7 \\cdot w_{7,3}$ flows into $z_3$')),
  relu: `<i class="sw sw-dead"></i>${line(rich('dead units: $z \\le 0$, so $\\pd{a}{z} = 0$'))}`,
  softmax: line(rich('every logit feeds every probability: $10 \\times 10$ local gradients')),
  loss: line(rich('only $p_y$ feeds the loss, $y = 2$ in this example')),
  params: line(rich('the three weight matrices: 265,482 parameter Neurons in the real network')),
  update: `${KEY.grad}${line(rich('parameters change as soon as they are popped'))}`,
  hot: `<i class="sw sw-hot"></i>${line(rich('where `local_grad_to` scans long lists'))}`,
  layer: line(rich('the real $W_1$ is $256 \\times 512$, one weight Neuron per cell')),
  depth: `<span class="key-depth">n</span>${line(rich('distance from $L$: every path to a Neuron has the same length'))}`
};
const STAGES = {
  forward: [
    '`multi` builds $z_1$. Each $x_i$ caches $\\loc{\\pd{z_j}{x_i} = w_{ij}}$, each weight caches $\\loc{x_i}$',
    '`relu` builds $a_1$. Each $z_j$ caches $\\loc{\\pd{a_j}{z_j}}$, 1 or 0',
    '`multi` builds $z_2$. $a_1$ and $W_2$ cache their $\\loc{\\pd{z_2}{\\cdot}}$',
    '`relu` builds $a_2$. $z_2$ caches $\\loc{\\pd{a_2}{z_2}}$, 1 or 0',
    '`multi` builds the logits $l$. $a_2$ and $W_3$ cache $\\loc{\\pd{l}{\\cdot}}$',
    '`sftmx` builds $p$. Each $l_j$ caches ten $\\loc{\\pd{p_i}{l_j}}$',
    '`inverse_log_max` builds $L$. $p_y$ caches $\\loc{\\pd{L}{p_y} = -\\frac{1}{p_y}}$',
    'forward done: every `local_grads` list is full, and no Neuron has a grad yet'
  ],
  backward: [
    '$\\gr{\\pd{L}{p_y}} = \\loc{-\\frac{1}{p_y}} \\cdot \\gr{1}$, the other $p_i$ get 0',
    '$\\gr{\\pd{L}{l}} = \\sum \\loc{\\pd{p}{l}} \\cdot \\gr{\\pd{L}{p}}$',
    '$\\gr{\\pd{L}{a_2}} = \\sum \\loc{\\pd{l}{a_2}} \\cdot \\gr{\\pd{L}{l}}$, and the same for $W_3$',
    '$\\gr{\\pd{L}{z_2}} = \\loc{\\pd{a_2}{z_2}} \\cdot \\gr{\\pd{L}{a_2}}$, 0 through dead units',
    '$\\gr{\\pd{L}{a_1}} = \\sum \\loc{\\pd{z_2}{a_1}} \\cdot \\gr{\\pd{L}{z_2}}$, and the same for $W_2$',
    '$\\gr{\\pd{L}{z_1}} = \\loc{\\pd{a_1}{z_1}} \\cdot \\gr{\\pd{L}{a_1}}$',
    '$\\gr{\\pd{L}{x}} = \\sum \\loc{\\pd{z_1}{x}} \\cdot \\gr{\\pd{L}{z_1}}$, and the same for $W_1$',
    'backward done: every Neuron holds $\\gr{\\pd{L}{\\text{itself}}}$'
  ]
};
const stageCaption = (kind, stage) => (kind === 'forward' ? KEY.cache : KEY.grad) + line(rich(STAGES[kind][stage]));

// --- drawing helpers ---------------------------------------------------------------------------------------------
function el(name, attrs = {}, parent) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  if (parent) parent.append(node);
  return node;
}
function text(parent, x, y, content, cls) {
  const node = el('text', { x, y, class: cls }, parent);
  node.textContent = content;
  return node;
}
// Writes an attribute or class only when it changes, so the animation touches the DOM as little as possible.
const memo = new WeakMap();
function set(node, attr, value) {
  let cache = memo.get(node);
  if (!cache) memo.set(node, cache = {});
  if (cache[attr] === value) return;
  cache[attr] = value;
  node.setAttribute(attr, value);
}
function flag(node, name, on) {
  let cache = memo.get(node);
  if (!cache) memo.set(node, cache = {});
  if (cache['.' + name] === on) return;
  cache['.' + name] = on;
  node.classList.toggle(name, on);
}

const POS = '31,58,138', NEG = '138,59,107';
const shade = (v, scale, amount = 1) => {
  if (!v) return { fill: '#fff', dark: false };
  const alpha = (0.2 + 0.7 * Math.min(1, Math.abs(v) / scale)) * amount;
  return { fill: `rgba(${v > 0 ? POS : NEG},${alpha.toFixed(3)})`, dark: alpha > 0.52 };
};
const noise = (a, b, c) => { const s = Math.sin(a * 12.9898 + b * 78.233 + c * 37.719) * 43758.5453; return (s - Math.floor(s)) * 2 - 1; };

const clamp01 = v => Math.max(0, Math.min(1, v));
const seg = (p, a, b) => clamp01((p - a) / (b - a));
const smooth = p => p * p * (3 - 2 * p);
const bell = (p, a, b) => Math.sin(Math.PI * seg(p, a, b));
const backOut = p => (p <= 0 ? 0 : p >= 1 ? 1 : 1 + 2.70158 * (p - 1) ** 3 + 1.70158 * (p - 1) ** 2);

export function buildGraph(root) {
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'graph-svg', role: 'img', 'aria-label': 'Computation graph for one drawing: x, 256 pixels, times the weight matrix W1 gives z1, 512 units, then relu gives a1; W2 gives z2 and a2, 256 units; W3 gives 10 logits l, softmax gives 10 probabilities p, and the loss L reads p.' });
  const layer = name => el('g', { class: name }, svg);
  const depthG = layer('depths'), edgeG = layer('edges'), flowG = layer('flow'), matrixG = layer('matrices');
  const ringG = layer('rings'), nodeG = layer('nodes'), badgeG = layer('badges'), particleG = layer('particles'), textG = layer('labels');
  const overlay = document.createElement('div');
  overlay.className = 'graph-labels';
  overlay.setAttribute('aria-hidden', 'true');
  const label = (x, y, source, cls, size) => {
    const span = document.createElement('span');
    span.className = `gl ${cls}`;
    span.style.left = `${(x / W) * 100}%`;
    span.style.top = `${(y / H) * 100}%`;
    span.style.setProperty('--fs', size);
    span.innerHTML = tex(source, false, true);
    overlay.append(span);
    return span;
  };
  const badge = (x, y, depth) => {
    const g = el('g', { class: 'depth-badge', style: `--delay:${(7 - depth) * 45}ms` }, depthG);
    el('circle', { cx: x, cy: y, r: 9 }, g);
    text(g, x, y + 3.5, String(depth), 'depth-label');
  };

  // nodes
  const nodes = [];
  COLUMNS.forEach((column, c) => {
    column.nodes = [];
    column.slots.forEach((slot, s) => {
      const x = X[c], y = column.ys[s];
      if (slot === null) {
        for (const d of [-5, 0, 5]) el('circle', { cx: x, cy: y + d, r: 1.2, class: 'ellipsis' }, nodeG);
        return;
      }
      const k = column.nodes.length;
      const node = { c, k, slot, x, y, value: column.values[k], grad: column.grads[k], scale: column.scale };
      node.dead = (column.id === 'a1' || column.id === 'a2') && node.value === 0;
      node.chosen = (column.id === 'x' && slot === CHOSEN.x) || (column.id === 'z1' && slot === CHOSEN.z);
      node.ring = el('circle', { cx: x, cy: y, r: column.r + 3.4, class: 'ring', opacity: 0 }, ringG);
      node.circle = el('circle', { cx: x, cy: y, r: column.r, class: 'node' }, nodeG);
      for (const [cls, on] of [['dead', node.dead], ['chosen', node.chosen], ['py', column.id === 'p' && slot === LABEL]]) if (on) node.circle.classList.add(cls);
      if (column.id === 'x' || column.id === 'a1') node.circle.classList.add('hot');
      node.bx = x + column.r * 0.72;
      node.by = y - column.r * 0.72;
      node.badge = el('rect', { x: -3.6, y: -3.6, width: 7.2, height: 7.2, rx: 1.8, class: 'badge', transform: `translate(${node.bx} ${node.by}) scale(0)` }, badgeG);
      const name = column.id === 'L' ? 'L' : `${column.letter}_{${slot}}`;
      node.label = label(x, y, name, `node-label${node.chosen ? ' chosen' : ''}`, slot > 99 ? 9.4 : 10.6);
      column.nodes.push(node);
      nodes.push(node);
    });
    const max = Math.max(...column.grads.map(Math.abs));
    column.nodes.forEach(node => {
      const t = Math.abs(node.grad) / max;
      node.ringT = node.grad ? 0.35 + 0.65 * t : 0;
      node.ring.setAttribute('stroke-width', (1.3 + 1.7 * t).toFixed(2));
    });
    label(X[c], 62, column.name, 'col-label', 16);
    text(textG, X[c], BOTTOM - 10, column.size, 'size-label');
    badge(X[c], 22, column.depth);
  });

  // edges: parent end first, so (x1, y1) is the parent and (x2, y2) the child
  const edges = [];
  const edge = (i, part, x1, y1, x2, y2, group, classes = []) => {
    const len = Math.hypot(x2 - x1, y2 - y1);
    const e = { link: i, part, x1, y1, x2, y2, len };
    e.base = el('line', { x1, y1, x2, y2, class: classes.join(' ') }, group);
    e.fwd = el('line', { x1, y1, x2, y2, class: 'flow-fwd', 'stroke-dasharray': `${len} ${len}`, 'stroke-dashoffset': len, opacity: 0 }, flowG);
    e.bwd = el('line', { x1: x2, y1: y2, x2: x1, y2: y1, class: 'flow-bwd', 'stroke-dasharray': `${len} ${len}`, 'stroke-dashoffset': len, opacity: 0 }, flowG);
    e.dot = el('circle', { cx: x2, cy: y2, r: 2.4, class: 'particle', opacity: 0 }, particleG);
    edges.push(e);
    return e;
  };
  LINKS.forEach((link, i) => {
    const a = COLUMNS[i], b = COLUMNS[i + 1];
    const group = el('g', { class: `link link-${link.kind}${link.matrix ? ' link-' + link.matrix.id : ''}` }, edgeG);
    if (link.kind === 'dense') {
      const m = link.matrix;
      a.nodes.forEach(node => {
        const row = a.slots.indexOf(node.slot);
        edge(i, 'in', node.x + a.r, node.y, m.x - BRACKET, m.rows[row], group, node.chosen ? ['chosen'] : []);
      });
      b.nodes.forEach(node => {
        const s = b.slots.indexOf(node.slot);
        const y = m.y + 3 + ((s + 0.5) / b.slots.length) * (m.h - 6);
        edge(i, 'out', m.x + m.w + BRACKET, y, node.x - b.r, node.y, group, node.chosen ? ['chosen'] : []);
      });
    } else if (link.kind === 'relu') {
      a.nodes.forEach((node, k) => edge(i, 'direct', node.x + a.r, node.y, b.nodes[k].x - b.r, b.nodes[k].y, group, b.nodes[k].dead ? ['to-dead'] : []));
    } else {
      a.nodes.forEach(from => b.nodes.forEach(to => edge(i, 'direct', from.x + a.r, from.y, to.x - b.r, to.y, group,
        link.kind === 'loss' && from.slot === LABEL ? ['py'] : [])));
    }
    // the last name is long, so it ends at the right edge instead of centring on its link
    const op = text(textG, link.matrix ? link.matrix.x + link.matrix.w / 2 : i === LINKS.length - 1 ? W - 8 : (X[i] + X[i + 1]) / 2, H - 12, link.op, 'op-label');
    if (i === LINKS.length - 1) op.style.textAnchor = 'end';
  });

  // weight matrices: one cell per weight, shaded like the nodes; a teal dot once it caches x_i, an orange frame
  // once it has a gradient
  MATRICES.forEach((m, n) => {
    const a = COLUMNS[m.link], b = COLUMNS[m.link + 1];
    const g = el('g', { class: `matrix matrix-${m.id}` }, matrixG);
    m.g = g;
    el('rect', { x: m.x - 1, y: m.y - 1, width: m.w + 2, height: m.h + 2, class: 'matrix-bg' }, g);
    el('path', { d: `M${m.x - BRACKET + 3} ${m.y - 3}h-3v${m.h + 6}h3M${m.x + m.w + BRACKET - 3} ${m.y - 3}h3v${m.h + 6}h-3`, class: 'bracket' }, g);
    m.cells = [];
    let maxGrad = 0;
    a.nodes.forEach(src => b.nodes.forEach(dst => {
      const row = a.slots.indexOf(src.slot), col = b.slots.indexOf(dst.slot);
      const w = noise(src.slot, dst.slot, n);
      const x = m.cols[col] - CELL / 2 + 0.8, y = m.rows[row] - CELL / 2 + 0.8;
      const chosen = n === 0 && src.slot === CHOSEN.x && dst.slot === CHOSEN.z;
      const cell = { row: src.k, col: dst.k, grad: Math.abs(src.value * dst.grad) };
      cell.rect = el('rect', { x, y, width: CELL - 1.6, height: CELL - 1.6, rx: 1.6, fill: `rgba(${w > 0 ? POS : NEG},${(0.12 + 0.5 * Math.abs(w)).toFixed(3)})`, class: `cell${chosen ? ' chosen' : ''}`, 'stroke-opacity': 0 }, g);
      cell.dot = el('circle', { cx: x + CELL - 3.6, cy: y + 2, r: 1.5, class: 'cell-cache', opacity: 0 }, g);
      cell.mark = el('rect', { x: x + 0.6, y: y + 0.6, width: CELL - 2.8, height: CELL - 2.8, rx: 1.2, class: 'cell-grad', opacity: 0 }, g);
      maxGrad = Math.max(maxGrad, cell.grad);
      m.cells.push(cell);
    }));
    m.cells.forEach(cell => { cell.gradT = cell.grad ? 0.3 + 0.7 * (cell.grad / maxGrad) : 0; });
    label(m.x + m.w / 2, m.y - 14, m.name, 'matrix-label', 14);
    label(m.x + m.w / 2, m.y + m.h + 13, m.dims, 'matrix-dims', 8.6);
    badge(m.x + m.w / 2, 22, m.depth);
  });

  // --- one pass, drawn as a function of time ---------------------------------------------------------------------
  // kind: 'base' (values only), 'forward' and 'backward' (animated, t in ms), 'grads' (backward finished).
  const last = COLUMNS.length - 1;
  function draw(kind, t = Infinity) {
    const progress = backward => LINKS.map((_, i) => clamp01((t - (backward ? last - 1 - i : i) * STAGE) / STAGE));
    const fwd = kind === 'forward' ? progress(false) : LINKS.map(() => 1);
    const bwd = kind === 'backward' ? progress(true) : LINKS.map(() => (kind === 'grads' ? 1 : 0));
    const cached = kind === 'forward' || kind === 'backward' || kind === 'grads';

    COLUMNS.forEach((column, c) => {
      const dense = c > 0 && LINKS[c - 1].kind === 'dense';
      const fill = c === 0 ? 1 : smooth(seg(fwd[c - 1], dense ? 0.55 : 0.4, dense ? 0.72 : 0.6));
      const out = LINKS[c];
      const outDense = out?.kind === 'dense';
      const cache = !cached || !out ? 0 : kind === 'forward' ? backOut(seg(fwd[c], outDense ? 0.88 : 0.8, outDense ? 0.98 : 0.9)) : 1;
      const read = kind === 'backward' && out ? bell(bwd[c], outDense ? 0.5 : 0.3, outDense ? 0.8 : 0.62) : 0;
      const ring = c === last ? (kind === 'backward' || kind === 'grads' ? 1 : 0)
        : out ? smooth(seg(bwd[c], outDense ? 0.6 : 0.46, outDense ? 0.84 : 0.72)) : 0;
      column.nodes.forEach(node => {
        const { fill: color, dark } = shade(node.value, node.scale, fill);
        set(node.circle, 'fill', color);
        flag(node.circle, 'pending', fill < 0.03);
        flag(node.label, 'on-dark', dark);
        flag(node.label, 'pending', fill < 0.03);
        set(node.badge, 'transform', `translate(${node.bx} ${node.by}) scale(${(cache * (1 + 0.4 * read)).toFixed(3)})`);
        flag(node.badge, 'is-read', read > 0.05);
        set(node.ring, 'opacity', (ring * node.ringT).toFixed(3));
      });
    });

    edges.forEach(e => {
      const p = fwd[e.link], pb = bwd[e.link];
      const [a, b] = e.part === 'in' ? [0, 0.3] : e.part === 'out' ? [0.38, 0.62] : [0, 0.5];
      const drawn = kind === 'forward' ? smooth(seg(p, a, b)) : 0;
      set(e.fwd, 'stroke-dashoffset', (e.len * (1 - drawn)).toFixed(1));
      set(e.fwd, 'opacity', kind === 'forward' ? (1 - 0.72 * seg(p, 0.8, 1)).toFixed(3) : 0);
      // the cached derivative travels from the child back to the parent that stores it
      const [ca, cb] = e.part === 'out' ? [0.64, 0.78] : e.part === 'in' ? [0.76, 0.9] : [0.6, 0.8];
      const q = kind === 'forward' ? seg(p, ca, cb) : 0;
      const s = smooth(q);
      set(e.dot, 'cx', (e.x2 + (e.x1 - e.x2) * s).toFixed(1));
      set(e.dot, 'cy', (e.y2 + (e.y1 - e.y2) * s).toFixed(1));
      set(e.dot, 'opacity', q > 0 && q < 1 ? 1 : 0);
      const [ba, bb] = e.part === 'out' ? [0, 0.28] : e.part === 'in' ? [0.36, 0.6] : [0, 0.48];
      const back = kind === 'backward' ? smooth(seg(pb, ba, bb)) : 0;
      set(e.bwd, 'stroke-dashoffset', (e.len * (1 - back)).toFixed(1));
      set(e.bwd, 'opacity', kind === 'backward' ? (1 - 0.7 * seg(pb, 0.82, 1)).toFixed(3) : 0);
    });

    MATRICES.forEach(m => {
      const p = fwd[m.link], pb = bwd[m.link];
      m.cells.forEach(cell => {
        set(cell.rect, 'stroke-opacity', kind === 'forward' ? bell(p, 0.24 + 0.03 * cell.col, 0.42 + 0.03 * cell.col).toFixed(3) : 0);
        const dot = kind === 'forward' ? seg(p, 0.72 + 0.02 * cell.row, 0.78 + 0.02 * cell.row) : cached ? 1 : 0;
        set(cell.dot, 'opacity', dot.toFixed(3));
        const grad = kind === 'backward' ? smooth(seg(pb, 0.22 + 0.03 * (4 - cell.col), 0.38 + 0.03 * (4 - cell.col))) : kind === 'grads' ? 1 : 0;
        set(cell.mark, 'opacity', (grad * cell.gradT).toFixed(3));
      });
    });
  }

  // --- assembly ----------------------------------------------------------------------------------------------------
  const highlights = Array.from({ length: 4 }, () => {
    const box = document.createElement('div');
    box.className = 'hl';
    box.setAttribute('aria-hidden', 'true');
    root.append(box);
    return box;
  });
  const tag = document.createElement('span');
  tag.className = 'hl-tag';
  highlights[0].append(tag);
  const caption = document.createElement('p');
  caption.className = 'graph-caption';
  caption.setAttribute('aria-live', 'polite');
  root.prepend(svg, overlay);
  root.after(caption);

  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
  let listener = () => {};
  let frame = 0;
  let playing = null;
  const stop = () => { cancelAnimationFrame(frame); playing = null; };
  const total = LINKS.length * STAGE + HOLD;
  function play(kind) {
    stop();
    if (reduceMotion.matches) {
      draw(kind, Infinity);
      caption.innerHTML = stageCaption(kind, LINKS.length);
      listener(kind, LINKS.length);
      return;
    }
    playing = { kind, start: performance.now(), stage: -1 };
    const tick = now => {
      let t = now - playing.start;
      if (t >= total) { playing.start = now; t = 0; }
      const stage = Math.min(LINKS.length, Math.floor(t / STAGE));
      if (stage !== playing.stage) {
        playing.stage = stage;
        caption.innerHTML = stageCaption(kind, stage);
        listener(kind, stage);
      }
      draw(kind, t);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
  }

  draw('base');
  return {
    onStage(callback) { listener = callback; },
    setState({ hl = [], mode = '', tag: label = '' }) {
      root.dataset.mode = mode;
      highlights.forEach((box, i) => {
        const region = REGIONS[hl[i]];
        box.classList.toggle('is-on', Boolean(region));
        if (!region) return;
        box.classList.toggle('tag-right', region.x + region.w / 2 > W * 0.6);
        box.style.left = `${(region.x / W) * 100}%`;
        box.style.top = `${(region.y / H) * 100}%`;
        box.style.width = `${(region.w / W) * 100}%`;
        box.style.height = `${(region.h / H) * 100}%`;
      });
      tag.textContent = label;
      tag.hidden = !label;
      if (mode === 'forward' || mode === 'backward') {
        if (playing?.kind !== mode) play(mode);
        return;
      }
      stop();
      listener('', -1);
      draw(mode === 'update' ? 'grads' : 'base');
      caption.innerHTML = CAPTIONS[mode] ?? CAPTIONS[''];
    }
  };
}
