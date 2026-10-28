// The computation graph kuangrad builds for one drawing, drawn as columns of Neurons. Dense layers are fully
// connected blocks of lines (each line is one weight Neuron), ReLU is one-to-one, softmax is 10 × 10, and the loss
// reads all ten probabilities but only depends on p_y. The yellow boxes are HTML overlays positioned in the
// SVG's coordinate space, so they can glide between regions with plain CSS transitions.
const W = 720, H = 300, R = 4.2, NS = 'http://www.w3.org/2000/svg';
const X = [48, 137, 226, 315, 404, 493, 582, 671];
const COLUMNS = [
  { id: 'x', label: 'x', size: '256', slots: 9, step: 15, gap: 4, depth: 7 },
  { id: 'z1', label: 'z₁', size: '512', slots: 11, step: 15, gap: 5, depth: 6 },
  { id: 'a1', label: 'a₁', size: '512', slots: 11, step: 15, gap: 5, depth: 5, dead: [1, 4, 6, 9] },
  { id: 'z2', label: 'z₂', size: '256', slots: 9, step: 15, gap: 4, depth: 4 },
  { id: 'a2', label: 'a₂', size: '256', slots: 9, step: 15, gap: 4, depth: 3, dead: [2, 5, 7] },
  { id: 'l', label: 'l', size: '10', slots: 10, step: 13, depth: 2 },
  { id: 'p', label: 'p', size: '10', slots: 10, step: 13, depth: 1 },
  { id: 'L', label: 'L', size: '1', slots: 1, step: 0, depth: 0 }
];
// Connection between column i and i + 1, and the operation that creates it.
const LINKS = [
  { kind: 'dense', op: 'W₁x + b₁', band: 'w1' },
  { kind: 'one', op: 'relu' },
  { kind: 'dense', op: 'W₂a₁ + b₂', band: 'w2' },
  { kind: 'one', op: 'relu' },
  { kind: 'dense', op: 'W₃a₂ + b₃', band: 'w3' },
  { kind: 'dense', op: 'softmax', band: 'softmax' },
  { kind: 'dense', op: '−log p_y', band: 'loss' }
];
const LABEL = 2; // the drawing in this example is a 2
const CHOSEN_EDGE = [6, 3]; // x slot → z₁ slot drawn as "one edge"

const slotY = (column, slot) => 150 + (slot - (column.slots - 1) / 2) * column.step;
const dots = column => Array.from({ length: column.slots }, (_, slot) => slot).filter(slot => slot !== column.gap);
const span = (from, to, top = 36, bottom = 284) => ({ x: X[from] - 22, y: top, w: X[to] - X[from] + 44, h: bottom - top });
const band = i => ({ x: X[i] + 9, y: 60, w: X[i + 1] - X[i] - 18, h: 224 });

export const REGIONS = {
  all: { x: 10, y: 10, w: 700, h: 280 },
  x: span(0, 0, 36, 256),
  neuron: { x: X[1] - 12, y: slotY(COLUMNS[1], 2) - 12, w: 24, h: 24 },
  edge: (() => {
    const y1 = slotY(COLUMNS[0], CHOSEN_EDGE[0]), y2 = slotY(COLUMNS[1], CHOSEN_EDGE[1]);
    return { x: X[0] - 13, y: Math.min(y1, y2) - 13, w: X[1] - X[0] + 26, h: Math.abs(y1 - y2) + 26 };
  })(),
  layer1: span(0, 1), relu1: span(1, 2), layer2: span(2, 3), relu2: span(3, 4), layer3: span(4, 5),
  softmax: span(5, 6), loss: span(6, 7),
  w1: band(0), w2: band(2), w3: band(4)
};

const CAPTIONS = {
  forward: '<span class="key key-forward"></span>values flow forward, left to right',
  backward: '<span class="key key-backward"></span>gradients flow back from L · <span class="key key-depth">n</span> BFS depth',
  depth: '<span class="key key-depth">n</span> distance from L: every path to a Neuron has the same length',
  relu: '<span class="key key-dead"></span>dead units: z ≤ 0, local gradient 0',
  loss: '<span class="key key-solid"></span>only p<sub>y</sub> feeds the loss · y = 2 in this example',
  edge: '<span class="key key-solid"></span>one weight: x₇ × w₇,₃ into z₃',
  params: 'the three weight blocks: 265,482 parameter Neurons in the real network',
  update: 'parameters change as soon as they are popped',
  hot: '<span class="key key-hot"></span>where local_grad_to scans long lists',
  softmax: 'every logit feeds every probability: 10 × 10 local gradients',
  layer: 'the real block has 256 × 512 lines, one weight Neuron each'
};

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

export function buildGraph(root) {
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'graph-svg', role: 'img', 'aria-label': 'Computation graph for one drawing: x, 256 pixels, through two hidden layers of 512 and 256 units, to 10 logits, 10 probabilities, and the loss L.' });
  const edges = el('g', { class: 'edges' }, svg);
  const flow = el('g', { class: 'flow', 'aria-hidden': 'true' }, svg);
  const nodes = el('g', { class: 'nodes' }, svg);
  const labels = el('g', { class: 'labels' }, svg);
  const depth = el('g', { class: 'depths', 'aria-hidden': 'true' }, svg);

  LINKS.forEach((link, i) => {
    const a = COLUMNS[i], b = COLUMNS[i + 1];
    const group = el('g', { class: `link link-${link.kind}${link.band ? ' band-' + link.band : ''}` }, edges);
    const pairs = link.kind === 'one' ? dots(a).map(slot => [slot, slot]) : dots(a).flatMap(s => dots(b).map(t => [s, t]));
    pairs.forEach(([s, t], k) => {
      const attrs = { x1: X[i] + R, y1: slotY(a, s), x2: X[i + 1] - R, y2: slotY(b, t) };
      const classes = [];
      if (link.kind === 'one' && b.dead?.includes(t)) classes.push('to-dead');
      if (link.band === 'loss' && s === LABEL) classes.push('py');
      if (i === 0 && s === CHOSEN_EDGE[0] && t === CHOSEN_EDGE[1]) classes.push('chosen');
      el('line', { ...attrs, class: classes.join(' ') }, group);
      if (link.kind === 'one' || k % 2 === 0 || link.band === 'loss') el('line', { ...attrs, class: `flow-line flow-${i}` }, flow);
    });
    const [main, sub] = link.op.split('_');
    const op = text(labels, (X[i] + X[i + 1]) / 2, 276, main, 'op-label');
    if (sub) el('tspan', { dy: 3, 'font-size': '0.8em' }, op).textContent = sub;
  });

  COLUMNS.forEach((column, i) => {
    const group = el('g', { class: `column col-${column.id}` }, nodes);
    dots(column).forEach(slot => {
      const cls = [];
      if (column.dead?.includes(slot)) cls.push('dead');
      if (column.id === 'p' && slot === LABEL) cls.push('py');
      el('circle', { cx: X[i], cy: slotY(column, slot), r: column.id === 'L' ? R + 1.6 : R, class: cls.join(' ') }, group);
    });
    if (column.gap !== undefined) for (const d of [-4, 0, 4]) el('circle', { cx: X[i], cy: slotY(column, column.gap) + d, r: 1.1, class: 'ellipsis' }, group);
    text(labels, X[i], 56, column.label, 'col-label');
    text(labels, X[i], 248, column.size, 'size-label');
    const badge = el('g', { class: 'depth-badge', style: `--delay:${(7 - column.depth) * 45}ms` }, depth);
    el('circle', { cx: X[i], cy: 26, r: 9 }, badge);
    text(badge, X[i], 29.5, String(column.depth), 'depth-label');
  });

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
  root.prepend(svg);
  root.after(caption);

  return {
    setState({ hl = [], mode = '', tag: label = '' }) {
      svg.dataset.mode = mode;
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
      caption.innerHTML = CAPTIONS[mode] || '&nbsp;';
    }
  };
}
