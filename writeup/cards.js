// Renders the cards that stack up in the left panel: code listings and the small visuals that sit between them.
const NS = 'http://www.w3.org/2000/svg';
const escapeHtml = value => value.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

// --- Python highlighting -----------------------------------------------------------------------------------------
const KEYWORDS = new Set('def return for in if else elif while not is import from class and or lambda as with pass break continue'.split(' '));
const CONSTANTS = new Set(['None', 'True', 'False']);
const BUILTINS = new Set('range len max min sum list set float int print enumerate abs round open uniform sqrt log randint deque perf_counter isinstance'.split(' '));
const TOKEN = /(#.*$)|("""|''')|(f?"(?:[^"\\]|\\.)*"|f?'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?(?:e-?\d+)?\b)|([A-Za-z_]\w*)|(\s+)|(.)/g;

function highlighter() {
  let inDoc = false;
  let afterDef = false;
  return line => {
    let out = '';
    let rest = line;
    if (inDoc) {
      const end = rest.indexOf('"""');
      if (end < 0) return `<span class="tk-str">${escapeHtml(rest)}</span>`;
      out += `<span class="tk-str">${escapeHtml(rest.slice(0, end + 3))}</span>`;
      rest = rest.slice(end + 3);
      inDoc = false;
    }
    TOKEN.lastIndex = 0;
    let match;
    while ((match = TOKEN.exec(rest))) {
      const [token, comment, triple, string, number, word] = match;
      if (comment) out += `<span class="tk-com">${escapeHtml(comment)}</span>`;
      else if (triple) {
        const close = rest.indexOf(triple, match.index + 3);
        if (close < 0) { out += `<span class="tk-str">${escapeHtml(rest.slice(match.index))}</span>`; inDoc = true; break; }
        out += `<span class="tk-str">${escapeHtml(rest.slice(match.index, close + 3))}</span>`;
        TOKEN.lastIndex = close + 3;
      }
      else if (string) out += `<span class="tk-str">${escapeHtml(string)}</span>`;
      else if (number) out += `<span class="tk-num">${number}</span>`;
      else if (word) {
        let cls = '';
        if (afterDef) cls = 'tk-def';
        else if (KEYWORDS.has(word)) cls = 'tk-kw';
        else if (CONSTANTS.has(word)) cls = 'tk-num';
        else if (word === 'self') cls = 'tk-self';
        else if (BUILTINS.has(word)) cls = 'tk-bi';
        else if (/^[A-Z][a-z]/.test(word)) cls = 'tk-type';
        afterDef = word === 'def' || word === 'class';
        out += cls ? `<span class="${cls}">${word}</span>` : word;
        continue;
      }
      else out += escapeHtml(token);
      if (!/^\s+$/.test(token)) afterDef = false;
    }
    return out;
  };
}

function codeCard(lines, focus = []) {
  const highlight = highlighter();
  const rows = lines.map(line => {
    if (line === null) return '<span class="ln gap"><span class="gutter">⋯</span><span class="src"></span></span>';
    const [n, source] = line;
    return `<span class="ln${focus.includes(n) ? ' focus' : ''}"><span class="gutter">${n}</span><span class="src">${highlight(source) || ' '}</span></span>`;
  });
  return `<pre class="code"><code>${rows.join('')}</code></pre>`;
}

function textCard(lines) {
  const highlight = highlighter();
  return `<pre class="code plain"><code>${lines.map(line => `<span class="ln"><span class="src">${highlight(line) || ' '}</span></span>`).join('')}</code></pre>`;
}

// --- digits ------------------------------------------------------------------------------------------------------
const bits = hex => [...hex].flatMap(c => [...parseInt(c, 16).toString(2).padStart(4, '0')].map(Number));
const pixelPath = (pixels, width = 16) => pixels.map((on, i) => on ? `M${i % width} ${Math.floor(i / width)}h1v1h-1z` : '').join('');

function digitsCard(digits) {
  const wrap = document.createElement('div');
  wrap.className = 'digits';
  wrap.innerHTML = `
    <figure class="digit-big">
      <svg viewBox="0 0 16 16" shape-rendering="crispEdges" aria-hidden="true"><path class="grid-lines" d="${Array.from({ length: 15 }, (_, i) => `M${i + 1} 0v16M0 ${i + 1}h16`).join('')}"/><path class="ink"/></svg>
      <figcaption>label <b></b> · <span></span> of 256 pixels on</figcaption>
    </figure>
    <div class="digit-picks" role="group" aria-label="Choose a sample">${digits.map((hex, label) => `
      <button type="button" data-label="${label}" aria-label="Sample labelled ${label}"><svg viewBox="0 0 16 16" shape-rendering="crispEdges" aria-hidden="true"><path d="${pixelPath(bits(hex))}"/></svg><span>${label}</span></button>`).join('')}
    </div>
    <div class="digit-strip">
      <svg viewBox="0 0 256 1" preserveAspectRatio="none" shape-rendering="crispEdges" aria-hidden="true"><path class="ink"/><path class="row-ticks" d="${Array.from({ length: 15 }, (_, i) => `M${(i + 1) * 16} 0v1`).join('')}"/></svg>
      <p><span>flatten(grid) → x₀ … x₂₅₅</span><span>row 0 · row 1 · … · row 15</span></p>
    </div>`;
  const show = label => {
    const pixels = bits(digits[label]);
    wrap.querySelector('.digit-big .ink').setAttribute('d', pixelPath(pixels));
    wrap.querySelector('.digit-strip .ink').setAttribute('d', pixelPath(pixels, 256));
    wrap.querySelector('figcaption b').textContent = label;
    wrap.querySelector('figcaption span').textContent = pixels.filter(Boolean).length;
    wrap.querySelectorAll('.digit-picks button').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.label) === label)));
  };
  wrap.querySelectorAll('.digit-picks button').forEach(button => {
    const pick = () => show(Number(button.dataset.label));
    button.addEventListener('mouseenter', pick);
    button.addEventListener('focus', pick);
    button.addEventListener('click', pick);
  });
  show(3);
  return wrap;
}

// --- one edge, stored twice --------------------------------------------------------------------------------------
function edgeCard() {
  const wrap = document.createElement('div');
  wrap.className = 'edge-card';
  wrap.innerHTML = `
    <svg viewBox="0 0 460 170" aria-label="x seven and w seven-three both feed z three. The derivative of z three with respect to x seven is w seven-three's value, 0.031; with respect to w seven-three it is x seven's value, 1.0.">
      <defs><marker id="edge-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 8 4 0 8z"/></marker></defs>
      <line x1="84" y1="44" x2="330" y2="85" marker-end="url(#edge-arrow)"/>
      <line x1="84" y1="128" x2="330" y2="91" marker-end="url(#edge-arrow)"/>
      <g class="edge-node"><circle cx="60" cy="40" r="23"/><text x="60" y="45">x₇</text><text class="edge-value" x="60" y="80">value 1.0</text></g>
      <g class="edge-node"><circle cx="60" cy="132" r="23"/><text x="60" y="137">w₇,₃</text><text class="edge-value" x="60" y="168">value 0.031</text></g>
      <g class="edge-node is-child"><circle cx="356" cy="88" r="25"/><text x="356" y="93">z₃</text></g>
      <text class="edge-label" x="200" y="45">∂z₃/∂x₇ = w₇,₃ = 0.031</text>
      <text class="edge-label" x="206" y="134">∂z₃/∂w₇,₃ = x₇ = 1.0</text>
    </svg>
    <div class="edge-lists">
      <div><h4>x₇.local_grads <span>512 entries</span></h4><ol>
        <li>[z₀, x₇, w₇,₀]</li><li>[z₁, x₇, w₇,₁]</li><li>[z₂, x₇, w₇,₂]</li><li class="is-hit">[z₃, x₇, 0.031]</li><li class="more">… 508 more</li></ol></div>
      <div><h4>w₇,₃.local_grads <span>1 entry</span></h4><ol><li class="is-hit">[z₃, w₇,₃, 1.0]</li></ol>
        <h4>z₃.parents <span>513</span></h4><ol class="inline"><li>x₀</li><li>w₀,₃</li><li>…</li><li class="is-hit">x₇</li><li class="is-hit">w₇,₃</li><li>…</li><li>b₃</li></ol></div>
    </div>`;
  return wrap;
}

// --- softmax Jacobian --------------------------------------------------------------------------------------------
const P = [0.02, 0.03, 0.70, 0.05, 0.02, 0.04, 0.03, 0.06, 0.03, 0.02];
const MID = [240, 239, 236], POS = [31, 58, 138], NEG = [180, 35, 24];
const mix = (to, t) => `rgb(${MID.map((m, i) => Math.round(m + (to[i] - m) * t)).join(',')})`;

function jacobianCard() {
  const wrap = document.createElement('div');
  wrap.className = 'jacobian';
  const J = P.map((pi, i) => P.map((pj, j) => pi * ((i === j ? 1 : 0) - pj)));
  const max = Math.max(...J.flat().map(Math.abs));
  const cell = 22, left = 34, top = 74;
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', `0 0 ${left + cell * 10 + 8} ${top + cell * 10 + 8}`);
  svg.setAttribute('aria-label', 'Ten by ten heatmap of the softmax Jacobian. The diagonal is positive, largest at 2,2 (0.21); off-diagonal entries are negative and concentrated in row and column 2.');
  let body = '';
  P.forEach((p, j) => {
    const h = p * 70;
    body += `<rect class="p-bar${j === 2 ? ' is-label' : ''}" x="${left + j * cell + 5}" y="${top - 10 - h}" width="${cell - 10}" height="${Math.max(h, 1)}" rx="2"><title>p${j} = ${p.toFixed(2)}</title></rect>`;
    body += `<text class="axis" x="${left + j * cell + cell / 2}" y="${top - 2}">${j}</text>`;
  });
  body += `<text class="axis-title" x="${left}" y="10">p (bars) · columns: logit j</text>`;
  J.forEach((row, i) => {
    body += `<text class="axis" x="${left - 10}" y="${top + i * cell + cell / 2 + 4}">${i}</text>`;
    row.forEach((v, j) => {
      const t = Math.sqrt(Math.abs(v) / max);
      body += `<rect class="j-cell" data-i="${i}" data-j="${j}" data-v="${v}" x="${left + j * cell + 1}" y="${top + i * cell + 1}" width="${cell - 2}" height="${cell - 2}" rx="2" fill="${mix(v >= 0 ? POS : NEG, t)}"/>`;
    });
  });
  svg.innerHTML = body;
  wrap.append(svg);
  const side = document.createElement('div');
  side.className = 'jacobian-side';
  side.innerHTML = `
    <p class="j-readout" aria-live="polite">Hover a cell</p>
    <div class="j-legend"><span>−</span><i></i><span>+</span></div>
    <p class="j-note">row i: p<sub>i</sub> · column j: logit z<sub>j</sub><br>colour: signed √|value|</p>
    <dl><dt>∂p₂/∂z₂</dt><dd>${J[2][2].toFixed(3)}</dd><dt>∂p₂/∂z₇</dt><dd>${J[2][7].toFixed(3)}</dd><dt>∂p₀/∂z₉</dt><dd>${J[0][9].toFixed(4)}</dd></dl>`;
  wrap.append(side);
  const readout = side.querySelector('.j-readout');
  svg.addEventListener('pointerover', event => {
    const target = event.target.closest('.j-cell');
    if (!target) return;
    svg.querySelectorAll('.j-cell.is-hover').forEach(c => c.classList.remove('is-hover'));
    target.classList.add('is-hover');
    const v = Number(target.dataset.v);
    readout.innerHTML = `∂p<sub>${target.dataset.i}</sub>/∂z<sub>${target.dataset.j}</sub> = ${v < 0 ? '−' : ''}${Math.abs(v).toFixed(4)}`;
  });
  return wrap;
}

// --- BFS order ---------------------------------------------------------------------------------------------------
function bfsCard() {
  const names = ['z₁', 'a₁', 'z₂', 'a₂', 'l', 'p', 'L'];
  const x = i => 36 + i * 66;
  const row = (y, depths, late) => names.map((name, i) => `
      <g class="bfs-node${late === i ? ' is-late' : ''}"><circle cx="${x(i)}" cy="${y}" r="15"/><text x="${x(i)}" y="${y + 5}">${name}</text>
      <text class="bfs-depth" x="${x(i)}" y="${y + 34}">${depths[i]}</text></g>`).join('') +
    names.slice(0, -1).map((_, i) => `<line x1="${x(i) + 15}" y1="${y}" x2="${x(i + 1) - 17}" y2="${y}" marker-end="url(#bfs-arrow)"/>`).join('');
  const wrap = document.createElement('div');
  wrap.className = 'bfs-card';
  wrap.innerHTML = `
    <svg viewBox="0 0 460 250" aria-label="Top: a layered chain z1, a1, z2, a2, l, p, L with BFS depths 6 down to 0; each Neuron's depth equals every path's length. Bottom: the same chain with a skip connection from a1 to l. BFS reaches a1 at depth 3 through the skip, before z2 at depth 4 has passed back its gradient.">
      <defs><marker id="bfs-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 8 4 0 8z"/></marker>
      <marker id="bfs-arrow-hot" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 8 4 0 8z"/></marker></defs>
      <text class="bfs-title" x="4" y="16">as built: BFS level = path length</text>
      ${row(48, [6, 5, 4, 3, 2, 1, 0])}
      <text class="bfs-title" x="4" y="128">with a skip a₁ → l</text>
      <path class="bfs-skip" d="M${x(1)} ${172} C ${x(1) + 30} 120, ${x(4) - 30} 120, ${x(4) - 4} ${158}" marker-end="url(#bfs-arrow-hot)"/>
      ${row(174, [6, 3, 4, 3, 2, 1, 0], 1)}
      <path class="bfs-late" d="M${x(2) - 11} ${185} C ${x(2) - 22} 236, ${x(1) + 22} 236, ${x(1) + 12} ${188}" marker-end="url(#bfs-arrow-hot)"/>
      <text class="bfs-late-label" x="${x(1) + 33}" y="244">z₂ reports after a₁ was popped</text>
    </svg>`;
  return wrap;
}

// --- cost ----------------------------------------------------------------------------------------------------------
function costCard() {
  const bars = [
    ['forward', 1.33, ''],
    ['backward · list scan', 10.03, 'is-hot'],
    ['backward · dict lookup', 0.75, '']
  ];
  const max = Math.max(...bars.map(b => b[1]));
  const facts = [
    ['parameters', '265,482'],
    ['Neurons visited per backward', '267,295'],
    ['local-gradient entries', '531,064'],
    ['comparisons in local_grad_to', '50,743,610'],
    ['10,000 iterations at ~11.4 s', '≈ 31 hours']
  ];
  const wrap = document.createElement('div');
  wrap.className = 'cost-card';
  wrap.innerHTML = `
    <h4>seconds per training step · median of 3</h4>
    <div class="bars">${bars.map(([label, s, cls]) => `
      <div class="bar-row ${cls}" title="${label}: ${s.toFixed(2)} s"><span class="bar-label">${label}</span><span class="bar-track"><span class="bar" style="--w:${(s / max) * 100}%"></span></span><span class="bar-value">${s.toFixed(2)} s</span></div>`).join('')}
    </div>
    <dl class="facts">${facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join('')}</dl>
    <p class="cost-note">Samples 500–502, the same weights for both lookups. All 265,482 parameter gradients came out identical. Forward times ranged from 0.5 to 1.7 s.</p>`;
  return wrap;
}

// --- card shell ----------------------------------------------------------------------------------------------------
export function renderCard(step, { snippets, digits }) {
  const card = step.card;
  const node = document.createElement('article');
  node.className = `card card-${card.type}`;
  node.setAttribute('aria-hidden', 'true');
  let title = card.title || card.file;
  let meta = card.meta || '';
  if (card.type === 'code') {
    const lines = snippets[step.id].filter(Boolean);
    meta = `lines ${lines[0][0]}–${lines[lines.length - 1][0]}`;
  }
  node.innerHTML = `<header class="card-head"><span class="card-title">${escapeHtml(title)}</span><span class="card-meta">${escapeHtml(meta)}</span></header>`;
  const body = document.createElement('div');
  body.className = 'card-body';
  if (card.type === 'code') body.innerHTML = codeCard(snippets[step.id], card.focus);
  else if (card.type === 'text') body.innerHTML = textCard(card.lines);
  else if (card.type === 'digits') body.append(digitsCard(digits));
  else if (card.type === 'edge') body.append(edgeCard());
  else if (card.type === 'jacobian') body.append(jacobianCard());
  else if (card.type === 'bfs') body.append(bfsCard());
  else if (card.type === 'cost') body.append(costCard());
  node.append(body);
  return node;
}
