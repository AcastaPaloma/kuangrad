import { run } from './results/run.js';
import { unpack, forward, argmax, frame } from './infer.js';
import { rich } from './format.js';
import './split.js';

const N = 16;
const pad = document.getElementById('pad');
const ctx = pad.getContext('2d');
const guessDigit = document.getElementById('guess-digit');
const guessNote = document.getElementById('guess-note');
const probsList = document.getElementById('probs');
const seen = document.getElementById('seen').getContext('2d');
const pixels = new Uint8Array(N * N); // row-major, 1 = ink: the same 256 values as a dataset.json sample
let layers = null;

// --- the training run --------------------------------------------------------------------------------------------
const count = n => n.toLocaleString('en-US');
const hours = (run.seconds / 3600).toFixed(1);
document.getElementById('summary').innerHTML = rich(`The network from the write-up, ${run.layers.map(([rows]) => rows).join(' → ')} → ${run.layers.at(-1)[1]}, after ${count(run.iterations)} training steps: one drawing per step, learning rate 0.003, ${hours} hours of pure Python on one CPU core. The drawing pad runs its ${count(run.parameters)} trained weights and biases through the same forward pass as \`inference.py\`.`);
document.getElementById('note').innerHTML = rich(`Over the last 200 steps the loss averaged ${run.last200.toFixed(2)}, down from 2.30 (a uniform guess over ten digits). It labels ${count(run.correct)} of the ${count(run.samples)} drawings in \`dataset.json\` correctly, but those are the drawings it trained on, so a digit you draw is the fairer test. It learned my handwriting: a 3 with a flat top works, a 3 with two round bumps usually doesn’t, because none of the 100 3s it saw look like that. It also learned where my digits sit: each input is one fixed cell, and moved two cells over, its own training drawings are right only 42% of the time. So before your drawing goes in, the pad crops it, scales it to 14 rows tall and centres it, the way all of mine are drawn.`);

const movingAverage = (values, window) => {
  let sum = 0;
  return values.map((v, i) => {
    sum += v;
    if (i >= window) sum -= values[i - window];
    return sum / Math.min(i + 1, window);
  });
};

// drawn at the chart's real width, so the tick labels stay at 11px on any screen
const lossChart = (losses, W) => {
  const H = Math.round(Math.min(Math.max(W * .5, 200), 320)), L = 34, R = 8, T = 10, B = 42;
  const top = Math.ceil(Math.max(...losses) * 2) / 2;
  const x = i => L + (W - L - R) * i / Math.max(losses.length - 1, 1);
  const y = v => T + (H - T - B) * (1 - Math.min(v, top) / top);
  const line = values => values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
  const yTicks = Array.from({ length: Math.round(top / 0.5) + 1 }, (_, i) => i * 0.5);
  const step = [500, 1000, 2000, 2500, 5000].find(s => losses.length / s <= Math.max(2, Math.floor(W / 90))) ?? 10000;
  const xTicks = Array.from({ length: Math.floor((losses.length - 1) / step) + 1 }, (_, i) => i * step);
  return `<svg viewBox="0 0 ${W} ${H}" width="${W}" height="${H}" role="img" aria-label="Training loss over ${count(losses.length)} steps, falling from ${losses[0].toFixed(2)} to an average of ${run.last200.toFixed(2)}">
    ${yTicks.map(v => `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text class="tick" x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(1)}</text>`).join('')}
    ${xTicks.map(i => `<text class="tick" x="${x(i)}" y="${H - B + 18}" text-anchor="middle">${count(i)}</text>`).join('')}
    <text class="axis" x="${(L + W - R) / 2}" y="${H - 4}" text-anchor="middle">training step</text>
    <polyline class="raw" points="${line(losses)}"/>
    <polyline class="avg" points="${line(movingAverage(losses, 100))}"/>
  </svg>`;
};
const chart = document.getElementById('loss');
let chartWidth = 0;
new ResizeObserver(() => {
  const width = Math.floor(chart.clientWidth);
  if (width && width !== chartWidth) chart.innerHTML = lossChart(run.losses, (chartWidth = width));
}).observe(chart);

// --- the pad -----------------------------------------------------------------------------------------------------
const draw = () => {
  const cell = pad.width / N;
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, pad.width, pad.height);
  ctx.fillStyle = '#1a1a1a';
  pixels.forEach((on, i) => { if (on) ctx.fillRect((i % N) * cell, Math.floor(i / N) * cell, cell, cell); });
  ctx.strokeStyle = '#e4e3de';
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 1; k < N; k++) {
    ctx.moveTo(k * cell + .5, 0); ctx.lineTo(k * cell + .5, pad.height);
    ctx.moveTo(0, k * cell + .5); ctx.lineTo(pad.width, k * cell + .5);
  }
  ctx.stroke();
};

probsList.innerHTML = Array.from({ length: 10 }, (_, d) => `<li><span class="prob-digit">${d}</span><span class="prob-track"><span class="prob-bar"></span></span><span class="prob-value"></span></li>`).join('');
const rows = [...probsList.children];
const predict = () => {
  const framed = frame(pixels);
  const cell = seen.canvas.width / N;
  seen.fillStyle = '#fff';
  seen.fillRect(0, 0, seen.canvas.width, seen.canvas.height);
  seen.fillStyle = '#1a1a1a';
  framed.forEach((on, i) => { if (on) seen.fillRect((i % N) * cell, Math.floor(i / N) * cell, cell, cell); });
  if (!layers) return;
  const empty = !pixels.some(Boolean);
  const probs = empty ? Array(10).fill(0) : forward(layers, framed);
  const best = argmax(probs);
  guessDigit.textContent = empty ? '?' : best;
  guessNote.textContent = empty ? 'Draw a digit to see its guess.' : `${(probs[best] * 100).toFixed(1)}% sure`;
  rows.forEach((row, d) => {
    row.classList.toggle('is-best', !empty && d === best);
    row.querySelector('.prob-bar').style.width = `${probs[d] * 100}%`;
    row.querySelector('.prob-value').textContent = empty ? '' : `${(probs[d] * 100).toFixed(1)}%`;
  });
};

// pointer -> cell, filling the cells between two pointer events so a quick stroke has no gaps
const cellAt = event => {
  const box = pad.getBoundingClientRect();
  const col = Math.floor((event.clientX - box.left) / box.width * N);
  const row = Math.floor((event.clientY - box.top) / box.height * N);
  return [Math.min(Math.max(col, 0), N - 1), Math.min(Math.max(row, 0), N - 1)];
};
let stroke = null; // { value, last: [col, row] }
const paintTo = ([col, row]) => {
  const [c0, r0] = stroke.last ?? [col, row];
  const steps = Math.max(Math.abs(col - c0), Math.abs(row - r0), 1);
  let changed = false;
  for (let s = 0; s <= steps; s++) {
    const i = Math.round(r0 + (row - r0) * s / steps) * N + Math.round(c0 + (col - c0) * s / steps);
    if (pixels[i] !== stroke.value) { pixels[i] = stroke.value; changed = true; }
  }
  stroke.last = [col, row];
  if (changed) { draw(); predict(); }
};
pad.addEventListener('pointerdown', event => {
  event.preventDefault();
  pad.setPointerCapture(event.pointerId);
  stroke = { value: event.button === 2 || event.shiftKey ? 0 : 1, last: null };
  paintTo(cellAt(event));
});
pad.addEventListener('pointermove', event => { if (stroke) paintTo(cellAt(event)); });
for (const type of ['pointerup', 'pointercancel']) pad.addEventListener(type, () => { stroke = null; });
pad.addEventListener('contextmenu', event => event.preventDefault());
document.getElementById('clear').addEventListener('click', () => { pixels.fill(0); draw(); predict(); });

draw();
fetch(new URL('./results/model.bin', import.meta.url))
  .then(response => { if (!response.ok) throw new Error(response.status); return response.arrayBuffer(); })
  .then(buffer => { layers = unpack(new Float32Array(buffer), run.layers); predict(); })
  .catch(() => { guessNote.textContent = 'Could not load the weights. Try reloading the page.'; });
