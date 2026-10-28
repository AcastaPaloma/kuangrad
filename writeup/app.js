import { steps, digits } from './content.js';
import { snippets } from './snippets.js';
import { buildGraph } from './graph.js';
import { renderCard } from './cards.js';

const reader = document.getElementById('reader');
const article = document.getElementById('article');
const stack = document.getElementById('stack');
const counter = document.getElementById('step-counter');
const chapterLabel = document.getElementById('chapter-label');
const toggle = document.getElementById('contents-toggle');
const menu = document.getElementById('contents-menu');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

// --- prose -------------------------------------------------------------------------------------------------------
const escapeHtml = value => value.replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
// Single-letter math names such as p_y and w_ij get a subscript; identifiers like local_grads or fan_in do not.
const subscripts = html => html.replace(/(?<![\w])([A-Za-zδ∂])_([a-z0-9]{1,2})(?![\w])/g, '$1<sub>$2</sub>');
const inline = source => source.split(/(`[^`]+`)/).map(part => part.startsWith('`')
  ? `<code>${escapeHtml(part.slice(1, -1))}</code>`
  : subscripts(escapeHtml(part).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\*(.+?)\*/g, '<em>$1</em>').replace(/\^\{(.+?)\}/g, '<sup>$1</sup>'))).join('');
const blockHtml = block => {
  if (typeof block === 'string') return `<p>${inline(block)}</p>`;
  if (block.list) return `<ul>${block.list.map(item => `<li>${inline(item)}</li>`).join('')}</ul>`;
  if (block.math) return `<p class="math">${inline(block.math)}</p>`;
  return '';
};

let chapter = '';
const chapters = steps.map(step => (chapter = step.chapter || chapter));
const blocks = steps.map((step, i) => {
  const section = document.createElement('section');
  section.className = `reading-block level-${step.level}${step.chapter ? ' starts-chapter' : ''}`;
  section.id = step.id;
  section.dataset.index = i;
  const heading = i === 0
    ? `<header class="title-block"><p class="kicker">a write-up</p><h1>${escapeHtml(step.title)}</h1><p class="subtitle">An autograd engine, one Neuron at a time</p><p class="byline">Kuan Yi Wang · <a href="https://github.com/AcastaPaloma/kuangrad" target="_blank" rel="noopener">source on GitHub</a></p></header>`
    : `${step.chapter ? `<p class="chapter-kicker">${escapeHtml(step.chapter)}</p>` : ''}<h${step.level}>${escapeHtml(step.title)}</h${step.level}>`;
  section.innerHTML = heading + step.body.map(blockHtml).join('');
  article.append(section);
  return section;
});
const end = document.createElement('footer');
end.className = 'article-end';
end.innerHTML = 'The code shown on the left is copied from the repository by <code>scripts/build-snippets.mjs</code>, line numbers included. <a href="https://github.com/AcastaPaloma/kuangrad" target="_blank" rel="noopener">Read it on GitHub</a>.';
article.append(end);

// --- contents ----------------------------------------------------------------------------------------------------
const toc = document.getElementById('toc');
toc.innerHTML = steps.map((step, i) => `${step.chapter ? `<p class="toc-chapter">${escapeHtml(step.chapter)}</p>` : ''}
  <button class="toc-link level-${step.level}" type="button" data-index="${i}"><span class="toc-number">${String(i + 1).padStart(2, '0')}</span>${escapeHtml(step.title)}</button>`).join('');
const tocLinks = [...toc.querySelectorAll('.toc-link')];
const setMenu = open => { menu.hidden = !open; toggle.setAttribute('aria-expanded', String(open)); };
toggle.addEventListener('click', () => setMenu(menu.hidden));
document.addEventListener('keydown', event => { if (event.key === 'Escape' && !menu.hidden) { setMenu(false); toggle.focus(); } });
document.addEventListener('pointerdown', event => { if (!menu.hidden && !menu.contains(event.target) && !toggle.contains(event.target)) setMenu(false); });
const goTo = (index, smooth = true) => {
  const top = blocks[index].offsetTop - (index === 0 ? 0 : 24);
  reader.scrollTo({ top, behavior: smooth && !reduceMotion.matches ? 'smooth' : 'auto' });
};
tocLinks.forEach(link => link.addEventListener('click', () => {
  const index = Number(link.dataset.index);
  setMenu(false);
  goTo(index);
  history.replaceState(null, '', `#${steps[index].id}`);
}));

// --- left panel --------------------------------------------------------------------------------------------------
const graph = buildGraph(document.getElementById('graph'));
const cards = steps.map(step => {
  const card = renderCard(step, { snippets, digits });
  stack.append(card);
  return card;
});
const STACK_DEPTH = 4;

let current = -1;
function show(index) {
  if (index === current) return;
  current = index;
  const step = steps[index];
  cards.forEach((card, i) => {
    const depth = index - i;
    card.classList.toggle('is-focus', depth === 0);
    card.classList.toggle('is-behind', depth > 0);
    card.classList.toggle('is-out', depth < 0);
    card.classList.toggle('is-buried', depth > STACK_DEPTH);
    card.style.setProperty('--depth', Math.max(0, Math.min(depth, STACK_DEPTH + 1)));
    card.setAttribute('aria-hidden', String(depth !== 0));
    card.inert = depth !== 0;
  });
  const body = cards[index].querySelector('.card-body');
  const focusLine = body.querySelector('.ln.focus');
  if (focusLine && body.scrollHeight > body.clientHeight) {
    body.scrollTo({ top: Math.max(0, focusLine.offsetTop - body.clientHeight * 0.3), behavior: reduceMotion.matches ? 'auto' : 'smooth' });
  }
  graph.setState(step.graph);
  blocks.forEach((block, i) => block.classList.toggle('is-current', i === index));
  tocLinks.forEach((link, i) => i === index ? link.setAttribute('aria-current', 'location') : link.removeAttribute('aria-current'));
  counter.textContent = `${String(index + 1).padStart(2, '0')} / ${steps.length}`;
  chapterLabel.textContent = chapters[index];
}

// The current step is the last passage whose top has crossed 38% of the reading column.
function measure() {
  const box = reader.getBoundingClientRect();
  const line = box.top + box.height * 0.38;
  let index = 0;
  blocks.forEach((block, i) => { if (block.getBoundingClientRect().top <= line) index = i; });
  if (reader.scrollTop + reader.clientHeight >= reader.scrollHeight - 2) index = blocks.length - 1;
  show(index);
}
let queued = false;
const schedule = () => {
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; measure(); });
};
reader.addEventListener('scroll', schedule, { passive: true });
addEventListener('resize', schedule);

// Start with every card off to the left, then let the first one jump in.
const start = steps.findIndex(step => `#${step.id}` === location.hash);
if (start > 0) goTo(start, false);
requestAnimationFrame(() => requestAnimationFrame(() => {
  document.body.classList.add('is-ready');
  measure();
}));
