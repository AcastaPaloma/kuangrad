import { steps, details, digits } from './content.js';
import { snippets } from './snippets.js';
import { buildGraph } from './graph.js';
import { renderCard, jacobianCard } from './cards.js';
import { escapeHtml, rich, tex } from './format.js';

const reader = document.getElementById('reader');
const article = document.getElementById('article');
const stack = document.getElementById('stack');
const counter = document.getElementById('step-counter');
const chapterLabel = document.getElementById('chapter-label');
const toggle = document.getElementById('contents-toggle');
const menu = document.getElementById('contents-menu');
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');

// --- prose -------------------------------------------------------------------------------------------------------
const stagesHtml = ({ kind, head, rows }) => `<table class="stages" data-kind="${kind}">
  <thead><tr><th></th>${head.map(h => `<th>${rich(h)}</th>`).join('')}</tr></thead>
  <tbody>${rows.map((row, i) => `<tr data-stage="${i}"><td class="stage-n">${i + 1}</td>${row.map(cell => `<td>${rich(cell)}</td>`).join('')}</tr>`).join('')}</tbody>
</table>`;
const blockHtml = block => {
  if (typeof block === 'string') return `<p>${rich(block)}</p>`;
  if (block.list) return `<ul>${block.list.map(item => `<li>${rich(item)}</li>`).join('')}</ul>`;
  if (block.math) return `<div class="math">${tex(block.math, true)}</div>`;
  if (block.note) return `<p class="note">${rich(block.note)}</p>`;
  if (block.stages) return stagesHtml(block.stages);
  if (block.details) return `<p><button class="details-btn" type="button" data-details="${block.details}" aria-haspopup="dialog">${rich(block.label)}<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3.5v9M3.5 8h9"/></svg></button></p>`;
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
    ? `<header class="title-block"><h1>${rich(step.title)}</h1><p class="subtitle">${rich('An autograd engine, one Neuron at a time')}</p><p class="byline">Kuan Yi Wang · <a href="https://github.com/AcastaPaloma/kuangrad" target="_blank" rel="noopener">source on GitHub</a></p></header>`
    : `${step.chapter ? `<p class="chapter-kicker">${escapeHtml(step.chapter)}</p>` : ''}<h${step.level}>${rich(step.title)}</h${step.level}>`;
  section.innerHTML = heading + step.body.map(blockHtml).join('');
  article.append(section);
  return section;
});
const end = document.createElement('footer');
end.className = 'article-end';
end.innerHTML = 'The code shown on the left is copied from the repository by <code>scripts/build-snippets.mjs</code>, line numbers included. <a href="https://github.com/AcastaPaloma/kuangrad" target="_blank" rel="noopener">Read it on GitHub</a>.';
article.append(end);

// --- details dialog ----------------------------------------------------------------------------------------------
const dialog = document.createElement('dialog');
dialog.className = 'details-dialog';
dialog.setAttribute('aria-labelledby', 'details-title');
dialog.innerHTML = '<header><h2 id="details-title"></h2><button class="dialog-close" type="button" aria-label="Close"><svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 4 8 8M12 4l-8 8"/></svg></button></header><div class="dialog-body"></div>';
document.body.append(dialog);
const dialogBody = dialog.querySelector('.dialog-body');
dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', event => { if (event.target === dialog) dialog.close(); }); // the backdrop
article.addEventListener('click', event => {
  const button = event.target.closest('.details-btn');
  if (!button) return;
  const entry = details[button.dataset.details];
  dialog.querySelector('#details-title').textContent = entry.title;
  dialogBody.innerHTML = entry.body.map(blockHtml).join('');
  if (entry.card === 'jacobian') dialogBody.append(jacobianCard());
  dialog.showModal();
  dialogBody.scrollTop = 0;
});

// --- contents ----------------------------------------------------------------------------------------------------
const toc = document.getElementById('toc');
toc.innerHTML = steps.map((step, i) => `${step.chapter ? `<p class="toc-chapter">${escapeHtml(step.chapter)}</p>` : ''}
  <button class="toc-link level-${step.level}" type="button" data-index="${i}"><span class="toc-number">${String(i + 1).padStart(2, '0')}</span><span>${rich(step.title)}</span></button>`).join('');
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
// The stage tables in the forward and backward passages follow the graph's animation.
const stageTables = [...article.querySelectorAll('.stages')];
graph.onStage((kind, stage) => {
  stageTables.forEach(table => {
    const live = table.dataset.kind === kind;
    table.classList.toggle('is-playing', live);
    table.querySelectorAll('tbody tr').forEach((row, i) => {
      row.classList.toggle('is-live', live && i === stage);
      row.classList.toggle('is-done', live && i < stage);
    });
  });
});

// Every card opens at its top. While there is more below, an arrow in its corner says so and scrolls it.
const cards = steps.map(step => {
  const card = renderCard(step, { snippets, digits });
  const body = card.querySelector('.card-body');
  const more = document.createElement('button');
  more.className = 'card-more';
  more.type = 'button';
  more.setAttribute('aria-label', 'Scroll this card down');
  more.innerHTML = '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="m4 6 4 4 4-4"/></svg>';
  more.addEventListener('click', () => body.scrollBy({ top: body.clientHeight * 0.7, behavior: reduceMotion.matches ? 'auto' : 'smooth' }));
  card.append(more);
  card.refreshMore = () => card.classList.toggle('has-more', body.scrollTop + body.clientHeight < body.scrollHeight - 6);
  body.addEventListener('scroll', card.refreshMore, { passive: true });
  const watch = new ResizeObserver(card.refreshMore);
  watch.observe(body);
  if (body.firstElementChild) watch.observe(body.firstElementChild);
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
  cards[index].querySelector('.card-body').scrollTop = 0;
  cards[index].refreshMore();
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
