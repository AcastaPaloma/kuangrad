// Text formatting shared by the article, the cards and the graph: `code`, **bold**, *italic*, $inline$ and $$display$$
// TeX typeset by the vendored KaTeX (loaded as a classic script before any module runs), and the word Neuron, which
// always reads as the class name.
const MACROS = {
  '\\pd': '\\frac{\\partial #1}{\\partial #2}',
  '\\child': '\\,\\text{child}',
  '\\parent': '\\,\\text{parent}',
  // the two colours of the chain rule: a cached local derivative, and a gradient ('#' would read as a parameter)
  '\\loc': '{\\color{0f766e}#1}',
  '\\gr': '{\\color{c2410c}#1}'
};

export const escapeHtml = value => String(value).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);

// `html` drops the hidden MathML copy, for labels that are aria-hidden anyway.
export const tex = (source, display = false, html = false) => window.katex.renderToString(source, {
  displayMode: display, throwOnError: false, output: html ? 'html' : 'htmlAndMathml', macros: { ...MACROS }
});

const prose = text => escapeHtml(text)
  .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
  .replace(/\*(.+?)\*/g, '<em>$1</em>')
  .replace(/\bNeurons?\b/g, '<code>$&</code>');

// Punctuation right after inline math stays on its line.
export const rich = source => String(source).split(/(`[^`]+`|\$\$[^$]+\$\$|\$[^$]+\$[.,;:)]*)/).map(part => {
  if (part.length > 1 && part.startsWith('`')) return `<code>${escapeHtml(part.slice(1, -1))}</code>`;
  if (part.startsWith('$$')) return tex(part.slice(2, -2), true);
  const math = /^\$([^$]+)\$(.*)$/.exec(part);
  if (math) return math[2] ? `<span class="nobr">${tex(math[1])}${escapeHtml(math[2])}</span>` : tex(math[1]);
  return prose(part);
}).join('');
