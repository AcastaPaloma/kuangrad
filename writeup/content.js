// The write-up, one entry per reading step. Each step pairs a passage (right column) with a card that slides into
// the stack on the left, and with the region of the computation graph that the yellow box should frame.
//   level     2 for a section heading, 3 for a sub-step
//   chapter   starts a new chapter in Contents and in the chapter label
//   body      paragraphs; **bold**, *italic*, `code`, $tex$ and $$display tex$$ are supported (see format.js), and
//             the word Neuron is always set as code. Other blocks: { list: [...] }, { math: 'tex' }, { note: '...' },
//             { preface: { lead, list, close } }, { stages: { kind, head, rows } } (a table that follows the graph's animation), { details: id, label }
//             (a button that opens details[id] in a dialog)
//   card      { type: 'code', file, ranges: [[from, to], ...], focus: [line, ...] }  lines come from the repo via
//             scripts/build-snippets.mjs, so they always match the source;
//             { type: 'text', title, lines } for hand-written listings; otherwise a visual from cards.js
//   graph     { hl: [region, ...], mode, tag }  regions and modes are defined in graph.js

export const steps = [
  {
    id: 'kuangrad', chapter: 'Overview', level: 2, title: 'kuangrad',
    body: [
      { preface: {
        lead: "Hey, thanks for being here. This short write-up is for the:",
        list: [
          "applied engineers who've completely lost it;",
          "robotics (physical AI, buzz buzz) fanatics, fluent in WAMs & VLAs, but forgot their roots;",
          "newbies (issalgud, I'm right there with you guys, and welcome to ML);"
        ],
        close: "have fun reading, I learned so much - try implementing one yourself, so rewarding !"
      } },
      { note: '**I didn’t bother with efficiency.** No NumPy, no tensors, no batching: every number is its own Python object, and every derivative is written down by hand. One training step takes 3 to 4 seconds on my M5 Pro.' },
      'kuangrad is an autograd engine where every number is its own Python object, plus a small neural network trained with it. A weight matrix is a list of lists of objects, and backpropagation is a walk over those objects. The engine is two short files, `neuron.py` and `operations.py`, about 115 lines together.',
      'The network reads a 16 × 16 black-and-white drawing of a digit and outputs ten probabilities, one per digit. The 1,000 training drawings were made by hand in a small tkinter app that lives in the same repo.',
      'The left panel follows along. The graph at the top is what kuangrad builds for one example. Each circle is a Neuron, shaded by its value: blue for positive, purple for negative, white for zero. Between the layers sit the weight matrices $W_1$, $W_2$ and $W_3$, one cell per weight, and every cell is a Neuron too. Every symbol is in the notation key in the corner.'
    ],
    card: {
      type: 'text', title: 'kuangrad/', lines: [
        'kuangrad/',
        '├── draw_dataset.py          # 16 × 16 drawing app → dataset.json',
        '├── dataset.json             # 1,000 drawings, 100 per digit',
        '├── neuron.py                # one number + the edges around it',
        '├── operations.py            # multi · relu · sftmx · inverse_log_max',
        '├── model_io.py              # save / load the weights as JSON',
        '└── forward_and_backward.py  # build, forward, backward, update',
        '',
        '# network, for one drawing:',
        '#   x[256] → multi → z₁[512] → relu → a₁[512]',
        '#          → multi → z₂[256] → relu → a₂[256]',
        '#          → multi → l[10] → sftmx → p[10]',
        '#          → inverse_log_max → L'
      ]
    },
    graph: { hl: ['all'], tag: 'one drawing' }
  },

  {
    id: 'dataset', chapter: 'Data', level: 2, title: 'Drawing the dataset',
    body: [
      '`draw_dataset.py` opens a 16 × 16 grid. Left-click paints a cell black, right-click erases it, and Save appends the drawing to `dataset.json` with the label typed in the box.',
      'Pixels are 0 or 1; there are no grey levels. The drawings on the left are real samples from the file, one per digit. Hover one to enlarge it.',
      'The 1,000 samples are stored in label order: the first 100 are zeros, the next 100 are ones, and so on. Training picks a random index from 5 to 995, so five of the zeros and four of the nines are never seen.'
    ],
    card: { type: 'digits', title: 'dataset.json', meta: 'one sample per label' },
    graph: { hl: ['x'], tag: 'x · 256 pixels' }
  },
  {
    id: 'flatten', level: 3, title: 'From grid to vector',
    body: [
      'The network has no notion of rows. `flatten` reads the grid row by row and returns 256 floats, so pixel $(r, c)$ becomes entry $16r + c$. That list is the input column $x$.',
      'Training never opens the drawing app. `forward_and_backward.py` only calls `load_dataset` and wraps each of the 256 values of every sample in a Neuron.'
    ],
    card: { type: 'code', file: 'draw_dataset.py', ranges: [[10, 27]], focus: [11, 12] },
    graph: { hl: ['x'], tag: 'flatten' }
  },

  {
    id: 'neuron', chapter: 'Engine', level: 2, title: 'A Neuron is one number',
    body: [
      'A Neuron holds four things:',
      { list: [
        '**value**: the number itself.',
        '**grad**: computed during backprop, using the all-crucial formula $$\\pd{L}{\\parent} = \\sum_{\\text{child}} \\pd{\\child}{\\parent} \\cdot \\pd{L}{\\child}$$',
        '**parents**: the Neurons this one was computed from.',
        '**local_grads**: $\\pd{\\child}{\\parent}$, computed and cached during the forward pass.'
      ] },
      'There is no tensor type. A 512-wide layer is a Python list of 512 Neurons, and a 256 × 512 weight matrix is a list of 256 lists of 512 Neurons. The yellow box has shrunk to a single circle: this is the unit everything else is made of.'
    ],
    card: { type: 'code', file: 'neuron.py', ranges: [[1, 17]], focus: [5, 6, 7, 10] },
    graph: { hl: ['neuron'], tag: 'Neuron' }
  },
  {
    id: 'edges', level: 3, title: 'Who stores the derivative',
    body: [
      'Each edge in the graph is recorded twice. The child lists the parent in `parents`. The parent keeps the local derivative in its own `local_grads`, as an entry [child, parent, $\\pd{\\child}{\\parent}$].',
      'To read an entry back, `local_grad_to(child)` walks the list from the start until it finds that child. Keep that loop in mind; it comes back in the last chapter.'
    ],
    card: { type: 'edge', title: 'one edge, stored', meta: '$x_7 \\cdot w_{7,3} \\to z_{1,3}$' },
    graph: { hl: ['edge'], mode: 'edge', tag: 'one edge' }
  },

  {
    id: 'multi', chapter: 'Operations', level: 2, title: 'One dense layer',
    body: [
      '`multi(features, matrix, bias)` computes one layer, $z = xW + b$, one output at a time. For output $j$ it starts from an empty Neuron, adds $w_{ij}\\,x_i$ for every input $i$, then adds the bias $b_j$.',
      'The derivatives are written down while the sum is built: $\\pd{z_j}{w_{ij}} = x_i$ and $\\pd{z_j}{x_i} = w_{ij}$, so every product adds one entry to the weight and one to the feature. The bias adds an entry of 1.',
      'The entries store `.value`, a float copied at that moment, rather than the Neuron itself. That copy is what later lets the weights change in the middle of the backward pass without corrupting gradients that still need the old values.',
      'For the first layer this loop runs 131,072 times per example and leaves 262,656 local-gradient entries behind.'
    ],
    card: { type: 'code', file: 'operations.py', ranges: [[8, 8], [22, 49]], focus: [32, 38, 39, 44] },
    graph: { hl: ['layer1'], mode: 'layer', tag: 'multi(x, w1, b1)' }
  },
  {
    id: 'relu', level: 3, title: 'ReLU',
    body: [
      '`relu` makes one new Neuron per input. Its value is $\\max(0, z)$, and its single local gradient is $\\pd{a}{z} = 1$ if $z > 0$, else $0$.',
    ],
    card: { type: 'code', file: 'operations.py', ranges: [[51, 68]], focus: [56, 57, 58, 59, 66] },
    graph: { hl: ['relu1', 'relu2'], mode: 'relu', tag: 'relu' }
  },
  {
    id: 'softmax', level: 2, title: 'Softmax',
    body: [
      'The last dense layer outputs ten logits. `sftmx` turns them into probabilities: subtract the largest logit so that `e**x` cannot overflow, exponentiate, divide by the sum.',
      { math: 'p_i = \\frac{e^{\\,l_i - m}}{\\sum_j e^{\\,l_j - m}}, \\qquad m = \\max_j l_j' },
      'Unlike ReLU, each $p_i$ depends on every logit in the layer, not only on $l_i$. That is the block of lines between $l$ and $p$, and it is why each logit receives ten local-gradient entries, one per $p_i$.',
      { details: 'jacobian', label: 'The full 10 × 10 Jacobian' }
    ],
    card: { type: 'code', file: 'operations.py', ranges: [[70, 87]], focus: [74, 75, 84, 85] },
    graph: { hl: ['softmax'], mode: 'softmax', tag: 'sftmx(l)' }
  },
  {
    id: 'loss', level: 2, title: 'The loss',
    body: [
      '`inverse_log_max` is cross-entropy for a single example, $L = -\\log p_y$, where $y$ is the label. The probability is clamped at $10^{-15}$, so a confident wrong answer gives a large, finite loss instead of infinity.',
      'Only $p_y$ appears in the formula, so only $p_y$ gets a non-zero local gradient, $\\pd{L}{p_y} = -\\frac{1}{p_y}$. In the graph, that is the one solid line into $L$; the other nine carry 0.',
      'Libraries usually fuse softmax and cross-entropy because the combined derivative is so simple:',
      { math: '\\pd{L}{l_j} = p_j - [\\,j = y\\,]' },
      'kuangrad never writes that formula down. It gets there in two hops through the graph: $-\\frac{1}{p_y}$ times $\\pd{p_y}{l_j} = p_y\\,(\\delta_{yj} - p_j)$. On a real example the two agree to within $10^{-17}$.'
    ],
    card: { type: 'code', file: 'operations.py', ranges: [[89, 99]], focus: [94, 96, 97] },
    graph: { hl: ['loss'], mode: 'loss', tag: 'inverse_log_max' }
  },

  {
    id: 'build', chapter: 'Training', level: 2, title: 'Building the network',
    body: [
      'The network is 256 → 512 → 256 → 10. Each weight and bias starts uniform in [[$\\pm\\frac{1}{\\sqrt{\\text{fan\\_in}}}$::`fan_in` is how many inputs feed one unit: 256 for $W_1$ and $W_3$, 512 for $W_2$. Each unit adds up `fan_in` random products $w_{ij}\\,x_i$, and a sum of $n$ random terms grows like $\\sqrt{n}$. Shrinking every weight’s range by $\\sqrt{\\text{fan\\_in}}$ cancels that growth, so the sum stays around 1 however wide the layer is: $\\pm 0.0625$ for $W_1$, $\\pm 0.044$ for $W_2$.]], so the sum feeding each unit stays around unit scale and the first logits don’t blow up.',
      'The parameters are just the Neurons inside `w1`, `w2`, `w3`, `b1`, `b2` and `b3`, collected into a set called `params`: 265,482 objects. In the graph they are the three matrices between the layers. Each weight is a Neuron like any other.'
    ],
    card: { type: 'code', file: 'forward_and_backward.py', ranges: [[26, 58]], focus: [35, 36, 58] },
    graph: { hl: ['w1', 'w2', 'w3'], mode: 'params', tag: 'params' }
  },
  {
    id: 'forward', level: 2, title: 'Forward pass',
    body: [
      'Each iteration picks one drawing at random and runs six calls in order. Every call returns a list of new Neurons wired to the list before it, and fills the `local_grads` of its inputs as it goes: $\\pd{\\child}{\\parent}$, computed and cached. The graph plays one pass: the teal squares are those entries landing, and the labelled chip follows one of them back to the parent that stores it.',
      { stages: {
        kind: 'forward',
        head: ['call', 'builds', 'fills `local_grads` of'],
        rows: [
          ['`multi`', '$z_1$', '$x$, $W_1$, $b_1$'],
          ['`relu`', '$a_1$', '$z_1$'],
          ['`multi`', '$z_2$', '$a_1$, $W_2$, $b_2$'],
          ['`relu`', '$a_2$', '$z_2$'],
          ['`multi`', '$l$', '$a_2$, $W_3$, $b_3$'],
          ['`sftmx`', '$p$', '$l$'],
          ['`inverse_log_max`', '$L$', '$p$']
        ]
      } },
      'By the time `loss` exists, the whole graph is in memory: the 256 inputs, 1,557 new Neurons for the activations, the 265,482 parameters, and 531,064 local-gradient entries connecting them. Nothing has a gradient yet.'
    ],
    card: { type: 'code', file: 'forward_and_backward.py', ranges: [[64, 86]], focus: [70, 73, 76, 79, 82, 83, 86] },
    graph: { hl: ['all'], mode: 'forward', tag: 'forward' }
  },
  {
    id: 'backward', level: 2, title: 'Backward: a breadth-first walk',
    body: [
      'Backpropagation starts at `loss` with grad = 1 and walks toward the inputs with a queue. Popping a Neuron pushes its gradient into each parent, which is the chain rule applied one edge at a time:',
      { math: '\\gr{\\texttt{parent.grad}} \\mathrel{+}= \\loc{\\pd{\\child}{\\parent}} \\cdot \\gr{\\texttt{child.grad}}' },
      'The teal factor was cached on the forward pass; the orange one arrives from the child. The graph plays the walk with real numbers from one pass: each orange chip is a child’s grad flying into its parent, and the sum under the graph is that parent’s grad adding up, term by term.',
      { stages: {
        kind: 'backward',
        head: ['popped', '`grad` flows into'],
        rows: [
          ['$L$', '$p$'],
          ['$p$', '$l$'],
          ['$l$', '$a_2$, $W_3$, $b_3$'],
          ['$a_2$', '$z_2$'],
          ['$z_2$', '$a_1$, $W_2$, $b_2$'],
          ['$a_1$', '$z_1$'],
          ['$z_1$', '$x$, $W_1$, $b_1$']
        ]
      } },
      'Each parent is queued the first time it is seen. Summed over every child, those += give exactly $\\pd{L}{\\parent} = \\sum \\pd{\\child}{\\parent} \\cdot \\pd{L}{\\child}$. A pixel’s gradient is the sum of contributions from all 512 first-layer units that used it, and those contributions arrive one at a time, as each unit is popped.'
    ],
    card: { type: 'code', file: 'forward_and_backward.py', ranges: [[95, 107]], focus: [102] },
    graph: { hl: ['all'], mode: 'backward', tag: 'backward' }
  },
  {
    id: 'bfs-order', level: 3, title: 'Why breadth-first is enough here',
    body: [
      'A Neuron should only pass its gradient on once that gradient is final, which means every child has already reported in. The general way to guarantee that is a topological sort. kuangrad uses plain BFS, and it works because of the shape of this network.',
      'Every path from $L$ to a given Neuron has the same length. The numbers above the graph are those lengths: $p$ is 1 step from $L$, the logits 2, $a_2$ and $W_3$ are 3, and so on down to the pixels and $W_1$ at 7. BFS pops everything at depth $d$ before anything at depth $d + 1$. Every child of a depth-$(d + 1)$ Neuron sits at depth $d$, so when a Neuron is popped, all of its children are already done.'
    ],
    card: { type: 'bfs', title: 'BFS order', meta: 'BFS level = path length' },
    graph: { hl: ['all'], mode: 'depth', tag: 'depth = BFS level' }
  },
  {
    id: 'update', level: 3, title: 'Updating while walking',
    body: [
      'There is no separate optimizer step. When a parameter is popped, its gradient is already final (by the same depth argument), so it is updated on the spot, $\\texttt{value} \\mathrel{-}= \\text{LR} \\cdot \\texttt{grad}$, with $\\text{LR} = 3 \\times 10^{-3}$.',
    ],
    card: { type: 'code', file: 'forward_and_backward.py', ranges: [[99, 107]], focus: [106, 107] },
    graph: { hl: ['w1', 'w2', 'w3'], mode: 'update', tag: 'value -= LR * grad' }
  },
  {
    id: 'reset', level: 3, title: 'Next forward pass',
    body: [
      'After the walk, every parameter has its gradient zeroed and its `local_grads` cleared. The same happens to the 256 input Neurons, which are reused across iterations.',
      'Clearing `local_grads` also drops the references from the parameters to this iteration’s activations. Once the next iteration rebinds `z1`, `a1` and the `seen` set, nothing points at the old activation Neurons any more, and Python can free them.'
    ],
    card: { type: 'code', file: 'forward_and_backward.py', ranges: [[109, 114]], focus: [110, 111, 112, 113, 114] },
    graph: { hl: ['x', 'w1', 'w2', 'w3'], mode: 'params', tag: 'reset' }
  },

  {
    id: 'cost', chapter: 'Cost', level: 2, title: 'Where the time goes',
    body: [
      'On my M5 Pro with Python 3.14, one iteration takes about 3 seconds: about 2.5 s backward, 0.2 s forward, and the rest logging and resetting the parameters. At that rate, the 10,000 iterations in the loop take about 8½ hours.',
      'Almost all of it is `local_grad_to` scanning lists from the front: 50.7 million comparisons per backward pass to find 531,064 numbers.',
      'A dict keyed by child brings the backward pass from 2.5 s to 0.18 s, with identical gradients.'
    ],
    card: { type: 'cost', title: 'one training step', meta: 'M5 Pro · Python 3.14' },
    graph: { hl: ['layer1', 'layer2'], mode: 'hot', tag: 'list scans' }
  },
  {
    id: 'next', level: 2, title: 'Next',
    body: [
      'Next, I’ll break down the optimizer, and backprop across more sophisticated models (maybe in one transformer module???).',
      'Thank you!'
    ],
    card: {
      type: 'text', title: 'next', lines: [
        '# next',
        '',
        '# 1. break down the optimizer',
        '',
        '# 2. backprop across more sophisticated models',
        '#    (maybe in one transformer module???)'
      ]
    },
    graph: { hl: ['all'], tag: 'kuangrad' }
  }
];

// The notation key that floats in the corner of the article: [symbol, one word, full definition].
export const notation = [
  ['x_i', 'pixel', 'pixel $i$ of the drawing, 0 or 1'],
  ['z_j', 'pre-ReLU', 'a hidden unit before ReLU, $\\sum_i w_{ij}\\,x_i + b_j$'],
  ['a_j', 'activation', 'the same unit after ReLU, $\\max(0, z_j)$'],
  ['z_{1,3}', 'layer, unit', 'layer 1, unit 3. A single number, as in $z_1$ or $W_2$, is the layer'],
  ['w_{ij}', 'weight', 'connects neuron $i$ to neuron $j$ in the next layer: row $i$, column $j$ of $W$'],
  ['b_j', 'bias', 'labelled $j$ because it belongs to the child neuron $j$: one bias per child'],
  ['l_j', 'logit', 'the score for digit $j$, before softmax'],
  ['p_j', 'probability', 'the probability of digit $j$; $p_y$ is the true label’s'],
  ['L', 'loss', '$-\\log p_y$ for one drawing'],
  ['\\loc{\\pd{c}{p}}', 'local grad', 'child over parent, cached in `parent.local_grads` during the forward pass'],
  ['\\gr{\\pd{L}{n}}', 'grad', '`n.grad`, filled in during backprop']
];

// Opened from a { details } block in a passage, in a dialog the reader can close.
export const details = {
  jacobian: {
    title: 'The softmax Jacobian',
    body: [
      'Every logit feeds every probability, with',
      { math: '\\pd{p_i}{l_j} = p_i\\,(\\delta_{ij} - p_j)' },
      'where $\\delta_{ij}$ is 1 when $i = j$ and 0 otherwise. Laid out as a matrix, those 100 numbers are the softmax Jacobian. The diagonal is positive: raising a logit raises its own probability. Every other entry is negative: raising one logit takes probability away from the other nine.',
      'The example is a network that is 70% sure the drawing is a 2, so most of the matrix sits in row and column 2. When the network is confident, the derivatives elsewhere are close to zero, and so are the gradients that pass through them.'
    ],
    card: 'jacobian'
  }
};

// One real sample per label from dataset.json (index 100·label + 10), 256 bits row-major as hex.
export const digits = [
  '03e00f3008181008300c30041004100430042004100c18180c30046007c00000',
  '004000c003c001c000c000c000c0008000800180018001800180018001800000',
  '03c00fe00820183010300060004000c0018001800300020006fe0ff000000000',
  '0000000000c03fc0308001800100030001c0006000200030003000600fc00000',
  '0000000000180030007000d00190031006100c300ff000fc0020002000200000',
  '0f000ff00c30080008001bc01fe00020003000100030002038e00fc002000000',
  '00000000003001f0078006000c000c001f801ec0186018601c600fe003800000',
  '1ff000100030002000200040004000c000800080018001000300020002000000',
  '000003e007f804180c180618073007f00ff01c781818181818180c1807f003e0',
  '0000008007e004300410041002f001b000200020006000401ec0078000000000'
];
