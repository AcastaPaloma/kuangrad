// The write-up, one entry per reading step. Each step pairs a passage (right column) with a card that slides into
// the stack on the left, and with the region of the computation graph that the yellow box should frame.
//   level     2 for a section heading, 3 for a sub-step
//   chapter   starts a new chapter in Contents and in the chapter label
//   body      paragraphs; **bold**, *italic*, `code`, $tex$ and $$display tex$$ are supported (see format.js), and
//             the word Neuron is always set as code. Other blocks: { list: [...] }, { math: 'tex' }, { note: '...' },
//             { stages: { kind, head, rows } } (a table that follows the graph's animation), { details: id, label }
//             (a button that opens details[id] in a dialog)
//   card      { type: 'code', file, ranges: [[from, to], ...], focus: [line, ...] }  lines come from the repo via
//             scripts/build-snippets.mjs, so they always match the source;
//             { type: 'text', title, lines } for hand-written listings; otherwise a visual from cards.js
//   graph     { hl: [region, ...], mode, tag }  regions and modes are defined in graph.js

export const steps = [
  {
    id: 'kuangrad', chapter: 'Overview', level: 2, title: 'kuangrad',
    body: [
      { note: '**We didn’t bother with efficiency.** No NumPy, no tensors, no batching: every number is its own Python object, and every derivative is written down by hand. One training step takes about 11 seconds. The point was to see every piece of backpropagation, not to train fast.' },
      'kuangrad is an autograd engine where every number is its own Python object, plus a small neural network trained with it. A weight matrix is a list of lists of objects, and backpropagation is a walk over those objects. The engine is two short files, `neuron.py` and `operations.py`, about 115 lines together.',
      'The network reads a 16 × 16 black-and-white drawing of a digit and outputs ten probabilities, one per digit. The 1,000 training drawings were made by hand in a small tkinter app that lives in the same repo.',
      'The left panel follows along. The graph at the top is what kuangrad builds for one example. Each circle is a Neuron, shaded by its value: blue for positive, purple for negative, white for zero. Between the layers sit the weight matrices $W_1$, $W_2$ and $W_3$, one cell per weight, and every cell is a Neuron too. Only a few rows of each layer are drawn; the real ones hold up to 512. The yellow box marks what the current passage is about, and its code slides in below. Scroll back up and it slides out again.'
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
      'The 1,000 samples are stored in label order: the first 100 are zeros, the next 100 are ones, and so on. Training picks a random index from 5 to 995, so five of the zeros and four of the nines are never seen. They could serve as a small held-out set.'
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
      'This is the reverse of micrograd, where each output carries a closure that knows how to push gradient into its inputs. In kuangrad, each input keeps one entry per output that used it. A pixel is used by all 512 units of $z_1$, so it carries 512 entries. A weight is used once per example, so it carries one.',
      'To read an entry back, `local_grad_to(child)` walks the list from the start until it finds that child. Keep that loop in mind; it comes back in the last chapter.'
    ],
    card: { type: 'edge', title: 'one edge, stored', meta: '$x_7 \\cdot w_{7,3} \\to z_3$' },
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
      'A local gradient of 0 cuts the path: any gradient that reaches a dead unit is multiplied by 0 on the way back, so nothing upstream of it learns from this example. With fresh weights, about half the units in each hidden layer are dead for a given drawing (264 of 512 and 130 of 256 on one sample). In the graph they are the white circles with dashed outlines.'
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
      'The network is 256 → 512 → 256 → 10. Each weight and bias starts uniform in $\\pm\\frac{1}{\\sqrt{\\text{fan\\_in}}}$, so the sum feeding each unit stays around unit scale and the first logits don’t blow up.',
      'The parameters are just the Neurons inside `w1`, `w2`, `w3`, `b1`, `b2` and `b3`, collected into a set called `params`: 265,482 objects. In the graph they are the three matrices between the layers. Each weight is a Neuron like any other.'
    ],
    card: { type: 'code', file: 'forward_and_backward.py', ranges: [[26, 58]], focus: [35, 36, 58] },
    graph: { hl: ['w1', 'w2', 'w3'], mode: 'params', tag: 'params' }
  },
  {
    id: 'forward', level: 2, title: 'Forward pass',
    body: [
      'Each iteration picks one drawing at random and runs six calls in order. Every call returns a list of new Neurons wired to the list before it, and fills the `local_grads` of its inputs as it goes: $\\pd{\\child}{\\parent}$, computed and cached. The graph plays one pass; the teal squares are those entries landing.',
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
      'The teal factor was cached on the forward pass; the orange one arrives from the child. The graph plays the walk; the orange rings are `grad` filling in.',
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
      'Every path from $L$ to a given Neuron has the same length. The numbers above the graph are those lengths: $p$ is 1 step from $L$, the logits 2, $a_2$ and $W_3$ are 3, and so on down to the pixels and $W_1$ at 7. BFS pops everything at depth $d$ before anything at depth $d + 1$. Every child of a depth-$(d + 1)$ Neuron sits at depth $d$, so when a Neuron is popped, all of its children are already done.',
      'A skip connection would break this, as the card shows. If $a_1$ also fed the logits directly, $a_1$ would be 3 steps from $L$ along one path and 5 along the other. BFS would pop it at depth 3, before the long path had delivered its share, and pass an incomplete sum down to $z_1$.',
      'For the network as it is, a finite-difference check on a small 6 → 7 → 5 → 4 network built from the same operations matches the BFS gradients to within $3 \\times 10^{-10}$.'
    ],
    card: { type: 'bfs', title: 'BFS order', meta: 'layered vs. skip connection' },
    graph: { hl: ['all'], mode: 'depth', tag: 'depth = BFS level' }
  },
  {
    id: 'update', level: 3, title: 'Updating while walking',
    body: [
      'There is no separate optimizer step. When a parameter is popped, its gradient is already final (by the same depth argument), so it is updated on the spot, $\\texttt{value} \\mathrel{-}= \\text{LR} \\cdot \\texttt{grad}$, with $\\text{LR} = 3 \\times 10^{-3}$.',
      'That would be a bug in most engines. It is safe here only because `multi` copied `weight.value` into the local gradients during the forward pass. The Neurons still waiting in the queue read those stored floats, never the weights’ new values.'
    ],
    card: { type: 'code', file: 'forward_and_backward.py', ranges: [[99, 107]], focus: [106, 107] },
    graph: { hl: ['w1', 'w2', 'w3'], mode: 'update', tag: 'value -= LR * grad' }
  },
  {
    id: 'reset', level: 3, title: 'Letting the graph go',
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
      'One iteration takes about 10 seconds backward and between 0.5 and 1.7 seconds forward (measured on Python 3.13 in a cloud container; your machine will differ). At that rate, the 10,000 iterations in the loop take more than a day.',
      'Almost all of it is `local_grad_to`. Each lookup scans the parent’s list from the front. A pixel has 512 entries and is looked up once by each of the 512 first-layer units: about 131,000 comparisons per pixel. A first-layer activation has 256 entries and is looked up 256 times. In total, one backward pass makes 50.7 million comparisons to find 531,064 numbers. The two highlighted blocks are where they happen.',
      'Keying the entries by child would make each lookup constant time. A quick test that builds a dict per Neuron on first lookup brings the backward pass from 10.0 s to 0.75 s, and all 265,482 parameter gradients come out identical.'
    ],
    card: { type: 'cost', title: 'one training step', meta: 'Python 3.13' },
    graph: { hl: ['layer1', 'layer2'], mode: 'hot', tag: 'list scans' }
  },
  {
    id: 'next', level: 2, title: 'Next',
    body: [
      'This is the first version of the write-up. A training curve and accuracy on held-out drawings will go here once a full run finishes.',
      'The card on the left lists the obvious next steps, in roughly the order they pay off.'
    ],
    card: {
      type: 'text', title: 'TODO', lines: [
        '# next',
        '',
        '# 1. key local_grads by child',
        '#    backward: 10.0 s → 0.75 s, same gradients',
        '',
        '# 2. hold out a test set',
        '#    samples 0–4 and 996–999 are never drawn',
        '',
        '# 3. plot the logged loss (train_log.txt) here',
        '',
        '# 4. mini-batches: average grads over k drawings',
        '#    before each update',
        '',
        '# 5. topological sort instead of BFS',
        '#    needed before any skip connection'
      ]
    },
    graph: { hl: ['all'], tag: 'kuangrad' }
  }
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
