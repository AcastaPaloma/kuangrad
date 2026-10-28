// inference.py's forward pass, on the float32 weights in results/model.bin. Shared by the Results tab and
// scripts/build-results.mjs, so the accuracy it prints is the accuracy of exactly what the page runs. The page also
// passes the pad through frame() first; the script scores the training drawings as they are.

// model.bin holds each layer's weights (rows x cols, row-major) and then its biases, in forward order.
export const unpack = (floats, shapes) => {
  let k = 0;
  return shapes.map(([rows, cols]) => {
    const w = floats.subarray(k, k += rows * cols);
    const b = floats.subarray(k, k += cols);
    return { rows, cols, w, b };
  });
};

// x: 256 pixel values -> 10 class probabilities
export const forward = (layers, x) => {
  let a = x;
  layers.forEach(({ rows, cols, w, b }, k) => {
    const z = Float64Array.from(b);
    for (let i = 0; i < rows; i++) {
      if (!a[i]) continue; // inputs are binary and relu zeroes about half the activations
      for (let j = 0; j < cols; j++) z[j] += a[i] * w[i * cols + j];
    }
    a = k === layers.length - 1 ? z : z.map(v => (v > 0 ? v : 0));
  });
  const m = Math.max(...a);
  const exps = Array.from(a, v => Math.exp(v - m));
  const total = exps.reduce((sum, e) => sum + e, 0);
  return exps.map(e => e / total);
};

export const argmax = values => values.reduce((best, v, i) => (v > values[best] ? i : best), 0);

// The training drawings fill about 14 of the 16 rows, centred, and the network has only seen digits framed that way
// (moved two cells, its own training drawings drop to 42%). frame() crops a drawing to its ink and scales it to that
// size and place. A cell is ink when at least 35% of the drawing under it is.
export const frame = (pixels, n = 16, height = 14) => {
  let r0 = n, r1 = -1, c0 = n, c1 = -1;
  pixels.forEach((on, i) => {
    if (!on) return;
    const r = Math.floor(i / n), c = i % n;
    r0 = Math.min(r0, r); r1 = Math.max(r1, r); c0 = Math.min(c0, c); c1 = Math.max(c1, c);
  });
  if (r1 < 0) return Uint8Array.from(pixels);
  const h = r1 - r0 + 1, w = c1 - c0 + 1, scale = Math.min(height / h, (n - 1) / w);
  const top = (n - h * scale) / 2, left = (n - w * scale) / 2;
  const out = new Uint8Array(n * n);
  for (let r = 0; r < n; r++) for (let c = 0; c < n; c++) {
    // the patch of the drawing under this cell
    const ra = r0 + (r - top) / scale, rb = ra + 1 / scale, ca = c0 + (c - left) / scale, cb = ca + 1 / scale;
    let ink = 0;
    for (let sr = Math.max(Math.floor(ra), r0); sr < Math.min(Math.ceil(rb), r1 + 1); sr++) {
      for (let sc = Math.max(Math.floor(ca), c0); sc < Math.min(Math.ceil(cb), c1 + 1); sc++) {
        if (pixels[sr * n + sc]) ink += (Math.min(rb, sr + 1) - Math.max(ra, sr)) * (Math.min(cb, sc + 1) - Math.max(ca, sc));
      }
    }
    out[r * n + c] = ink * scale * scale >= 0.35 ? 1 : 0;
  }
  return out;
};
