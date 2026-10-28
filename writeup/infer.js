// inference.py's forward pass, on the float32 weights in results/model.bin. Shared by the Results tab and
// scripts/build-results.mjs, so the accuracy it prints is the accuracy of exactly what the page runs.

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
