"""
Serve a saved model for inference.

    python serve.py [model.json] [--port 8000]

Open http://localhost:8000 and draw a digit, or call the API:

    curl -s localhost:8000/predict -d '{"x": [0, 1, 0, ... 256 values ...]}'
    -> {"digit": 3, "probs": [...10 floats...]}
"""
import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

from inference import predict
from model_io import load_model

GRID = 16

PAGE = """<!doctype html>
<meta charset="utf-8"><title>kuangrad</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 2rem; }
  #grid { display: grid; grid-template-columns: repeat(%(n)d, 24px); gap: 1px; background: #ccc; width: max-content; user-select: none; }
  #grid div { width: 24px; height: 24px; background: #fff; }
  #grid div.on { background: #000; }
  #out { margin-top: 1rem; font-size: 1.1rem; }
  .bar { display: flex; align-items: center; gap: .5rem; font-variant-numeric: tabular-nums; }
  .bar span { display: inline-block; height: 10px; background: #2a6fdb; }
</style>
<p>Draw with the mouse (hold shift to erase).</p>
<div id="grid"></div>
<p><button id="clear">Clear</button></p>
<div id="out"></div>
<script>
const N = %(n)d, cells = [], grid = document.getElementById("grid"), out = document.getElementById("out");
let drawing = false;
for (let i = 0; i < N * N; i++) {
  const d = document.createElement("div");
  d.onmousedown = e => { drawing = true; paint(i, e); };
  d.onmouseenter = e => { if (drawing) paint(i, e); };
  grid.appendChild(d); cells.push(d);
}
addEventListener("mouseup", () => drawing = false);
function paint(i, e) { cells[i].classList.toggle("on", !e.shiftKey); predict(); }
document.getElementById("clear").onclick = () => { cells.forEach(c => c.classList.remove("on")); out.textContent = ""; };
async function predict() {
  const x = cells.map(c => c.classList.contains("on") ? 1 : 0);
  const r = await (await fetch("/predict", { method: "POST", body: JSON.stringify({ x }) })).json();
  out.innerHTML = "<b>Prediction: " + r.digit + "</b>" + r.probs.map((p, d) =>
    '<div class="bar">' + d + ' <span style="width:' + Math.round(p * 200) + 'px"></span> ' + (p * 100).toFixed(1) + '%%</div>').join("");
}
</script>
""" % {"n": GRID}


def make_handler(layers, n_inputs):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code, body, content_type):
            data = body.encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/":
                self._send(200, PAGE, "text/html; charset=utf-8")
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path != "/predict":
                return self._send(404, "not found", "text/plain")
            try:
                length = int(self.headers.get("Content-Length", 0))
                x = [float(v) for v in json.loads(self.rfile.read(length))["x"]]
                if len(x) != n_inputs:
                    raise ValueError(f"expected {n_inputs} values, got {len(x)}")
            except (ValueError, KeyError, TypeError) as err:
                return self._send(400, json.dumps({"error": str(err)}), "application/json")
            digit, probs = predict(layers, x)
            self._send(200, json.dumps({"digit": digit, "probs": probs}), "application/json")

        def log_message(self, *args):
            pass

    return Handler


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="model.json")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    layers = load_model(args.model)
    n_inputs = len(layers[0][0])
    print(f"loaded {args.model} ({n_inputs} inputs, {len(layers)} layers); serving on http://localhost:{args.port}")
    HTTPServer(("127.0.0.1", args.port), make_handler(layers, n_inputs)).serve_forever()
