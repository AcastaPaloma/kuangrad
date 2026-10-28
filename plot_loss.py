"""Plot loss against wall-clock training time from a saved training log.

    python plot_loss.py train_log.txt            # -> loss_vs_time.svg
    python plot_loss.py pasted_terminal.txt -o out.svg --window 200
    pbpaste | python plot_loss.py -              # straight from the clipboard (macOS)

Reads lines like `iter 12: loss 2.2931  (3.41s since last iter)`; anything else in the
file is ignored. Standard library only, the output is an SVG you can open in a browser.
"""
import argparse
import re
import sys
from math import ceil, floor, log10

LINE = re.compile(r"iter (\d+): loss ([0-9.]+(?:e[-+]?\d+)?)\s+\(([0-9.]+)s since last iter\)")


def parse(text):
    """-> (iters, losses, cumulative seconds). Repeated iters (e.g. a log pasted twice) keep the last one."""
    rows = {}
    for m in LINE.finditer(text):
        rows[int(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    iters = sorted(rows)
    losses = [rows[i][0] for i in iters]
    elapsed, total = [], 0.0
    for i in iters:
        total += rows[i][1]
        elapsed.append(total)
    return iters, losses, elapsed


def moving_average(values, window):
    out, running = [], 0.0
    for i, v in enumerate(values):
        running += v
        if i >= window:
            running -= values[i - window]
        out.append(running / min(i + 1, window))
    return out


def nice_ticks(lo, hi, target=6):
    span = hi - lo or 1.0
    raw = span / target
    mag = 10 ** floor(log10(raw))
    step = min((s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw), default=10 * mag)
    first = ceil(lo / step) * step
    ticks, t = [], first
    while t <= hi + 1e-9:
        ticks.append(round(t, 10))
        t += step
    return ticks


def pick_unit(seconds):
    if seconds < 120:
        return "seconds", 1.0
    if seconds < 2 * 3600:
        return "minutes", 60.0
    return "hours", 3600.0


def build_svg(iters, losses, elapsed, window):
    unit, div = pick_unit(elapsed[-1])
    xs_raw = [t / div for t in elapsed]
    smooth = moving_average(losses, window)

    W, H = 960, 520
    L, R, T, B = 70, 30, 50, 60
    x_hi = xs_raw[-1] or 1.0
    y_hi = max(losses)
    y_ticks = nice_ticks(0, y_hi * 1.05)
    y_hi = y_ticks[-1] if y_ticks[-1] >= y_hi else y_hi * 1.05
    x_ticks = nice_ticks(0, x_hi)

    sx = lambda x: L + (W - L - R) * x / x_hi
    sy = lambda y: H - B - (H - T - B) * y / y_hi
    poly = lambda xs, ys: " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="system-ui, sans-serif" font-size="12">',
        f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
        f'<text x="{L}" y="28" font-size="16" font-weight="600" fill="#222">Training loss vs time</text>',
        f'<text x="{W - R}" y="28" text-anchor="end" fill="#666">{len(iters)} iterations, '
        f'{x_hi:.2f} {unit}, final avg loss {smooth[-1]:.3f}</text>',
    ]
    for y in y_ticks:
        out.append(f'<line x1="{L}" x2="{W - R}" y1="{sy(y):.1f}" y2="{sy(y):.1f}" stroke="#e5e5e5"/>')
        out.append(f'<text x="{L - 8}" y="{sy(y) + 4:.1f}" text-anchor="end" fill="#555">{y:g}</text>')
    for x in x_ticks:
        out.append(f'<line x1="{sx(x):.1f}" x2="{sx(x):.1f}" y1="{H - B}" y2="{H - B + 5}" stroke="#888"/>')
        out.append(f'<text x="{sx(x):.1f}" y="{H - B + 20}" text-anchor="middle" fill="#555">{x:g}</text>')
    out += [
        f'<line x1="{L}" x2="{L}" y1="{T}" y2="{H - B}" stroke="#888"/>',
        f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="#888"/>',
        f'<text x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle" fill="#333">time ({unit})</text>',
        f'<text transform="translate(18 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle" fill="#333">loss</text>',
        f'<polyline points="{poly(xs_raw, losses)}" fill="none" stroke="#2a6fdb" stroke-opacity="0.25" stroke-width="1"/>',
        f'<polyline points="{poly(xs_raw, smooth)}" fill="none" stroke="#c4321a" stroke-width="2"/>',
        f'<line x1="{W - R - 230}" x2="{W - R - 210}" y1="{T + 14}" y2="{T + 14}" stroke="#2a6fdb" stroke-opacity="0.5"/>',
        f'<text x="{W - R - 204}" y="{T + 18}" fill="#555">per-iteration loss (single sample)</text>',
        f'<line x1="{W - R - 230}" x2="{W - R - 210}" y1="{T + 34}" y2="{T + 34}" stroke="#c4321a" stroke-width="2"/>',
        f'<text x="{W - R - 204}" y="{T + 38}" fill="#555">moving average ({window})</text>',
        "</svg>",
    ]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="training log file, or - for stdin")
    ap.add_argument("-o", "--output", default="loss_vs_time.svg")
    ap.add_argument("--window", type=int, default=100, help="moving-average window in iterations")
    args = ap.parse_args()

    text = sys.stdin.read() if args.log == "-" else open(args.log, errors="replace").read()
    iters, losses, elapsed = parse(text)
    if not iters:
        sys.exit("no 'iter N: loss X (Ts since last iter)' lines found")

    with open(args.output, "w") as f:
        f.write(build_svg(iters, losses, elapsed, args.window))
    avg = moving_average(losses, args.window)
    print(f"{len(iters)} iterations, {elapsed[-1] / 3600:.2f} h; "
          f"first-{args.window} avg {sum(losses[:args.window]) / min(len(losses), args.window):.4f}, "
          f"last-{args.window} avg {avg[-1]:.4f}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
