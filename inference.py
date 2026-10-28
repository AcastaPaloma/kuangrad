"""Forward pass on plain floats: same maths as multi / relu / sftmx, without building a graph."""
from math import exp

from model_io import load_model


def forward(layers, x):
    """x: 256 pixel values -> list of 10 class probabilities."""
    a = x
    for k, (weights, biases) in enumerate(layers):
        z = list(biases)
        for a_i, row in zip(a, weights):
            if a_i:  # inputs are binary and relu zeroes about half the activations, so skipping zeros is a big saving
                for j, w_ij in enumerate(row):
                    z[j] += a_i * w_ij
        a = z if k == len(layers) - 1 else [v if v > 0 else 0.0 for v in z]

    m = max(a)
    exps = [exp(v - m) for v in a]
    total = sum(exps)
    return [e / total for e in exps]


def predict(layers, x):
    probs = forward(layers, x)
    digit = max(range(len(probs)), key=probs.__getitem__)
    return digit, probs


if __name__ == "__main__":
    import sys

    from draw_dataset import load_dataset

    path = sys.argv[1] if len(sys.argv) > 1 else "model.json"
    layers = load_model(path)
    data = load_dataset()
    correct = sum(predict(layers, s["x"])[0] == s["y"] for s in data)
    print(f"{path}: {correct}/{len(data)} correct on dataset.json ({correct / len(data):.1%}); "
          "note the training loop samples from this same data, so this is training accuracy")
