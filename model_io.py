import json
import os


def save_model(path, layers):
    """
    Save the trained parameters as JSON.

    layers: [(weights, biases), ...] where weights is rows x cols of Neuron and biases is a list of Neuron,
    in forward order. Written to a temp file first so a checkpoint is never half-written if training is killed.
    """
    data = {
        "layers": [
            {"w": [[n.value for n in row] for row in weights], "b": [n.value for n in biases]}
            for weights, biases in layers
        ]
    }
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def load_model(path):
    """-> [(weights, biases), ...] as plain floats (no Neuron objects, no graph)."""
    with open(path) as f:
        data = json.load(f)
    return [(layer["w"], layer["b"]) for layer in data["layers"]]
