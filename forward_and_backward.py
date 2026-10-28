from typing import List

from draw_dataset import load_dataset
from math import *
from random import *

from neuron import Neuron
from operations import multi, relu, sftmx, inverse_log_max
from model_io import save_model

import os
from collections import deque
from time import perf_counter
LR = 3e-3

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "train_log.txt")   # plot with: python plot_loss.py train_log.txt
MODEL_PATH = os.path.join(HERE, "model.json")    # serve with: python serve.py
CHECKPOINT_EVERY = 100

### Get input and labels
dataset = load_dataset()
X = [[Neuron(value=v) for v in sample["x"]] for sample in dataset]
Y = [sample["y"] for sample in dataset]

##### ##### #####
### Building the network
input_dim = 256
hw1 =512
hw2 = 256
hw3 = 10

def init_neuron(fan_in):
    """Weights/biases start in +-1/sqrt(fan_in) so the logits don't explode."""
    bound = 1 / sqrt(fan_in)
    return Neuron(value=round(uniform(-bound, bound), 5))

### MLP layer 1 
w1 = []
b1 = [init_neuron(input_dim) for _ in range(hw1)]
for row in range(input_dim):
    w1.append([init_neuron(input_dim) for _ in range(hw1)])

### MLP Layer 2
w2 = []
b2 = [init_neuron(hw1) for _ in range(hw2)]
for row in range(hw1):
    w2.append([init_neuron(hw1) for _ in range(hw2)])

### MLP Layer 3
w3 = []
b3 = [init_neuron(hw2) for _ in range(hw3)]
for row in range(hw2):
    w3.append([init_neuron(hw2) for _ in range(hw3)])
##### ##### #####

# only the weights and biases are parameters, everything else is recomputed every forward pass
params = {n for layer in (w1, w2, w3) for row in layer for n in row} | {*b1, *b2, *b3}

layers = [(w1, b1), (w2, b2), (w3, b3)]
log_file = open(LOG_PATH, "w")

last_log = perf_counter()
for fp in range(10000):
    # One forward pass, for now
    chosen_dataset_entry = randint(5, 995)
    features = X[chosen_dataset_entry]

    # hidden dimension 1
    z1 = multi(features, w1, b1)

    # activation 1 
    a1 = relu(z1)

    # hidden dimension 2
    z2 = multi(a1, w2, b2)

    # activation 2
    a2 = relu(z2)

    # logits
    l = multi(a2, w3, b3)
    probs = sftmx(l) # neurons holding probabilities

    LABEL = Y[chosen_dataset_entry]
    loss = inverse_log_max(probs=probs, label=LABEL)

    ##### COMPUTE GRADIENTS ####
    now = perf_counter()
    line = f"iter {fp}: loss {loss.value:.4f}  ({now - last_log:.2f}s since last iter)"
    print(line)
    log_file.write(line + "\n")
    log_file.flush()
    last_log = now
    loss.grad = 1.0

    queue = deque([loss])
    seen = {loss}
    while queue:
        this_neuron = queue.popleft()
        for parent in this_neuron.parents:
            parent.grad += parent.local_grad_to(this_neuron) * this_neuron.grad
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)
        if this_neuron in params:
            this_neuron.value -= LR * this_neuron.grad

    # reset the parameters and cut their links into this iteration's graph, so it can be freed
    for p in params:
        p.grad = 0.0
        p.local_grads.clear()
    for f in features:
        f.local_grads.clear()

    if (fp + 1) % CHECKPOINT_EVERY == 0:
        save_model(MODEL_PATH, layers)

save_model(MODEL_PATH, layers)
log_file.close()
print("DONE!!!")
        

            

    





