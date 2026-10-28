from typing import List
from neuron import Neuron

from math import e, log

eps = 1e-15

def multi(features: List[Neuron], matrix: List[List[Neuron]], bias: List[Neuron]) -> List[Neuron]:
    """
    One forward layer for MLPs
    """

    row_count = len(matrix)
    col_count = len(matrix[0])

    features_count = len(features)

    if row_count != features_count:
        print("ERROR: Feature count doesn't match row count of weight matrix.")
        return

    next_features = []

    # Start creating the output matrix (its shape is the amount of columns in the weight matrix)
    for col_i in range(col_count):
        new_feature = Neuron()

        # Multiply every feature by its weight
        for feature_i in range(features_count):
            weight = matrix[feature_i][col_i]
            feature = features[feature_i]
            new_feature.value += weight.value * feature.value

            # THIS POINTS TO THE OBJECT ITSELF! DON'T THINK USES UNNECESSARY MEMORY (NOT DEEPCOPIES)
            # d(output)/d(feature) is the weight, d(output)/d(weight) is the feature
            new_feature.parents += [feature, weight]

            weight.local_grads.append([new_feature, weight, feature.value])
            feature.local_grads.append([new_feature, feature, weight.value])

        # Add bias, d(output)/d(bias) is 1
        this_bias = bias[col_i]
        new_feature.value += this_bias.value
        this_bias.local_grads.append([new_feature, this_bias, 1.0])

        new_feature.parents.append(bias[col_i])
        next_features.append(new_feature)

    return next_features

def relu(features: list[Neuron]) -> list[Neuron]:
    """Classic activation function returning new Neuron objects."""
    activations = []
    for feature in features:
        val = feature.value
        local_grad = 1.0
        if val <= 0:
            val = 0
            local_grad = 0.0

        activation = Neuron(value=val, parents=[feature])
        activations.append(
            activation
            )

        feature.local_grads.append([activation, feature, local_grad])
        
    return activations

def sftmx(features: List[Neuron]) -> List[Neuron]:
    """
    Classic softmax function. Could be reused not only on logits, therefore not naming argumebt as 'logits'.
    """
    max_logit = max(feature.value for feature in features)
    exps = [e**(feature.value - max_logit) for feature in features]
    denom = sum(exps)
    probs = [exp / denom for exp in exps]

    # Every output depends on every input: dp_i/dz_j = p_i * ((1 if i == j else 0) - p_j)
    outputs = [Neuron(value=p_i, parents=list(features)) for p_i in probs]

    for j, feature in enumerate(features):
        for i, output in enumerate(outputs):
            local_grad = probs[i] * ((1.0 if i == j else 0.0) - probs[j])
            feature.local_grads.append([output, feature, local_grad])

    return outputs

def inverse_log_max(probs: List[Neuron], label) -> Neuron:
    """
    Classic loss function for continuous classification
    """

    loss = Neuron(value=-log(max(probs[label].value, eps)), grad=1, parents=list(probs))
    for i, prob in enumerate(probs):
        local_grad = -1 / prob.value if i == label and prob.value > eps else 0.0
        prob.local_grads.append([loss, prob, local_grad])

    return loss

    

