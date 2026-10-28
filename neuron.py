from typing import List

class Neuron:
    def __init__(self, value=0, grad=0, parents=None, local_grads=None):
        self.value: float = value
        self.grad: float = grad # dL/dself
        self.parents: List[Neuron] = parents if parents is not None else []

        # one [child, self, dchild/dself] entry per neuron that uses this one
        self.local_grads: List[list] = local_grads if local_grads is not None else []

    def local_grad_to(self, child):
        """dchild/dself for one specific child."""
        for c, _, grad in self.local_grads:
            if c is child:
                return grad

