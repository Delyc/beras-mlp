from collections import defaultdict

from beras.core import Diffable, Tensor
import numpy as np


class GradientTape:

    def __init__(self):
        self.previous_layers: defaultdict[int, Diffable | None] = defaultdict(lambda: None)

    def __enter__(self):
        if Diffable.gradient_tape is not None:
            raise RuntimeError("Cannot nest gradient tape scopes.")

        Diffable.gradient_tape = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        Diffable.gradient_tape = None

    def gradient(self, target: Tensor, sources: list[Tensor]) -> list[Tensor]:
        """
        Computes the gradient of the target tensor with respect to the sources.

        :param target: the tensor to compute the gradient of, typically loss output
        :param sources: the list of tensors to compute the gradient with respect to
        In order to use tensors as keys to the dictionary, use the python built-in ID function here: https://docs.python.org/3/library/functions.html#id.
        """
     
        queue = [target]                    ## Live queue; will be used to propagate backwards via breadth-first-search.
        grads = defaultdict(lambda: None)   ## Grads to be recorded. Initialize to None. Note: stores {id: list[gradients]}
        # Use id(tensor) to get the object id of a tensor object.
        # in the end, your grads dictionary should have the following structure:
        # {id(tensor): [gradient]}

        # What tensor and what gradient is for you to implement!
        # compose_input_gradients and compose_weight_gradients are methods that will be helpful

        loss_layer = self.previous_layers.get(id(target), None)
        if loss_layer is None:
            raise RuntimeError("No loss layer found for target tensor.")

        upstream_jacobians = loss_layer.get_input_gradients()  
        for inp, grad in zip(loss_layer.inputs, upstream_jacobians):
            grads[id(inp)] = [grad]
            queue.append(inp)

        while queue:
            tensor = queue.pop(0)
            layer = self.previous_layers.get(id(tensor), None)
            if layer is None:
                continue

            upstream_jacobians = grads[id(tensor)]
            input_jacobians = layer.compose_input_gradients(upstream_jacobians)
            weight_jacobians = layer.compose_weight_gradients(upstream_jacobians)

            for inp, g in zip(layer.inputs, input_jacobians):
                if grads[id(inp)] is None:
                    grads[id(inp)] = [g]
                    queue.append(inp)
                else:
                    grads[id(inp)].append(g)

            for w, g in zip(layer.weights, weight_jacobians):
                if grads[id(w)] is None:
                    grads[id(w)] = [g]
                else:
                    grads[id(w)].append(g)

        output_grads = []
        for src in sources:
            g_list = grads.get(id(src), None)
            if g_list is None:
                output_grads.append(Tensor(np.zeros_like(src)))
            else:
                total = np.sum(g_list, axis=0)
                output_grads.append(Tensor(total))

        return output_grads

