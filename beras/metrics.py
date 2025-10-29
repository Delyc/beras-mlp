import numpy as np

from beras.core import Callable


class CategoricalAccuracy(Callable):
    def forward(self, probs, labels):
        pred_classes = np.argmax(probs, axis=1)
        true_classes = np.argmax(labels, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
        return accuracy
