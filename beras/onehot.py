import numpy as np

from beras.core import Callable


class OneHotEncoder(Callable):
    """
    One-Hot Encodes labels. First takes in a candidate set to figure out what elements it
    needs to consider, and then one-hot encodes subsequent input datasets in the
    forward pass.

    SIMPLIFICATIONS:
     - Implementation assumes that entries are individual elements.
     - Forward will call fit if it hasn't been done yet; most implementations will just error.
     - keras does not have OneHotEncoder; has LabelEncoder, CategoricalEncoder, and to_categorical()
    """

    def fit(self, data):
        """
        Fits the one-hot encoder to a candidate dataset. Said dataset should contain
        all encounterable elements.

        :param data: 1D array containing labels.
            For example, data = [0, 1, 3, 3, 1, 9, ...]
        """
        unique_labels = np.unique(data)
        num_classes = len(unique_labels)

        eye = np.eye(num_classes, dtype=int)

        self.label_to_onehot = {label: eye[i] for i, label in enumerate(unique_labels)}

        self.onehot_to_label = {tuple(eye[i]): label for i, label in enumerate(unique_labels)}

        self.classes_ = unique_labels

    def forward(self, data):
        if not hasattr(self, "label_to_onehot"):
            self.fit(data)

        onehots = np.array([self.label_to_onehot[label] for label in data])
        return onehots


    def inverse(self, data):
        data = np.asarray(data)
        indices = np.argmax(data, axis=1)
        return np.array([self.classes_[i] for i in indices])
