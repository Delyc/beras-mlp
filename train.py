from types import SimpleNamespace
from beras.activations import ReLU, LeakyReLU, Softmax
from beras.layers import Dense
from beras.losses import CategoricalCrossEntropy, MeanSquaredError
from beras.metrics import CategoricalAccuracy
from beras.onehot import OneHotEncoder
from beras.optimizers import Adam
from preprocess import load_and_preprocess_data
import numpy as np

from beras.model import SequentialModel

def get_model():
    model = SequentialModel(
        [
           Dense(784, 256, initializer="kaiming"),  
            LeakyReLU(alpha=0.01),  
            Dense(256, 128, initializer="kaiming"),
            LeakyReLU(alpha=0.01),
            Dense(128, 10, initializer="kaiming"),
            Softmax(),
        ]
    )
    return model

def get_optimizer():
    return Adam(learning_rate=0.001)

def get_loss_fn():
    return CategoricalCrossEntropy()

def get_acc_fn():
    return CategoricalAccuracy()

if __name__ == '__main__':

    train_inputs, train_labels, test_inputs, test_labels = load_and_preprocess_data()

    encoder = OneHotEncoder()
    encoder.fit(train_labels)
    train_labels = encoder.forward(train_labels)
    test_labels = encoder.forward(test_labels)

    model = get_model()
    model.compile(
        optimizer=get_optimizer(),
        loss_fn=get_loss_fn(),
        acc_fn=get_acc_fn()
    )

    print("\n Training started...\n")
    model.fit(train_inputs, train_labels, epochs=10, batch_size=64)

    print("\n Evaluating on test set...\n")
    test_results = model.evaluate(test_inputs, test_labels, batch_size=64)

    acc = test_results['acc']
    if isinstance(acc, list):
        acc = acc[0]  

    loss = test_results['loss']
    if isinstance(loss, list):
        loss = loss[0]

    print("\n" + "="*50)
    print("FINAL RESULTS:")
    print(f"Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Test Loss: {loss:.4f}")
    print("="*50)

    with open('FINAL.txt', 'w') as f:
        f.write(f"Local test accuracy: {acc:.4f}\n")
        f.write(f"Local test loss: {loss:.4f}\n")

    print("\n FINAL.txt updated with results.")
