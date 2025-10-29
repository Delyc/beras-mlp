import numpy as np

from beras.core import Diffable, Tensor

import tensorflow as tf


class Loss(Diffable):
    @property
    def weights(self) -> list[Tensor]:
        return []

    def get_weight_gradients(self) -> list[Tensor]:
        return []
    
class MeanSquaredError(Loss):
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        diff = y_pred - y_true             
        per_sample = np.mean(diff**2, axis=1)  
        return Tensor(np.mean(per_sample))     

    def get_input_gradients(self) -> list[Tensor]:
        y_pred, y_true = self.inputs
        batch_size, n = y_pred.shape
        grad_pred = (2.0 / n) * (y_pred - y_true) / batch_size   
        grad_true = -grad_pred                                   
        return [Tensor(grad_pred), Tensor(grad_true)]


class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        """Categorical cross entropy forward pass!"""
        eps = 1e-12
        y = np.clip(y_pred, eps, 1 - eps)
        per_sample = -np.sum(y_true * np.log(y), axis=1)  
        return Tensor(np.mean(per_sample))                

    def get_input_gradients(self):
        """Categorical cross entropy input gradient method!"""
        y_pred, y_true = self.inputs
        batch_size, _ = y_pred.shape
        eps = 1e-12
        y = np.clip(y_pred, eps, 1 - eps)
        grad_pred = -(y_true / y) / batch_size            
        grad_true = np.zeros_like(grad_pred)              
        return [Tensor(grad_pred), Tensor(grad_true)]


