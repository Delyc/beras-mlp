import numpy as np

from .core import Diffable,Tensor

class Activation(Diffable):
    @property
    def weights(self): return []

    def get_weight_gradients(self): return []
    def compose_weight_gradients(self, J):
        return []
    
class LeakyReLU(Activation):

    def __init__(self, alpha=0.3):
        self.alpha = alpha

    def forward(self, x) -> Tensor:
        """Leaky ReLu forward propagation!"""
        out = np.where(x > 0, x, self.alpha * x)
        return Tensor(out)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Returns diagonal Jacobian for LeakyReLU.
        For element-wise activation, Jacobian is diagonal.
        """
        x = self.inputs[0]
        grad = np.where(x > 0, 1.0, self.alpha)
        
        return [Tensor(grad)]

    def compose_input_gradients(self, J):
        """Element-wise multiplication for diagonal Jacobian"""
        if J is None or J[0] is None:
            return self.get_input_gradients()
        
        inp_grads = self.get_input_gradients()  
        J_out = []
        
        for upstream_jacobian in J:
            if len(upstream_jacobian.shape) == 2:
                result = upstream_jacobian * inp_grads[0]
            else:
                batch_size = upstream_jacobian.shape[0]
                result = upstream_jacobian * inp_grads[0].reshape(batch_size, 1, -1)
            J_out.append(result)
        
        return J_out



class ReLU(LeakyReLU):
    def __init__(self):
        super().__init__(alpha=0)

class Sigmoid(Activation):    
    def forward(self, x) -> Tensor:
        out = 1 / (1 + np.exp(-x))
        return Tensor(out)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Returns diagonal values for Sigmoid Jacobian.
        sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        """
        y = self.outputs[0]
        grad = y * (1 - y)
        
        return [Tensor(grad)]

    def compose_input_gradients(self, J):
        """Element-wise multiplication for diagonal Jacobian"""
        if J is None or J[0] is None:
            return self.get_input_gradients()
        
        inp_grads = self.get_input_gradients() 
        J_out = []
        
        for upstream_jacobian in J:
            if len(upstream_jacobian.shape) == 2:
                result = upstream_jacobian * inp_grads[0]
            else:
                batch_size = upstream_jacobian.shape[0]
                result = upstream_jacobian * inp_grads[0].reshape(batch_size, 1, -1)
            J_out.append(result)
        
        return J_out


class Softmax(Activation):

    def forward(self, x):
        """Softmax forward propagation!"""
        x_shift = x - np.max(x, axis=1, keepdims=True)
        exps = np.exp(x_shift)
        outs = exps / np.sum(exps, axis=1, keepdims=True)
        return Tensor(outs)

    def get_input_gradients(self):
        """
        Softmax input gradients - full Jacobian needed (not diagonal)
        """
        x, y = self.inputs + self.outputs
        batch_size, n = y.shape
        grad = np.zeros((batch_size, n, n))

        for b in range(batch_size):
            y_b = y[b]
            grad[b] = np.diag(y_b) - np.outer(y_b, y_b)

        return [Tensor(grad)]
    
    def compose_input_gradients(self, J):
        """Matrix multiplication for full Jacobian"""
        if J is None or J[0] is None:
            return self.get_input_gradients()
        
        inp_grads = self.get_input_gradients()  
        J_out = []
        
        for upstream_jacobian in J:
            batch_size = upstream_jacobian.shape[0]
            out_dim = upstream_jacobian.shape[1]
            in_dim = inp_grads[0].shape[1]
            
            j_wrt_lay_inp = np.zeros((batch_size, out_dim, in_dim))
            
            for sample in range(batch_size):
                j_wrt_lay_inp[sample] = upstream_jacobian[sample] @ inp_grads[0][sample]
            
            J_out.append(j_wrt_lay_inp)
        
        return J_out
