import numpy as np

from typing import Literal
from beras.core import Diffable, Variable, Tensor

DENSE_INITIALIZERS = Literal["zero", "normal", "xavier", "kaiming", "xavier uniform", "kaiming uniform"]

class Dense(Diffable):

    def __init__(self, input_size, output_size, initializer: DENSE_INITIALIZERS = "normal"):
        self.w, self.b = self._initialize_weight(initializer, input_size, output_size)

    @property
    def weights(self) -> list[Tensor]:
        return self.w, self.b

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for a dense layer! Refer to lecture slides for how this is computed.
        """
        return Tensor(x @ self.w + self.b)

    def get_input_gradients(self) -> list[Tensor]:
        """
        Returns the Jacobian of outputs w.r.t. inputs.
        For Dense layer: dY/dX where Y = XW + b
        Shape: (batch, output_dim, input_dim)
        """
        return [Tensor(self.w)]
       

    def compose_input_gradients(self, J):
        """
        Override for Dense layer efficiency.
        Composes upstream gradients with this layer's input gradients.
        """
        if J is None or J[0] is None:
            return self.get_input_gradients()
        
        J_out = []
        for upstream_jacobian in J:
            j_wrt_lay_inp = upstream_jacobian @ self.w.T
            J_out.append(j_wrt_lay_inp)
        return J_out
    
    def get_weight_gradients(self) -> list[Tensor]:
        """
        Returns the local Jacobian of outputs w.r.t. weights.
        For Dense layer: dY/dW and dY/db where Y = XW + b
        """
        x = self.inputs[0]  
        batch_size = x.shape[0]
        output_dim = self.w.shape[1]
        
        # Gradient w.r.t W: (batch, input_dim, output_dim)
        grad_W = np.repeat(x[:, :, np.newaxis], output_dim, axis=2)
        
        # Gradient w.r.t b: (output_dim,) - NOT batched!
        grad_b = np.ones(output_dim)
        
        return [Tensor(grad_W), Tensor(grad_b)]
    
    def compose_weight_gradients(self, J):
        """
        Override for Dense layer to properly compute weight gradients.
        """
        if J is None or J[0] is None:
            return self.get_weight_gradients()
        
        x = self.inputs[0]  
        batch_size = x.shape[0]
        
        
        grad_W_total = np.zeros_like(self.w)
        grad_b_total = np.zeros_like(self.b)
        
        for upstream_jacobian in J:
          
            if len(upstream_jacobian.shape) == 3:
                batch_size_up = upstream_jacobian.shape[0]
                output_dim = upstream_jacobian.shape[-1]
                upstream = upstream_jacobian.reshape(batch_size_up, -1)[:, :output_dim]
            else:
                upstream = upstream_jacobian  
            
            grad_W_total += (x.T @ upstream)
          
            grad_b_total += np.sum(upstream, axis=0)
        
       
        grad_W_total /= batch_size
        grad_b_total /= batch_size
        
        return [Tensor(grad_W_total), Tensor(grad_b_total)]

    
    @staticmethod
    def _initialize_weight(initializer, input_size, output_size) -> tuple[Variable, Variable]:
        """
        Initializes the values of the weights and biases. The bias weights should always start at zero.
        However, the weights should follow the given distribution defined by the initializer parameter
        (zero, normal, xavier, or kaiming). You can do this with an if statement
        cycling through each option!

        Details on each weight initialization option:
            - Zero: Weights and biases contain only 0's. Generally a bad idea since the gradient update
            will be the same for each weight so all weights will have the same values.
            - Normal: Weights are initialized according to a normal distribution.
            - Xavier: Goal is to initialize the weights so that the variance of the activations are the
            same across every layer. This helps to prevent exploding or vanishing gradients. Typically
            works better for layers with tanh or sigmoid activation.
            - Kaiming: Similar purpose as Xavier initialization. Typically works better for layers
            with ReLU activation.
        """

        initializer = initializer.lower()
        assert initializer in (
            "zero",
            "normal",
            "xavier",
            "kaiming",
        ), f"Unknown dense weight initialization strategy '{initializer}' requested"

        if initializer == "zero":
            w = np.zeros((input_size, output_size))
        elif initializer == "normal":
            w = np.random.normal(0, 1, (input_size, output_size))
        elif initializer == "xavier":
            std = np.sqrt(2.0 / (input_size + output_size))
            w = np.random.normal(0, std, (input_size, output_size))
        elif initializer == "kaiming":
            std = np.sqrt(2.0 / input_size)
            w = np.random.normal(0, std, (input_size, output_size))

        b = np.zeros((output_size,))
        return Variable(w), Variable(b)
