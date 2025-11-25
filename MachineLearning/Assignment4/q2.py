import numpy as np

from typing import List
from q1 import Layer, Dense

## SET GLOBAL SEED
## Do not modify this for reproducibility
np.random.seed(33)

"""## **Question 2: Putting it all together: MLP**

Now, we will put everything together and implement an MLP (multi-layer perceptron) class which is capable enough of stacking multiple layers.
"""

class MLP(Layer):
    """
    Multi-layer perceptron.
    """
    def __init__(self, layers: List[Layer]):
        """
        Initialize the MLP object. The passed list of layers usually
        follows the order: [Dense, Activation, Dense, Activation, ...]
        Parameters:
            layers (list): list of layers of the MLP
        """
        super().__init__()
        self.layers = layers
        self.init_weights()

    def init_weights(self):
        """
        Initialize the weights of the MLP.
        By default, each Dense layer will use the Kaiming initialization.
        Parameters:
            seed (int): seed for random number generation
        """
        # BEGIN SOLUTIONS
        # END SOLUTIONS

            
    def forward(self, input):
        """
        Forward pass of the MLP.
        Parameter:
            input (np.ndarray): input of the MLP, shape: (batch_size, input_size)
                                (NOTE: input_size is the size of the input of the first layer)
        Returns:
            output (np.ndarray): output of the MLP, shape: (batch_size, output_size)
                                 (NOTE: output_size is the size of the output of the last layer)
        """
        # BEGIN SOLUTIONS
        pass
        # END SOLUTIONS

    def backward(self, output_grad):
        """
        Backward pass of the MLP.
        Parameter:
            output_grad (np.ndarray): gradient of the output of the MLP (dy)
        Returns:
            input_grad (np.ndarray): gradient of the input of the MLP (dx)
        """
        # BEGIN SOLUTIONS
        pass
        # END SOLUTIONS

    def update(self, learning_rate):
        """
        Update the MLP parameters. Normally, this is done by using the
        gradients computed in the backward pass; therefore, .backward() must
        be called before update(). In other words, this function assumes that
        the gradients are already computed via backward pass.
        Parameter:
            learning_rate (float): learning rate used for updating
        """
        # assumes self.backward() function has been called before
        # BEGIN SOLUTIONS
        pass
        # END SOLUTIONS
