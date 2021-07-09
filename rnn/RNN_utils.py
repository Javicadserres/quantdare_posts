import numpy as np


class Softmax:
    """
    Class for the Softmax function.
    Applies the softmax function to a given input:
    :math:`Softmax(x_i) = \frac{e^{x_i}}{\sum_{j=1}e^{x_j}}`
    """
    def __init__(self):
        self.type = 'Softmax'
        self.eps = 1e-15

    def forward(self, Z):
        """
        Computes the forward propagation.
        Parameters
        ----------
        Z : numpy.array
            Input.
        Returns
        -------
        A : numpy.array
            Output.
        """
        self.Z = Z

        t = np.exp(Z - np.max(Z, axis=0))
        self.A =  t / np.sum(t, axis=0, keepdims=True)

        return self.A


class Tanh:
    """
    Class for the Hyperbolic tangent activation function.
    Applies the hyperbolic tangent function:
    :math:`Tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}`  
    """
    def __init__(self):
        self.type = 'Tanh'

    def forward(self, Z):
        """
        Computes the forward propagation.
        Parameters
        ----------
        Z : numpy.array
            Input.
        Returns
        -------
        A : numpy.array
            Output.
        """
        self.A = np.tanh(Z)

        return self.A

    def backward(self, dA):
        """
        Computes the backward propagation.
        Parameters
        ----------
        dA : numpy.array
            Gradients of the activation function output.
        Returns
        -------
        dZ : numpy.array
            Gradients of the activation function input.
        """
        dZ = dA * (1 - np.power(self.A, 2))

        return dZ


class CrossEntropyLoss:
    """
    Class that implements the Cross entropy loss function.
    Given a target math:`y` and an estimate math:`\hat{y}` the 
    Cross Entropy loss can be written as:
    .. math::
        \begin{aligned}
            l_{\hat{y}, class} = -\log\left(\frac{\exp(\hat{y_n}[class])}{\sum_j \exp(\hat{y_n}[j])}\right), \\
            L(\hat{y}, y) = \frac{\sum^{N}_{i=1} l_{i, class[i]}}{\sum^{N}_{i=1} weight_{class[i]}},
        \end{aligned}
    References
    ----------
    .. [1] Wikipedia - Cross entropy:
       https://en.wikipedia.org/wiki/Cross_entropy    
    """
    def __init__(self):
        self.type = 'CELoss'
        self.eps = 1e-15
        self.softmax = Softmax()
    
    def forward(self, Y_hat, Y):
        """
        Computes the forward propagation.
        Parameters
        ----------
        Y_hat : numpy.array
            Array containing the predictions.
        Y : numpy.array
            Array with the real labels.
        
        Returns
        -------
        Numpy.arry containing the cost.
        """
        self.Y = Y
        self.Y_hat = Y_hat

        _loss = - Y * np.log(self.Y_hat)
        loss = np.sum(_loss, axis=0).mean()

        return np.squeeze(loss) 

    def backward(self):
        """
        Computes the backward propagation.
        Returns
        -------
        grad : numpy.array
            Array containg the gradients of the weights.
        """
        grad = self.Y_hat - self.Y
        
        return grad


class SGD:
    """
    Class that implements the gradient descent algorithm.
    The formula (with momentum) can be expressed as:
    .. math::
        \begin{aligned}
            v_{t+1} & = \beta * v_{t} + (1 - \beta) * g_{t+1}, \\
            w_{t+1} & = w_{t} - \text{lr} * v_{t+1},
        \end{aligned}
    where :math:`w`, :math:`g`, :math:`v` and :math:`\beta` denote the 
    parameters, gradient, velocity, and beta respectively.
    References
    ----------
    .. [1] Wikipedia - Stochastic gradient descent:
       https://en.wikipedia.org/wiki/Stochastic_gradient_descent
    
    .. [2] Sutskever, Ilya, et al. "On the importance of 
       initialization and momentum in deep learning." International
       conference on machine learning. PMLR, 2013.
       http://www.cs.toronto.edu/~hinton/absps/momentum.pdf
    .. [3] PyTorch - Stochastic gradient descent:
       https://pytorch.org/docs/stable/optim.html
    """
    def __init__(self, lr=0.0075, beta=0.9):
        """
        Parameters
        ----------
        lr : int, default: 0.0075
            Learing rate to use for the gradient descent.
        beta : int, default: 0.9
            Beta parameter.
        """
        self.beta = beta
        self.lr = lr

    def optim(self, weights, gradients, velocities=None):
        """
        Parameters
        ---------
        weights : numpy.array
            Weigths of a given layer.
        bias : numpy.array
            Bias of a given layer.
        dW : numpy.array
            The gradients of the weights.
        db : numpy.array
            The gradients of the bias
        velocities : tuple
            Tuple containing the velocities to compute the gradient
            descent with momentum.
        Returns
        -------
        weights : numpy.array
            Updated weigths of the given layer.
        bias : numpy.array
            Updated bias of the given layer.
        (V_dW, V_db) : tuple
            Tuple of ints containing the velocities for the weights
            and biases.
        """
        if velocities is None: velocities = [0 for weight in weights]

        velocities = self._update_velocities(
            gradients, self.beta, velocities
        )
        new_weights = []

        for weight, velocity in zip(weights, velocities):
            weight -= self.lr * velocity
            new_weights.append(weight)

        return new_weights, velocities

    def _update_velocities(self, gradients, beta, velocities):
        """
        Updates the velocities of the derivates of the weights and 
        bias.
        """
        new_velocities = []

        for gradient, velocity in zip(gradients, velocities):

            new_velocity = beta * velocity + (1 - beta) * gradient
            new_velocities.append(new_velocity)

        return new_velocities


def one_hot_encoding(input, size):
    """
    Do one hot encoding for a given input and size.
    
    Parameters
    ----------
    input : list
        list containing the numbers to make the 
        one hot encoding
    size : int
        Maximum size of the one hot encoding.
        
    Returns
    -------
    output : list
        List with the one hot encoding arrays.
    """
    output = []

    for index, num in enumerate(input):
        one_hot = np.zeros((size,1))

        if (num != None):
            one_hot[num] = 1
    
        output.append(one_hot)

    return output