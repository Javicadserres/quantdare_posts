import numpy as np
from DNet.layers import LinearLayer, Tanh, Base



class RNNCell(Base):
    """
    Recurrent neural network implementation.
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initialize the parameters with the input, output and hidden
        dimensions. 

        Parameters
        ----------
        input_dim : int
            Dimension of the input. 
        output_dim : int
            Dimension of the output.
        hidden_dim : int
            Number of units in the RNN cell.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.lineal_h = LinearLayer(input_dim + hidden_dim, hidden_dim)
        self.lineal_o = LinearLayer(hidden_dim, output_dim)
        self.tanh = Tanh()

        self.parameters_o = [0, 0]
        self.parameters_h = [0, 0]


    def forward(self, input_X, hidden=None):
        """
        Computes the forward propagation of the RNN.

        Parameters
        ----------
        input_X : numpy.array or list
            List containing all the inputs that will be used to 
            propagete along the RNN cell.

        Returns
        -------
        y_preds : list
            List containing all the preditions for each input of the
            input_X list.
        """
        if hidden is None: hidden = np.zeros((self.hidden_dim, 1))

        # combine the input with the hidden state
        combined = np.concatenate((input_X, hidden), axis=1)
        input_hidden = self.lineal_h.forward(combined)
        # hidden state
        hidden = self.tanh.forward(input_hidden)
        # output
        output = self.lineal_o.forward(hidden)

        return output, hidden    


    def backward(self, dZ, d_hidden=0):  
        """
        Computes the backward propagation of the model.

        Parameters
        ----------
        dZ : numpy.array
            The gradient of the of the output with respect to the
            next layer.

        Returns
        -------
        d_output : numpy.array
            The gradient of the input with respect to the current 
            layer.
        d_hidden : numpy.array
            The gradient of the input with respect to the current 
            layer.
        """  
        # derivative of the output
        d_output = self.lineal_o.backward(dZ) + d_hidden
        # derivative of the hyperbolic tangent
        d_tanh = self.tanh.backward(d_output)
        # derivative of the hidden state
        d_hidden = self.lineal_h.backward(d_tanh)
        # update parameters
        self._update_parameters()

        return d_output, d_hidden


    def optimize(self, method):
        """
        Updates the parameters of the model using a given optimize 
        method.

        Parameters
        ----------
        method: Class
            Method to use in order to optimize the parameters.
        """
        for layer in [self.lineal_o, self.tanh, self.lineal_h]:
            layer.optimize(method)

    def _update_parameters(self):
        """
        Updates parameters
        """
        # actualize parameters
        [self.lineal_o.dW, self.lineal_o.db] += self.parameters_o
        [self.lineal_h.dW, self.lineal_h.db] += self.parameters_h
        # retrieve old parameters
        self.parameters_o = [self.lineal_o.dW, self.lineal_o.db]
        self.parameters_h = [self.lineal_h.dW, self.lineal_h.db]


class RNN(Base):
    """
    """
    def __init__(self, input_dim, output_dim, hidden_dim):
        """
        Initialize the parameters with the input, output and hidden
        dimensions. 

        Parameters
        ----------
        input_dim : int
            Dimension of the input. 
        output_dim : int
            Dimension of the output.
        hidden_dim : int
            Number of units in the RNN cell.
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        self.rnn_cell = RNNCell(input_dim, output_dim, hidden_dim)


    def forward(self, input_X, hidden=None):
        """
        Computes the forward propagation of the RNN.

        Parameters
        ----------
        input_X : numpy.array or list
            List containing all the inputs that will be used to 
            propagete along the RNN cell.

        Returns
        -------
        y_preds : list
            List containing all the preditions for each input of the
            input_X list.
        """
        if hidden is None: hidden = np.zeros((self.hidden_dim, 1))

        outputs = []

        for input in input_X:
            output, hidden = self.rnn_cell.forward(input, hidden)
            outputs.append(output.tolist())

        return np.array(output), hidden    


    def backward(self, dZ, d_hidden=0):  
        """
        Computes the backward propagation of the model.

        Parameters
        ----------
        dZ : numpy.array
            The gradient of the of the output with respect to the
            next layer.

        Returns
        -------
        d_output : numpy.array
            The gradient of the input with respect to the current 
            layer.
        d_hidden : numpy.array
            The gradient of the input with respect to the current 
            layer.
        """
        d_hidden = 0

        for dz in dZ:
            d_output, d_hidden = self.rnn_cell.backward(dz, d_hidden)

        return d_output, d_hidden


    def optimize(self, method):
        """
        Updates the parameters of the model using a given optimize 
        method.

        Parameters
        ----------
        method: Class
            Method to use in order to optimize the parameters.
        """
        self.rnn_cell.optimize(method)