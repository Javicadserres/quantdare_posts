import torch.nn as nn


class LSTM(nn.Module):
    """
    NN model with a LSTM layer.

    Parameters
    ----------
    input_size: int
        This is the number of features.
    out_size: int
        Number of targets.
    hidden_size: int, default=1
    n_layers: int, default=1
    """
    def __init__(self, input_size, out_size, hidden_size=50, n_layers=3):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True   
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        """
        Forward propagation.

        Parameters
        ----------
        x: torch.tensor
        """
        out, (h, c) = self.lstm(x)
        out = self.lin(out)

        return out[:, -1, :]

class AttentionLSTM(nn.Module):
    """
    LSTM using a multihead attention mechanism.

    Parameters
    ----------
    embed_dim: int
    out_size: int
    hidden_size: int, default=20    
    n_layers: int, default=2
    """
    def __init__(self, embed_dim, out_size, hidden_size=20, n_layers=2):
        super(AttentionLSTM, self).__init__()
        self.att = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=embed_dim
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=n_layers
        )
        self.lin = nn.Linear(hidden_size, out_size)

    def forward(self, X):
        out, weights = self.att(X, X, X)
        out, (h, c) = self.lstm(out)
        out = self.lin(out)
        return out[:, -1, :]
