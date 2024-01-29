import torch.nn as nn
from collections import OrderedDict
import torch

class MLP(nn.Module):
    """
    A multilayer perceptron with ReLU activations and optional BatchNorm.

    Careful: if activation is set to ReLU, ReLU is only applied to the second half of NN outputs! 
            ReLU is applied to standard deviation not mean
    """

    def __init__(
        self,
        sizes,
        batch_norm=True,
        last_layer_act="linear",
    ):
        super(MLP, self).__init__()
        layers = []
        for s in range(len(sizes) - 1):
            layers += [
                nn.Linear(sizes[s], sizes[s + 1]),
                nn.BatchNorm1d(sizes[s + 1])
                if batch_norm and s < len(sizes) - 2
                else None,
                nn.ReLU(),
            ]

        layers = [l for l in layers if l is not None][:-1]
        
        self.activation = last_layer_act
        if self.activation == "linear":
            pass
        elif self.activation == "ReLU":
            self.relu = nn.ReLU()
        else:
            raise ValueError("last_layer_act must be one of 'linear' or 'ReLU'")

        
        layers_dict = OrderedDict(
                {str(i): module for i, module in enumerate(layers)}
            )

        self.network = nn.Sequential(layers_dict)

    def forward(self, x):
        if self.activation == "ReLU":
            x = self.network(x)
            dim = x.size(1) // 2
            return torch.cat((x[:, :dim], self.relu(x[:, dim:])), dim=1)
        return self.network(x)