import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        n_hidden: int = 3,
        nodes_per_layer: int = 3,
        activation: callable = nn.ReLU(),
        output_activation: callable = None,
        batchnorm: bool = False,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_hidden = n_hidden
        self.nodes_per_layer = nodes_per_layer
        self.activation = activation
        self.output_activation = output_activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.layers = nn.Sequential()

        if self.n_hidden == 0:
            self.nodes_per_layer = [self.n_inputs]
        elif type(self.nodes_per_layer) == int:
            self.nodes_per_layer = [self.nodes_per_layer] * self.n_hidden
        # elif type(self.nodes_per_layer) == tuple:
        #     assert len(self.nodes_per_layer) == self.n_hidden,
        #       "nodes_per_layer must be a tuple of length n_hidden"

        for layer in range(self.n_hidden):
            if layer == 0:
                feats_in = self.n_inputs
            else:
                feats_in = self.nodes_per_layer[layer - 1]

            feats_out = self.nodes_per_layer[layer]

            self.layers.add_module(
                "hidden_" + str(layer + 1), nn.Linear(feats_in, feats_out)
            )

            if self.batchnorm:
                self.layers.add_module(
                    "batchnorm_" + str(layer + 1), nn.BatchNorm1d(feats_out)
                )

            if self.dropout:
                self.layers.add_module(
                    "dropout_" + str(layer + 1), nn.Dropout(p=self.dropout)
                )

            self.layers.add_module("activation_" + str(layer + 1), self.activation)

        self.layers.add_module(
            "output", nn.Linear(self.nodes_per_layer[-1], self.n_outputs)
        )

        if self.output_activation:
            self.layers.add_module("output_activation", self.output_activation)

    def forward(self, x):
        return self.layers(x)
