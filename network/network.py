from torch import nn

class DynamicNet(nn.Module):
    def __init__(self, hidden_units):
        super().__init__()
        layers = []

        input_size = 8
        output_size = 1

        prev = input_size

        # Hidden layers
        for hidden in hidden_units:
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.ReLU())
            prev = hidden


        # Output layer
        layers.append(nn.Linear(hidden_units[-1], output_size))

        self.stack = nn.Sequential(*layers)

    def forward(self, x):
        return self.stack(x)
