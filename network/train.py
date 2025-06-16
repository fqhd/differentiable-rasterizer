import torch
from data import get_batch
from network import DynamicNet
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
print(device)

def train(config, iterations=1000):
    net = DynamicNet(hidden_units=config['layer_params'])

    batch_size = config['batch_size']

    net = net.to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=config['learning_rate'])

    losses = []

    for epoch in range(iterations):
        batch = torch.from_numpy(get_batch(batch_size))
        inputs = batch[:, :8]
        labels = batch[:, -1:]

        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())


    avg = sum(losses[-100:]) / 100

    return avg, losses, net
