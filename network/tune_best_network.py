import json
from train import train
import torch

def get_best_config():
    with open('output.json', 'r') as f:
        configs = json.load(f)

    def myFunc(c):
        return c['loss']

    configs.sort(key=myFunc)

    return configs[0]

config = get_best_config()
config['batch_size'] = 16384

loss, losses, net = train(config, iterations=1_000_000)

print(f'Saving network with loss: {loss}...')

torch.save(net.state_dict(), 'net.pth')

print('Network saved successfully!')
