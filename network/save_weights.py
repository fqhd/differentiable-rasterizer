import torch
import json
import numpy as np
from network import DynamicNet

def get_best_config():
    with open('output.json', 'r') as f:
        configs = json.load(f)

    def myFunc(c):
        return c['loss']

    configs.sort(key=myFunc)

    return configs[0]

config = get_best_config()

net = DynamicNet(hidden_units=config['layer_params'])
net.load_state_dict(torch.load('net.pth', weights_only=True))

idx = 1
for layer in net.stack.children():
	if isinstance(layer, torch.nn.Linear):
		weight = layer.weight.detach().numpy()
		bias = layer.bias.detach().numpy()
		np.save(f'weights/layer_{idx}_weight.npy', weight)
		np.save(f'weights/layer_{idx}_bias.npy', bias)
		idx += 1
