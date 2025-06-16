import torch
from torch import nn
import torch.utils.data as dutils
import random
import json
import time
from network import DynamicNet
from data import get_batch
from train import train

def generate_config():
    num_layers = random.randint(2, 5)
    layer_params = []
    for _ in range(num_layers):
        layer_params.append(random.randint(16, 64))
    lower_bound = 1e-5
    upper_bound = 1e-2
    t = random.random()
    learning_rate = lower_bound * (1 - t) + t * upper_bound
    batch_size = random.randint(16, 512)

    return {
        'layer_params': layer_params,
        'learning_rate': learning_rate,
        'batch_size': batch_size
    }


configs = []
for _ in range(100):
    config = generate_config()
    loss, _, _ = train(config)
    config['loss'] = loss
    configs.append(config)

with open('output.json', 'r') as f:
    saved_confs = json.load(f)

with open('output.json', 'w') as f:
    json.dump(saved_confs + configs, f)
