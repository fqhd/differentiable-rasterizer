import numpy as np
import random

data = np.fromfile('training_data', dtype='<f4')
data = data.reshape(-1, 9)

def get_batch(batch_size):
    batch = np.empty(shape=[batch_size, 9], dtype=np.float32)
    for i in range(batch_size):
        batch[i] = random.choice(data)
    return batch
