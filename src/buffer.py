import random
from src.dataset import D3RLPYDataset, D4RLDataset
from collections import OrderedDict
import numpy as np
from typing import Union


class OfflineReplayBuffer(object):
    def __init__(self, offline_dataset: Union[D3RLPYDataset, D4RLDataset]):
        self._dataset = offline_dataset

    def random_batch(self, batch_size: int) -> dict:
        idxs = random.sample(list(range(len(self._dataset))), k=batch_size)
        batch_data = [self._dataset[i] for i in idxs]
        unpacked_batch_data = list(zip(*batch_data))
        return dict(
            observations=np.array(unpacked_batch_data[0], dtype=np.float32),
            rewards=np.array(unpacked_batch_data[1], dtype=np.float32),
            actions=np.array(unpacked_batch_data[2], dtype=np.float32),
            next_observations=np.array(unpacked_batch_data[3], dtype=np.float32),
            terminals=np.array(unpacked_batch_data[4], dtype=np.bool8)
        )

    def num_steps_can_sample(self):
        return len(self._dataset)

    def get_diagnostics(self):
        return OrderedDict([
            ('size', len(self._dataset))
        ])

    def __len__(self):
        return len(self._dataset)


if __name__ == "__main__":
    try:
        cartpole = D3RLPYDataset(data_path="../dataset/d3rlpy/cartpole.h5")
        cartpole_buffer = OfflineReplayBuffer(offline_dataset=cartpole)
        cartpole_buffer.random_batch(128)
        print("CARTPOLE buffer loaded successfully")
    except Exception as e:
        print("When loading cartpole dataset occurred: ", e)

    try:
        pendulum = D3RLPYDataset(data_path="../dataset/d3rlpy/pendulum.h5")
        pendulum_buffer = OfflineReplayBuffer(offline_dataset=pendulum)
        pendulum_buffer.random_batch(128)
        print("PENDULUM buffer loaded successfully")
    except Exception as e:
        print("When loading pendulum dataset occurred: ", e)

    try:
        hopper = D4RLDataset(data_path="../dataset/d4rl/hopper_medium.hdf5")
        hopper_buffer = OfflineReplayBuffer(offline_dataset=hopper)
        hopper_buffer.random_batch(128)
        print("HOPPER buffer loaded successfully")
    except Exception as e:
        print("When loading hopper dataset occurred: ", e)
