import d4rl
import gym
import numpy as np
import random


class D4RLBuffer(object):
    def __init__(self, qlearning_dataset: dict):
        self.dataset = qlearning_dataset
        self.available_indexes = list(range(len(self)))

    def random_batch(self, batch_size: int) -> dict:
        idxs = random.sample(self.available_indexes, k=batch_size)
        observations = [self.dataset['observations'][i] for i in idxs]
        actions = [self.dataset['actions'][i] for i in idxs]
        next_observations = [self.dataset['next_observations'][i] for i in idxs]
        rewards = [self.dataset['rewards'][i] for i in idxs]
        terminals = [self.dataset['terminals'][i] for i in idxs]
        return dict(
            observations=np.array(observations, dtype=np.float64),
            rewards=np.array(rewards, dtype=np.float64),
            actions=np.array(actions, dtype=np.float64),
            next_observations=np.array(next_observations, dtype=np.float64),
            terminals=np.array(terminals, dtype=np.bool8)
        )

    def __len__(self):
        return len(self.dataset['observations'])


if __name__ == "__main__":
    env = gym.make("hopper-medium-v0")
    dataset = d4rl.qlearning_dataset(env)
    buffer = D4RLBuffer(dataset)
    batch = buffer.random_batch(256)
    print()
