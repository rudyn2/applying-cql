import gym
import numpy as np
from src.policy import TanhGaussianPolicy
import src.utils as ptu
import torch


class Evaluator:

    def __init__(self, task: str, max_steps: int = 1000):
        if task == "pendulum":
            self.env = gym.make("Pendulum-v1")
        else:
            raise ValueError("Cant evaluate in that task")

        self.max_steps = max_steps

    def evaluate(self, policy: TanhGaussianPolicy, n_episodes: int = 100, render: bool = False):
        rewards = []
        for _ in range(n_episodes):
            obs = self.env.reset()
            ep_rewards = []
            for _ in range(self.max_steps):
                if render:
                    self.env.render()
                with torch.no_grad():
                    action = policy(ptu.tensor(obs), deterministic=True)[0] * 2
                    action = action.cpu().numpy().tolist()
                obs, rew, done, info = self.env.step(action)
                obs = obs.squeeze()
                ep_rewards.append(rew)
                if done:
                    break
            rewards.append(np.sum(ep_rewards))
        return rewards


if __name__ == "__main__":
    random_policy = TanhGaussianPolicy(hidden_sizes=(16, 16), obs_dim=3, action_dim=1)
    evaluator = Evaluator(policy=random_policy, task="pendulum")
    evaluator.evaluate()
