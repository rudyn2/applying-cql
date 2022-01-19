from termcolor import colored
import numpy as np
from d3rlpy.dataset import MDPDataset
import h5py


class D3RLPYDataset(object):

    def __init__(self, data_path: str):
        self._data_path = data_path
        self._mdp_dataset = MDPDataset.load(data_path)
        self._transitions = None
        self._load_transitions()

    def _load_transitions(self):
        transitions = []        # list of tuples (s, r, a, s', done)
        for episode in self._mdp_dataset.episodes:
            transition = episode[0]
            while transition.next_transition:
                obs = transition.observation
                act = transition.action
                rew = transition.reward
                next_obs = transition.next_observation
                done = transition.terminal

                transition = transition.next_transition
                transitions.append((obs, rew, act, next_obs, done))

        print(colored(f"# transitions: {len(transitions)}", "green"))
        self._transitions = transitions

    def __getitem__(self, item: int):
        return self._transitions[item]

    def __len__(self):
        return len(self._transitions)


class D4RLDataset(object):

    def __init__(self, data_path: str):
        self._data_path = data_path
        self._transitions = []
        self._load_dataset()
        print(colored(f"# transitions: {len(self._transitions)}", "green"))

    def _load_dataset(self):
        with h5py.File(self._data_path, "r") as f:
            observations = np.array(f["observations"])
            actions = np.array(f["actions"])
            rewards = np.array(f["rewards"])
            terminals = np.array(f["terminals"])
            timeouts = np.array(f["timeouts"])

        N = rewards.shape[0]
        terminate_on_end = False
        episode_step = 0
        for i in range(N - 1):
            obs = observations[i].astype(np.float32)
            new_obs = observations[i + 1].astype(np.float32)
            action = actions[i].astype(np.float32)
            reward = rewards[i].astype(np.float32)
            done_bool = bool(terminals[i])
            final_timestep = timeouts[i]
            if (not terminate_on_end) and final_timestep:
                # Skip this transition and don't apply terminals on the last step of an episode
                episode_step = 0
                continue
            if done_bool or final_timestep:
                episode_step = 0

            self._transitions.append((obs, reward, action, new_obs, done_bool))
            episode_step += 1

    def __getitem__(self, item: int):
        return self._transitions[item]

    def __len__(self):
        return len(self._transitions)


if __name__ == "__main__":
    try:
        cartpole = D3RLPYDataset(data_path="../dataset/d3rlpy/cartpole.h5")
        print("CARTPOLE dataset loaded successfully")
    except Exception as e:
        print("When loading cartpole dataset occurred: ", e)

    try:
        pendulum = D3RLPYDataset(data_path="../dataset/d3rlpy/pendulum.h5")
        print("PENDULUM dataset loaded successfully")
    except Exception as e:
        print("When loading pendulum dataset occurred: ", e)

    try:
        hopper = D4RLDataset(data_path="../dataset/d4rl/hopper_medium.hdf5")
        print("HOPPER dataset loaded successfully")
    except Exception as e:
        print("When loading hopper dataset occurred: ", e)


