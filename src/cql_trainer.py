import os
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn
from typing import Union

import src.utils as ptu
from src.utils import create_stats_ordered_dict


torch.autograd.set_detect_anomaly(True)


class CQLTrainer(object):
    def __init__(
            self,
            # networks
            policy,
            qf1,
            qf2,
            target_qf1,
            target_qf2,

            # hyper-parameters
            discount: float = 0.99,
            reward_scale: float = 1.0,
            policy_lr: float = 1e-3,
            qf_lr: float = 1e-3,
            soft_target_tau: float = 1e-2,
            use_automatic_entropy_tuning=True,
            target_entropy=None,
            policy_eval_start=0,
            num_qs: int = 2,
            action_dim: int = 2,
            optimizer_class=optim.Adam,

            # CQL
            min_q_version=3,
            temp=1.0,
            min_q_weight=1.0,

            # sort of backup
            max_q_backup=False,
            deterministic_backup=True,
            num_random=10,
            with_lagrange=False,
            lagrange_thresh=0.0
    ):
        super().__init__()

        self.policy = policy
        self.qf1 = qf1
        self.qf2 = qf2
        self.target_qf1 = target_qf1
        self.target_qf2 = target_qf2
        self.soft_target_tau = soft_target_tau

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        self.target_entropy = target_entropy
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                self.target_entropy = -action_dim
            self.log_alpha = torch.tensor([np.log(temp)], requires_grad=True, device=ptu.device)
            self.alpha_optimizer = optimizer_class([self.log_alpha], lr=policy_lr)

        self.with_lagrange = with_lagrange
        if self.with_lagrange:
            self.target_action_gap = lagrange_thresh
            self.log_alpha_prime = ptu.zeros(1, requires_grad=True)
            self.alpha_prime_optimizer = optimizer_class([self.log_alpha_prime], lr=qf_lr)

        self.qf_criterion = nn.MSELoss()
        self.vf_criterion = nn.MSELoss()

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qf1_optimizer = optimizer_class(
            self.qf1.parameters(),
            lr=qf_lr,
        )
        self.qf2_optimizer = optimizer_class(
            self.qf2.parameters(),
            lr=qf_lr,
        )

        self.policy_lr = policy_lr
        self.qf_lr = qf_lr
        self.discount = discount
        self.reward_scale = reward_scale
        self.eval_statistics = OrderedDict()
        self._n_train_steps_total = 0
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start

        self._num_train_steps = 0
        self._current_epoch = 0
        self._policy_update_ctr = 0
        self._num_q_update_steps = 0
        self._num_policy_update_steps = 0
        self._num_policy_steps = 1

        self.num_qs = num_qs

        # min Q
        self.temp = temp
        self.min_q_version = min_q_version
        self.min_q_weight = min_q_weight

        self.softmax = torch.nn.Softmax(dim=1)
        self.softplus = torch.nn.Softplus(beta=int(self.temp), threshold=20)

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.num_random = num_random

        ptu.soft_update_from_to(self.qf1, self.target_qf1, 1)
        if self.num_qs > 1:
            ptu.soft_update_from_to(self.qf2, self.target_qf2, 1)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def _get_tensor_values(self, obs: Union[tuple, list, torch.Tensor], actions, network=None):
        action_shape = actions.shape[0]
        if isinstance(obs, list) or isinstance(obs, tuple):
            obs_shape = len(obs)
        else:
            obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        preds = network(obs, actions, num_repeat=num_repeat)
        preds = preds.view(obs_shape, num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs, reparameterize=True, return_log_prob=True, num_repeat=num_actions,
        )
        return new_obs_actions, new_obs_log_pi.view(len(obs), num_actions, 1)

    def train(self, np_batch: dict):
        self._num_train_steps += 1
        torch_batch = {}
        for k, v in np_batch.items():
            if "obs" in k:
                torch_batch[k] = ptu.from_numpy(v).float()
            elif v.dtype == np.bool8:
                torch_batch[k] = ptu.from_numpy(v.astype(int))
            else:
                torch_batch[k] = ptu.from_numpy(v).float()
        self.train_from_torch(torch_batch)

    def train_from_torch(self, batch):
        self._current_epoch += 1
        rewards = batch['rewards']
        terminals = batch['terminals']
        obs = batch['observations']
        actions = batch['actions']
        next_obs = batch['next_observations']

        # region: ACTOR-CRITIC
        """
        Policy and Alpha Loss
        """
        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = self.temp

        if self.num_qs == 1:
            q_new_actions = self.qf1(obs, new_obs_actions)
        else:
            q_new_actions = torch.min(
                self.qf1(obs, new_obs_actions),
                self.qf2(obs, new_obs_actions)
            )

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        if self._current_epoch < self.policy_eval_start:
            """
            For the initial few epochs, try doing behavioral cloning, if needed
            conventionally, there's not much difference in performance with having 20k
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.log_prob(obs, actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()

        """
        QF Loss
        """
        q2_pred = None
        q1_pred = self.qf1(obs, actions)
        if self.num_qs > 1:
            q2_pred = self.qf2(obs, actions)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs, reparameterize=True, return_log_prob=True
        )
        new_curr_actions, _, _, new_curr_log_pi, *_ = self.policy(
            obs, reparameterize=True, return_log_prob=True
        )

        if not self.max_q_backup:
            if self.num_qs == 1:
                target_q_values = self.target_qf1(next_obs, new_next_actions)
            else:
                target_q_values = torch.min(
                    self.target_qf1(next_obs, new_next_actions),
                    self.target_qf2(next_obs, new_next_actions),
                )

            if not self.deterministic_backup:
                target_q_values = target_q_values - alpha * new_log_pi
        else:
            """when using max q backup"""
            next_actions_temp, _ = self._get_policy_actions(next_obs, num_actions=10, network=self.policy)
            target_qf1_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf1).max(1)[0].view(-1, 1)
            target_qf2_values = self._get_tensor_values(next_obs, next_actions_temp, network=self.target_qf2).max(1)[0].view(-1, 1)
            target_q_values = torch.min(target_qf1_values, target_qf2_values)

        q_target = self.reward_scale * rewards.unsqueeze(1) + \
                   (1. - terminals.unsqueeze(1)) * self.discount * target_q_values
        q_target = q_target.detach()

        qf2_loss = None
        qf1_loss = self.qf_criterion(q1_pred, q_target)
        if self.num_qs > 1:
            qf2_loss = self.qf_criterion(q2_pred, q_target)
        # endregion

        # region: CQL
        # sample actions from an uniform distribution
        random_actions_tensor = torch.FloatTensor(q2_pred.shape[0] * self.num_random, actions.shape[-1])
        random_actions_tensor = random_actions_tensor.uniform_(-1, 1).to(self._device)

        # get policy actions a_policy_t and a_policy_t1 from the same observation repeated num_actions times
        curr_actions_tensor, curr_log_pis = self._get_policy_actions(obs,
                                                                     num_actions=self.num_random,
                                                                     network=self.policy)
        new_curr_actions_tensor, new_log_pis = self._get_policy_actions(next_obs,
                                                                        num_actions=self.num_random,
                                                                        network=self.policy)

        # get q-values of (s, a_rand), (s, a_policy_t), (s, a_policy_t1) for both q-functions
        q1_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf1)
        q2_rand = self._get_tensor_values(obs, random_actions_tensor, network=self.qf2)
        q1_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf1)
        q2_curr_actions = self._get_tensor_values(obs, curr_actions_tensor, network=self.qf2)
        q1_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf1)
        q2_next_actions = self._get_tensor_values(obs, new_curr_actions_tensor, network=self.qf2)

        cat_q1 = torch.cat(
            [q1_rand, q1_pred.unsqueeze(1), q1_next_actions, q1_curr_actions], 1
        )
        cat_q2 = torch.cat(
            [q2_rand, q2_pred.unsqueeze(1), q2_next_actions, q2_curr_actions], 1
        )
        std_q1 = torch.std(cat_q1, dim=1)   # just calculated to evaluate stats
        std_q2 = torch.std(cat_q2, dim=1)   # just calculated to evaluate stats

        if self.min_q_version == 3:
            # importance sampled version
            random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            cat_q1 = torch.cat(
                [q1_rand - random_density, q1_next_actions - new_log_pis.detach(),
                 q1_curr_actions - curr_log_pis.detach()], 1
            )
            cat_q2 = torch.cat(
                [q2_rand - random_density, q2_next_actions - new_log_pis.detach(),
                 q2_curr_actions - curr_log_pis.detach()], 1
            )

        min_qf1_loss = torch.logsumexp(cat_q1 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp
        min_qf2_loss = torch.logsumexp(cat_q2 / self.temp, dim=1, ).mean() * self.min_q_weight * self.temp

        """Subtract the log likelihood of data"""
        min_qf1_loss = min_qf1_loss - q1_pred.mean() * self.min_q_weight
        min_qf2_loss = min_qf2_loss - q2_pred.mean() * self.min_q_weight

        alpha_prime, alpha_prime_loss = None, None
        if self.with_lagrange:
            alpha_prime = torch.clamp(self.log_alpha_prime.exp(), min=0.0, max=1000000.0)
            min_qf1_loss = alpha_prime * (min_qf1_loss - self.target_action_gap)
            min_qf2_loss = alpha_prime * (min_qf2_loss - self.target_action_gap)

            self.alpha_prime_optimizer.zero_grad()
            alpha_prime_loss = (-min_qf1_loss - min_qf2_loss) * 0.5
            alpha_prime_loss.backward(retain_graph=True)
            self.alpha_prime_optimizer.step()
        # endregion

        qf1_loss = qf1_loss + min_qf1_loss
        qf2_loss = qf2_loss + min_qf2_loss

        """
        Update networks
        """
        # Update the Q-functions iff
        self._num_q_update_steps += 1
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward(retain_graph=True)

        if self.num_qs > 1:
            self.qf2_optimizer.zero_grad()
            qf2_loss.backward(retain_graph=True)
            # self.qf2_optimizer.step()

        self._num_policy_update_steps += 1
        self.policy_optimizer.zero_grad()
        policy_loss.backward(retain_graph=False)
        # self.policy_optimizer.step()

        self.qf1_optimizer.step()
        if self.num_qs > 1:
            self.qf2_optimizer.step()
        self.policy_optimizer.step()

        """
        Soft Updates
        """
        ptu.soft_update_from_to(self.qf1, self.target_qf1, self.soft_target_tau)
        if self.num_qs > 1:
            ptu.soft_update_from_to(self.qf2, self.target_qf2, self.soft_target_tau)

        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False
            """
            Eval should set this to None.
            This way, these statistics are only computed for one batch.
            """
            policy_loss = (log_pi - q_new_actions).mean()

            self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(qf1_loss))
            self.eval_statistics['min QF1 Loss'] = np.mean(ptu.get_numpy(min_qf1_loss))
            if self.num_qs > 1:
                self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(qf2_loss))
                self.eval_statistics['min QF2 Loss'] = np.mean(ptu.get_numpy(min_qf2_loss))

            self.eval_statistics['Std QF1 values'] = np.mean(ptu.get_numpy(std_q1))
            self.eval_statistics['Std QF2 values'] = np.mean(ptu.get_numpy(std_q2))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 in-distribution values',
                ptu.get_numpy(q1_curr_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 in-distribution values',
                ptu.get_numpy(q2_curr_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 random values',
                ptu.get_numpy(q1_rand),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 random values',
                ptu.get_numpy(q2_rand),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF1 next_actions values',
                ptu.get_numpy(q1_next_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'QF2 next_actions values',
                ptu.get_numpy(q2_next_actions),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'actions',
                ptu.get_numpy(actions)
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'rewards',
                ptu.get_numpy(rewards)
            ))

            self.eval_statistics['Num Q Updates'] = self._num_q_update_steps
            self.eval_statistics['Num Policy Updates'] = self._num_policy_update_steps
            self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
                policy_loss
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q1 Predictions',
                ptu.get_numpy(q1_pred),
            ))
            if self.num_qs > 1:
                self.eval_statistics.update(create_stats_ordered_dict(
                    'Q2 Predictions',
                    ptu.get_numpy(q2_pred),
                ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Q Targets',
                ptu.get_numpy(q_target),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Log Pis',
                ptu.get_numpy(log_pi),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy mu',
                ptu.get_numpy(policy_mean),
            ))
            self.eval_statistics.update(create_stats_ordered_dict(
                'Policy log std',
                ptu.get_numpy(policy_log_std),
            ))

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()
                self.eval_statistics['Target Entropy'] = self.target_entropy

            if self.with_lagrange:
                self.eval_statistics['Alpha_prime'] = alpha_prime.item()
                self.eval_statistics['min_q1_loss'] = ptu.get_numpy(min_qf1_loss).mean()
                self.eval_statistics['min_q2_loss'] = ptu.get_numpy(min_qf2_loss).mean()
                self.eval_statistics['threshold action gap'] = self.target_action_gap
                self.eval_statistics['alpha prime loss'] = alpha_prime_loss.item()

        self._n_train_steps_total += 1

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qf1,
            self.qf2,
            self.target_qf1,
            self.target_qf2,
        ]
        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qf1=self.qf1,
            qf2=self.qf2,
            target_qf1=self.target_qf1,
            target_qf2=self.target_qf2,
        )

    def get_config(self) -> dict:
        return dict(
            temperature=self.temp,
            discount=self.discount,
            reward_scale=self.reward_scale,
            policy_lr=self.policy_lr,
            qf_lr=self.qf_lr,
            num_qs=self.num_qs,
            use_automatic_entropy_tunning=self.use_automatic_entropy_tuning,
            with_lagrange=self.with_lagrange,
            target_entropy=self.target_entropy,
            num_random=self.num_random,
            max_q_backup=self.max_q_backup,
            deterministic_backup=self.deterministic_backup,
            min_q_version=self.min_q_version,
            min_q_wight=self.min_q_weight,
            device=self._device
        )

    def save_checkpoint(self, global_path: str, tag: str = "", save_in_wandb: bool = False):

        policy_path = f"checkpoint_policy_{tag}.pth"
        target_qf1_path = f"checkpoint_target_qf1_{tag}.pth"
        target_qf2_path = f"checkpoint_target_qf2_{tag}.pth"

        # save checkpoints in local
        torch.save(self.policy.state_dict(), os.path.join(os.getcwd(), policy_path))
        torch.save(self.target_qf1.state_dict(), os.path.join(os.getcwd(), target_qf1_path))
        torch.save(self.target_qf2.state_dict(), os.path.join(os.getcwd(), target_qf2_path))

        if save_in_wandb:
            # save in wandb
            # wandb.save(os.path.join(wandb.run.dir, "checkpoint*"), base_path=wandb.run.dir)
            pass


if __name__ == "__main__":
    from src.models.policy import TanhGaussianPolicy
    from src.models.mlp import FlattenMlp

    obs_dim = 762       # expl_env.observation_space.low.size
    action_dim = 2      # eval_env.action_space.low.size
    M = 128

    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
        metadata_idx=27,
        max_n_points=2000
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
        metadata_idx=27,
        max_n_points=2000
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
        metadata_idx=27,
        max_n_points=2000
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M],
        metadata_idx=27,
        max_n_points=2000
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M],
        metadata_idx=27,
        max_n_points=2000
    )
    trainer = CQLTrainer(policy, qf1, qf2, target_qf1, target_qf2)
    print("The trainer was instantiated successfully!")
