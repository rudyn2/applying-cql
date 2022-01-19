from src.cql_trainer import CQLTrainer
from src.buffer import OfflineReplayBuffer
from src.evaluator import Evaluator
from tqdm import tqdm
import numpy as np
import wandb


class CQLAlgorithm(object):

    def __init__(self,
                 trainer: CQLTrainer,
                 evaluator: Evaluator,
                 replay_buffer: OfflineReplayBuffer,
                 num_epochs: int,
                 batch_size: int,
                 num_trains_per_epoch: int,
                 num_eval_episodes: int,
                 progress_bar: bool = True,
                 log_wandb: bool = False,
                 ):

        self.trainer = trainer
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self.num_epochs = num_epochs
        self.num_trains_per_epoch = num_trains_per_epoch
        self.num_eval_per_epoch = num_eval_episodes
        self.progress_bar = progress_bar
        self.log_wandb = log_wandb

    def train(self):
        for epoch in range(self.num_epochs):

            # TRAINING
            self.training_mode(True)
            loop_iter = range(self.num_trains_per_epoch)
            loop_iter = tqdm(loop_iter, "Training") if self.progress_bar else loop_iter
            for _ in loop_iter:
                train_data = self.replay_buffer.random_batch(self.batch_size)
                self.trainer.train(train_data)
            self.training_mode(False)

            # EVALUATION
            rewards = self.evaluator.evaluate(self.trainer.get_policy(), self.num_eval_per_epoch)
            additional_metrics = dict(mean_return=np.mean(rewards),
                                      max_return=np.max(rewards),
                                      min_return=np.min(rewards))
            print(f"epoch: {epoch}, average return: {np.mean(rewards)}")

            self.trainer.end_epoch(epoch)
            if self.log_wandb:
                self._log_stats_wandb(epoch, **additional_metrics)

    def _log_stats_wandb(self, epoch: int, **kwargs):
        """
        Log stats in wandb.
        """
        stats = self.trainer.get_diagnostics()
        train_prefix = "train/"
        qf1_prefix = "qf1/"
        qf2_prefix = "qf2/"
        policy_prefix = "policy/"
        to_log = {
            "epoch": epoch,
            "num_q_updates": stats["Num Q Updates"],
            "num_policy_updates": stats["Num Policy Updates"],
            "rewards/mean": stats["rewards Mean"],
            "rewards/max": stats["rewards Max"],
            "rewards/min": stats["rewards Min"],
            "actions/mean": stats["actions Mean"],
            "actions/min": stats["actions Min"],
            "actions/max": stats["actions Max"],

            # qf1 values
            train_prefix + qf1_prefix + "loss": stats["QF1 Loss"],
            train_prefix + qf1_prefix + "min_loss": stats["min QF1 Loss"],
            train_prefix + qf1_prefix + "std_values": stats["Std QF1 values"],
            train_prefix + qf1_prefix + "in_distribution/mean": stats["QF1 in-distribution values Mean"],
            train_prefix + qf1_prefix + "random/mean": stats["QF1 random values Mean"],
            train_prefix + qf1_prefix + "next_actions/mean": stats["QF1 next_actions values Mean"],
            train_prefix + qf1_prefix + "predictions/mean": stats["Q1 Predictions Mean"],

            # qf2 values
            train_prefix + qf2_prefix + "loss": stats["QF2 Loss"],
            train_prefix + qf2_prefix + "min_loss": stats["min QF2 Loss"],
            train_prefix + qf2_prefix + "std_values": stats["Std QF2 values"],
            train_prefix + qf2_prefix + "in_distribution/mean": stats["QF2 in-distribution values Mean"],
            train_prefix + qf2_prefix + "random/mean": stats["QF2 random values Mean"],
            train_prefix + qf2_prefix + "next_actions/mean": stats["QF2 next_actions values Mean"],
            train_prefix + qf2_prefix + "predictions/mean": stats["Q2 Predictions Mean"],

            # policy values
            train_prefix + policy_prefix + "loss": stats["Policy Loss"],
            train_prefix + policy_prefix + "mu/mean": stats["Policy mu Mean"],
            train_prefix + policy_prefix + "log_std/mean": stats["Policy log std Mean"],
            train_prefix + policy_prefix + "log_pis/mean": stats["Log Pis Mean"],
        }

        if "Alpha" in stats.keys():
            to_log[train_prefix + "alpha"] = stats["Alpha"]
            to_log[train_prefix + "alpha_loss"] = stats["Alpha Loss"]
            to_log[train_prefix + "target_entropy"] = stats["Target Entropy"]

        to_log.update(**kwargs)
        wandb.log(to_log)

    def to(self, device):
        for net in self.trainer.networks:
            net.to(device)

    def training_mode(self, mode):
        for net in self.trainer.networks:
            net.train(mode)
