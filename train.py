import d4rl
import gym

import src.utils as ptu
from src.policy import TanhGaussianPolicy
from src.cql_trainer import CQLTrainer
from src.flatten_mlp import FlattenMlp
from src.d4rl_buffer import D4RLBuffer
from src.cql_algo import CQLAlgorithm
from src.evaluator import Evaluator
import argparse
import numpy as np
import wandb


def experiment(variant: dict):
    wandb_config = variant
    if variant["wandb"]:
        wandb.init(project="applying-cql", entity="rudyn", config=variant)
        wandb_config = wandb.config

    env = gym.make(wandb_config["task"])
    dataset = d4rl.qlearning_dataset(env)
    buffer = D4RLBuffer(dataset)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    M = wandb_config["layer_size"]
    qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M]
    )
    qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M]
    )
    target_qf1 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M]
    )
    target_qf2 = FlattenMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        hidden_sizes=[M, M, M]
    )
    policy = TanhGaussianPolicy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_sizes=[M, M, M]
    )

    trainer = CQLTrainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        max_q_backup=True if wandb_config['max_q_backup'] == 'True' else False,
        deterministic_backup=True if wandb_config['deterministic_backup'] == 'True' else False,
        min_q_weight=wandb_config['min_q_weight'],
        policy_lr=wandb_config['policy_lr'],
        qf_lr=wandb_config['qf_lr'],
        min_q_version=wandb_config['min_q_version'],
        reward_scale=wandb_config['reward_scale'],
        with_lagrange=wandb_config['lagrange_thresh'] > 0,
        lagrange_thresh=wandb_config['lagrange_thresh'],
        use_automatic_entropy_tuning=bool(wandb_config['use_automatic_entropy_tuning']),
        num_random=wandb_config['num_random'],
        num_qs=wandb_config['num_qs'],
    )
    evaluator = Evaluator(env)
    # replace this
    algorithm = CQLAlgorithm(
        trainer=trainer,
        evaluator=evaluator,
        replay_buffer=buffer,
        num_epochs=wandb_config['epochs'],
        batch_size=wandb_config['batch_size'],
        num_trains_per_epoch=wandb_config['num_trains_per_epoch'],
        num_eval_episodes=wandb_config['num_eval_episodes'],
        progress_bar=wandb_config['progress_bar'],
        log_wandb=wandb_config['wandb']
    )

    print('TRAINING')
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, type=str, help="Name of the task [pendulum].")
    parser.add_argument("--checkpoint_path", required=False, default="", type=str, help="Checkpoint path.")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--layer_size", default=128, type=int, help="Layer size")
    parser.add_argument("--epochs", default=1000, type=int, help="Epochs.")
    parser.add_argument("--num_trains_per_epoch", default=10, type=int, help="Num batch updates per epoch.")
    parser.add_argument("--num_eval_episodes", default=10, type=int, help="Num batch updates per epoch.")
    parser.add_argument("--gpu", default='0', type=str)

    # TRAINER KWARGS
    # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--max_q_backup", type=str, default="False")
    # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--use_automatic_entropy_tuning", type=int, default=1)
    parser.add_argument("--deterministic_backup", type=str, default="True")
    parser.add_argument('--temp', default=1.0, type=float)
    # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--num_qs', default=2, type=float)
    parser.add_argument('--min_q_weight', default=1.0, type=float)
    parser.add_argument('--policy_lr', default=1e-3, type=float)  # Policy learning rate
    parser.add_argument('--qf_lr', default=1e-3, type=float)  # Policy learning rate
    parser.add_argument('--soft_target_tau', default=1e-2, type=float)  # soft target update
    parser.add_argument('--min_q_version', default=3, type=int)  # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho))
    parser.add_argument('--reward_scale', default=1.0, type=float)
    parser.add_argument('--num_random', default=10, type=int)

    # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--lagrange_thresh', default=5.0, type=float)
    parser.add_argument("--wandb", default=True, type=bool, help="Wheter to log in wandb or not")
    parser.add_argument("--progress-bar", action="store_true", help="Wheter to use progress bar or not")
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()
    rnd = np.random.randint(low=0, high=1000000)
    ptu.set_gpu_mode(True)

    experiment(vars(args))


