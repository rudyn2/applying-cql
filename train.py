import src.utils as ptu
from src.policy import TanhGaussianPolicy
from src.cql_trainer import CQLTrainer
from src.flatten_mlp import FlattenMlp
from src.buffer import OfflineReplayBuffer
from src.dataset import D3RLPYDataset
from src.cql_algo import CQLAlgorithm
import argparse
import numpy as np
import wandb
from pprint import pprint


def experiment(variant: dict):
    wandb_config = variant
    if variant["algorithm_kwargs"]["log_wandb"]:
        wandb.init(project="applying-cql", entity="rudyn", config=variant)
        wandb_config = wandb.config

    if variant["task"] == "pendulum":
        obs_dim = 3
        action_dim = 1
    elif variant["task"] == "hopper":
        obs_dim = 11
        action_dim = 3
    else:
        raise ValueError

    pprint(wandb_config)
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
    offline_dataset = D3RLPYDataset(data_path=wandb_config["offline_buffer"])
    replay_buffer = OfflineReplayBuffer(offline_dataset=offline_dataset)
    trainer = CQLTrainer(
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        **wandb_config['trainer_kwargs']
    )
    # replace this
    algorithm = CQLAlgorithm(
        trainer=trainer,
        replay_buffer=replay_buffer,
        **wandb_config['algorithm_kwargs']
    )

    print('TRAINING')
    algorithm.to(ptu.device)
    algorithm.train()


if __name__ == "__main__":
    # noinspection PyTypeChecker
    variant = dict(
        algorithm="CQL",
        version="normal",
        algorithm_kwargs=dict(
            num_epochs=500,
            num_trains_per_train_loop=5,               # n batchs per epoch
            min_num_steps_before_training=0,
            batch_size=4,
        ),
        trainer_kwargs=dict(
            discount=0.99,
            soft_target_tau=5e-3,
            policy_lr=1E-4,
            qf_lr=3E-4,
            reward_scale=1,
            use_automatic_entropy_tuning=True,

            # Target nets/ policy vs Q-function update
            policy_eval_start=0,
            num_qs=2,

            # min Q
            temp=0.1,
            min_q_version=3,
            min_q_weight=1.0,

            # lagrange
            with_lagrange=True,
            lagrange_thresh=10.0,

            # extra params
            num_random=10,
            max_q_backup=False,
            deterministic_backup=False,
        ),
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=str, help="Data path.")
    parser.add_argument("--task", required=True, type=str, help="Name of the task [pendulum].")
    parser.add_argument("--checkpoint_path", required=False, default="", type=str, help="Checkpoint path.")

    parser.add_argument("--batch_size", default=4, type=int, help="Batch size")
    parser.add_argument("--layer_size", default=128, type=int, help="Layer size")
    parser.add_argument("--epochs", default=1000, type=int, help="Epochs.")
    parser.add_argument("--num_trains_per_train_loop", default=2, type=int, help="Num batch updates per epoch.")
    parser.add_argument("--gpu", default='0', type=str)
    # if we want to try max_{a'} backups, set this to true
    parser.add_argument("--max_q_backup", type=str, default="False")
    # defaults to true, it does not backup entropy in the Q-function, as per Equation 3
    parser.add_argument("--use_automatic_entropy_tuning", type=int, default=1)
    parser.add_argument("--deterministic_backup", type=str, default="True")
    # Defaulted to 20000 (40000 or 10000 work similarly)
    parser.add_argument("--policy_eval_start", default=0, type=int)
    # the value of alpha, set to 5.0 or 10.0 if not using lagrange
    parser.add_argument('--min_q_weight', default=1.0, type=float)
    parser.add_argument('--policy_lr', default=1e-4, type=float)  # Policy learning rate
    parser.add_argument('--qf_lr', default=3e-4, type=float)  # Policy learning rate
    parser.add_argument('--min_q_version', default=3, type=int)  # min_q_version = 3 (CQL(H)), version = 2 (CQL(rho))
    parser.add_argument('--reward_scale', default=1.0, type=float)
    parser.add_argument('--data_percentage', default=1.0, type=float)

    # the value of tau, corresponds to the CQL(lagrange) version
    parser.add_argument('--lagrange_thresh', default=5.0, type=float)
    parser.add_argument("--wandb", default=True, type=bool, help="Wheter to log in wandb or not")

    parser.add_argument("--progress-bar", action="store_true", help="Wheter to use progress bar or not")
    parser.add_argument('--seed', default=10, type=int)

    args = parser.parse_args()

    # TRAINER KWARGS
    variant['trainer_kwargs']['max_q_backup'] = (True if args.max_q_backup == 'True' else False)
    variant['trainer_kwargs']['deterministic_backup'] = (True if args.deterministic_backup == 'True' else False)
    variant['trainer_kwargs']['min_q_weight'] = args.min_q_weight
    variant['trainer_kwargs']['policy_lr'] = args.policy_lr
    variant['trainer_kwargs']['qf_lr'] = args.policy_lr
    variant['trainer_kwargs']['min_q_version'] = args.min_q_version
    variant['trainer_kwargs']['reward_scale'] = args.reward_scale
    variant['trainer_kwargs']['policy_eval_start'] = args.policy_eval_start
    variant['trainer_kwargs']['lagrange_thresh'] = args.lagrange_thresh
    variant['trainer_kwargs']['use_automatic_entropy_tuning'] = bool(args.use_automatic_entropy_tuning)
    if args.lagrange_thresh <= 0.0:
        variant['trainer_kwargs']['with_lagrange'] = False

    # ALGORITHM KWARGS
    variant["algorithm_kwargs"]["progress_bar"] = args.progress_bar
    variant["algorithm_kwargs"]["log_wandb"] = args.wandb
    variant["algorithm_kwargs"]["batch_size"] = args.batch_size
    variant["algorithm_kwargs"]["num_epochs"] = args.epochs
    variant["algorithm_kwargs"]["num_trains_per_train_loop"] = args.num_trains_per_train_loop
    if args.checkpoint_path != "":
        variant["algorithm_kwargs"]["checkpoint_metric"] = "dataset_q1_values"
        variant["algorithm_kwargs"]["save_checkpoint"] = True
        variant["algorithm_kwargs"]["checkpoint_path"] = args.checkpoint_path

    # GENERAL ARGS
    variant["offline_buffer"] = args.data
    variant["task"] = args.task
    variant['seed'] = args.seed
    variant['layer_size'] = args.layer_size
    variant['data_percentage'] = args.data_percentage

    rnd = np.random.randint(low=0, high=1000000)
    ptu.set_gpu_mode(True)

    experiment(variant)


