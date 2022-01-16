from src.mlp import Mlp
import torch


class FlattenMlp(Mlp):
    """
    Encode observation, flatten along dimension 1, concatenate the action and then pass through MLP.
    """

    def __init__(self, *args, **kwargs):
        super(FlattenMlp, self).__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        # usually observations = inputs[0], actions = inputs[1]
        obs = inputs[0]
        act = inputs[1]
        if "num_repeat" in kwargs.keys():
            num_repeat = kwargs["num_repeat"]
            obs = obs.unsqueeze(1).repeat(1, num_repeat, 1).view(obs.shape[0] * num_repeat, obs.shape[1])
        flat_inputs = torch.cat([obs, act], dim=1)
        return super().forward(flat_inputs, **kwargs)
