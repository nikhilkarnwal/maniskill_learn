'''
IRL algorithms
'''

import abc
import torch
from torch._C import device
import torch.nn.functional as F
from mani_skill_learn.env.env_utils import build_env
from mani_skill_learn.env.wrappers import build_wrapper
from mani_skill_learn.methods.builder import BRL

from mani_skill_learn.networks import build_model
from mani_skill_learn.optimizers import build_optimizer
from mani_skill_learn.utils.data import to_torch
from mani_skill_learn.utils.torch import BaseAgent
from mani_skill_learn.networks.builder import build_backbone, build_dense_head
from mani_skill_learn.networks.utils import replace_placeholder_with_args, get_kwargs_from_shape

from stable_baselines3 import PPO, ppo
from imitation.algorithms.adversarial import gail, airl
from imitation.util import util

class Adversarial(BaseAgent):
    def __init__(self,nn_cfg, obs_shape, action_shape, action_space, batch_size=128):
        super().__init__()

        self.batch_size = batch_size

        policy_optim_cfg = nn_cfg.pop("optim_cfg")

        # nn_cfg['obs_shape'] = obs_shape
        # nn_cfg['action_shape'] = action_shape
        # nn_cfg['action_space'] = action_space

        replaceable_kwargs = get_kwargs_from_shape(obs_shape, action_shape)
        nn_cfg = replace_placeholder_with_args(nn_cfg, **replaceable_kwargs)
        self.backbone = build_backbone(nn_cfg)

    @abc.abstractmethod
    def update_parameters():
        raise NotImplementedError("")


# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from imitation.util import util
# import torch as th
# import torch.nn as nn


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space, backbone, features_dim):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=features_dim)

#         extractors = {}

#         self.backbone = backbone

#         # Update the features dim manually
#         self._features_dim = features_dim

#     def forward(self, observations) -> th.Tensor:
        


#         return th.cat(encoded_tensor_list, dim=1)

@BRL.register_module()
class GAILSB(Adversarial):

    def __init__(self, nn_cfg, obs_shape, action_shape, action_space,env_cfg, batch_size=128):
        self.gail_config = nn_cfg.pop('gail_cfg')

        super().__init__(nn_cfg, obs_shape, action_shape, action_space, batch_size=batch_size)

        env = build_env(env_cfg)
        env.device = 'cuda'
        env.set_backbone(self.backbone)
        gen_algo = PPO("MlpPolicy", env = env, verbose=True, device='cuda')
        self.model = gail.GAIL(demonstrations=None, demo_batch_size=self.gail_config['demo_batch_size'],venv=gen_algo.get_env(), gen_algo=gen_algo)

    def update_demo(self,demo):
        sampled_batch = demo.sample(self.batch_size)
        
        sampled_batch = dict(obs=sampled_batch['obs'], actions=sampled_batch["actions"])
        sampled_batch = to_torch(sampled_batch, device=self.device, dtype='float32')
        for key in sampled_batch:
            if not isinstance(sampled_batch[key], dict) and sampled_batch[key].ndim == 1:
                sampled_batch[key] = sampled_batch[key][..., None]
        final_obs = self.backbone(sampled_batch['obs'])
        # print(sampled_batch)
        # print(final_obs)
        self.model.set_demonstrations([{'obs':final_obs.to('cpu'), 'acts':sampled_batch["actions"].to('cpu')}])

    def update_parameters(self,re,**kvargs):
        self.model.train(total_timesteps=self.gail_config['total_timesteps'])
        return {
            'policy_abs_error': 1,
            'policy_loss': 1
        }
