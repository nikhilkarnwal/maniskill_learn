# from hashlib import new
# from botocore import vendored
# import gym
# from gym.spaces import space
# import mani_skill.env
# from numpy.core.fromnumeric import shape


# env = gym.make('OpenCabinetDrawer-v0')
# # full environment list can be found in available_environments.txt

# env.set_env_mode(obs_mode='state', reward_type='sparse')
# # obs_mode can be 'state', 'pointcloud' or 'rgbd'
# # reward_type can be 'sparse' or 'dense'
# print(env.observation_space) # this shows the observation structure in Openai Gym's format
# print(env.action_space) # this shows the action space in Openai Gym's format
# obs = env.reset()
# # print(obs['pointcloud']['seg'].shape)
# # for level_idx in range(0, 5): # level_idx is a random seed
# #     obs = env.reset(level=level_idx)
# #     print('#### Level {:d}'.format(level_idx))
# #     for i_step in range(100000):
# #         env.render('human') # a display is required to use this function; note that rendering will slow down the running speed
# #         action = env.action_space.sample()
# #         obs, reward, done, info = env.step(action) # take a random action
# #         print('{:d}: reward {:.4f}, done {}'.format(i_step, reward, done))
# #         if done:
# #             break
# # env.close()

# import gym
# import torch as th
# from torch import nn

# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
# from imitation.util import util


# class CustomCombinedExtractor(BaseFeaturesExtractor):
#     def __init__(self, observation_space: gym.spaces.Dict):
#         # We do not know features-dim here before going over all the items,
#         # so put something dummy for now. PyTorch requires calling
#         # nn.Module.__init__ before adding modules
#         super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

#         extractors = {}

#         total_concat_size = 0
#         # We need to know size of the output of this extractor,
#         # so go over all the spaces and compute output feature sizes
#         for key, subspace in observation_space.spaces.items():
#             if key == "image":
#                 # We will just downsample one channel of the image by 4x4 and flatten.
#                 # Assume the image is single-channel (subspace.shape[0] == 0)
#                 extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
#                 total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
#             elif key == "agent":
#                 # Run through a simple MLP
#                 extractors[key] = nn.Linear(subspace.shape[0], 16)
#                 total_concat_size += 16

#         self.extractors = nn.ModuleDict(extractors)

#         # Update the features dim manually
#         self._features_dim = total_concat_size

#     def forward(self, observations) -> th.Tensor:
#         encoded_tensor_list = []

#         # self.extractors contain nn.Modules that do all the processing.
#         for key, extractor in self.extractors.items():
#             encoded_tensor_list.append(extractor(observations[key]))
#         # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
#         return th.cat(encoded_tensor_list, dim=1)

# from stable_baselines3 import PPO

# policy_kwargs = dict(
#     features_extractor_class=CustomCombinedExtractor,
#     # features_extractor_kwargs=dict(features_dim=128),
# )

# def set_mode(env, i):
#     env = gym.make('OpenCabinetDrawer-v0')
#     env.set_env_mode(obs_mode='state', reward_type='sparse')
#     print('Called')
#     # print(env.observation_space)
#     return env

# # from stable_baselines3.common.vec_env import vec_extract_dict_obs, vec_monitor
# # vec_extract_dict_obs.VecExtractDictObs



# import numpy as np

# class NewEnv(gym.ObservationWrapper):
#     def __init__(self, env: gym.Env, i) -> None:
#         super().__init__(env)
#         # new_obs_space = self.observation_space['pointcloud']
#         # new_obs_space['agent']=gym.spaces.Box(-np.inf,np.inf,shape=self.observation_space['agent'],dtype=np.float32)
#         if isinstance(self.observation_space,dict):
#             self.observation_space = gym.spaces.Dict(self.flatten_dict(self.observation_space))
#         else:
#             self.observation_space = gym.spaces.Dict({'agent':self.observation_space})
#         print(self.observation_space)

#     def flatten_dict(self, obs):
#         new_obs = {}
#         for key in obs.keys():
#             if isinstance(obs[key],dict):
#                 new_obs.update(obs[key])
#         if 'agent' in obs.keys():
#             new_obs['agent']=gym.spaces.Box(-np.inf,np.inf,shape=obs['agent'],dtype=np.float32)
#         return new_obs

#     def observation(self, obs):
#         if isinstance(obs, dict):
#             new_obs = {}
#             for key in obs.keys():
#                 if isinstance(obs[key],dict):
#                     new_obs.update(obs[key])    
#             if 'agent' in obs.keys():
#                 new_obs['agent']=obs['agent']
#             obs = new_obs
#         else:
#             obs = {'agent':obs}
#         return obs 

# def func1(temp:gym.spaces.Space):
#     print(temp.spaces)
#     return temp
# temp = func1(env.observation_space)
# print(type(env.observation_space))
# venv = util.make_vec_env('OpenCabinetDrawer-v0', n_envs=1, parallel=False, post_wrappers=[set_mode, NewEnv])
# # venv = vec_extract_dict_obs.VecExtractDictObs(venv)
# venv = gym.make('OpenCabinetDrawer-v0')
# venv = set_mode(venv,0)
# venv = NewEnv(venv,0)
# model = PPO('MultiInputPolicy', env = venv, policy_kwargs=policy_kwargs, verbose=True)
# print(model.policy)
# model.learn(100)

from h5py import File

from mani_skill_learn.utils.fileio.h5_utils import load_h5_as_dict_array
f = File('./full_mani_skill_data/OpenCabinetDrawer/OpenCabinetDrawer_1056_link_0-v0.h5', 'r')
# f is a h5py.Group with keys traj_0 ... traj_n
print(f['traj_0'].keys())
print(load_h5_as_dict_array(f['traj_0']))

