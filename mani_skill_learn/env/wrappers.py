import gym
from gym.core import Wrapper, ObservationWrapper
from mani_skill_learn.utils.data.converter import to_np, to_torch
from mani_skill_learn.utils.meta import Registry, build_from_cfg
from mani_skill_learn.utils.data import dict_to_seq, concat_dict_of_list_array
from .observation_process import process_mani_skill_base
from collections import deque
import numpy as np


WRAPPERS = Registry('wrapper of env')


@WRAPPERS.register_module()
class MujocoWrapper(Wrapper):
    def get_state(self):
        if hasattr(self.env, 'goal'):
            return np.concatenate([self.env.sim.get_state().flatten(), self.env.goal], axis=-1)
        else:
            return self.env.sim.get_state().flatten()

    def set_state(self, state):
        if hasattr(self.env, 'goal'):
            sim_state_len = self.env.sim.get_state().flatten().shape[0]
            self.env.sim.set_state(self.env.sim.get_state().from_flattened(
                state[:sim_state_len], self.env.sim))
            self.env.goal = state[sim_state_len:]
        else:
            self.env.sim.set_state(
                self.env.sim.get_state().from_flattened(state, self.env.sim))

    def get_obs(self):
        return self.env._get_obs()

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps


@WRAPPERS.register_module()
class PendulumWrapper(Wrapper):
    def get_state(self):
        return np.array(self.env.state)

    def set_state(self, state):
        self.env.state = state

    def get_obs(self):
        return self.env._get_obs()

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps


@WRAPPERS.register_module()
class SapienRLWrapper(ObservationWrapper):
    def __init__(self, env, stack_frame=1):
        """
        Stack k last frames for point clouds or rgbd and remap the rendering configs
        """
        super(SapienRLWrapper, self).__init__(env)
        self.stack_frame = stack_frame
        self.buffered_data = {}

    def get_state(self):
        return self.env.get_state(True)

    def _update_buffer(self, obs):
        for key in obs:
            if key not in self.buffered_data:
                self.buffered_data[key] = deque(
                    [obs[key]] * self.stack_frame, maxlen=self.stack_frame)
            else:
                self.buffered_data[key].append(obs[key])

    def _get_buffer_content(self):
        axis = 0 if self.obs_mode == 'pointcloud' else -1
        return {key: np.concatenate(self.buffered_data[key], axis=axis) for key in self.buffered_data}

    def observation(self, observation):
        if self.obs_mode == "state":
            return observation
        observation = process_mani_skill_base(observation, self.env)
        visual_data = observation[self.obs_mode]
        self._update_buffer(visual_data)
        visual_data = self._get_buffer_content()
        state = observation['agent']
        # Convert dict of array to list of array with sorted key
        ret = {}
        ret[self.obs_mode] = visual_data
        ret['state'] = state
        return ret

    def get_obs(self):
        return self.observation(self.env.get_obs())

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps

    def render(self, mode='human', *args, **kwargs):
        if mode == 'human':
            self.env.render(mode, *args, **kwargs)
            return

        if mode == 'rgb_array':
            img = self.env.render(mode='color_image', *args, **kwargs)
        else:
            img = self.env.render(mode=mode, *args, **kwargs)
        if isinstance(img, dict):
            if 'world' in img:
                img = img['world']
            elif 'main' in img:
                img = img['main']
            else:
                print(img.keys())
                exit(0)
        if isinstance(img, dict):
            img = img['rgb']
        if img.ndim == 4:
            assert img.shape[0] == 1
            img = img[0]
        if img.dtype in [np.float32, np.float64]:
            img = np.clip(img, a_min=0, a_max=1) * 255
        img = img[..., :3]
        img = img.astype(np.uint8)
        return img


@WRAPPERS.register_module()
class IRLWrapper(ObservationWrapper):
    def __init__(self, env) -> None:
        super().__init__(env)
        print('Wrappers')
        self.backbone = None
        self.device = None
        # if isinstance(self.observation_space, dict):
        #     self.observation_space = gym.spaces.Dict(
        #         self.flatten_dict(self.observation_space))
        # else:
        #     self.observation_space = gym.spaces.Dict(
        #         {'agent': self.observation_space})

    def set_backbone(self, backbone):
        self.backbone = backbone
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, shape=(256,), dtype=np.float32)

    def flatten_dict(self, obs):
        new_obs = {}
        for key in obs.keys():
            if isinstance(obs[key], dict):
                new_obs.update(obs[key])
        if 'agent' in obs.keys():
            new_obs['agent'] = gym.spaces.Box(-np.inf,
                                              np.inf, shape=obs['agent'], dtype=np.float32)
        return new_obs

    def observation(self, obs):
        # if isinstance(obs, dict):
        #     new_obs = {}
        #     for key in obs.keys():
        #         if isinstance(obs[key], dict):
        #             new_obs.update(obs[key])
        #     if 'agent' in obs.keys():
        #         new_obs['agent'] = obs['agent']
        #     obs = new_obs
        # else:
        #     obs = {'agent': obs}
        if self.backbone:
            self.backbone.to(self.device)
            # print(self.device)
            # print(next(self.backbone.parameters()).device)
            obs = to_torch(obs, device=self.device, dtype='float32')
            if 'pointcloud' in obs:
                curr_obs = obs['pointcloud']    
                for key in curr_obs:
                    if not isinstance(curr_obs[key], dict):
                        curr_obs[key] = curr_obs[key].unsqueeze(0)
            obs['state']=obs['state'].unsqueeze(0)
            # print(obs)
            obs = self.backbone(obs)[1][0]
            obs = to_np(obs)
            # print(obs)
        return obs

    def get_state(self):
        return self.env.get_state()

    def get_obs(self):
        return self.observation( self.env._get_obs())

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps


def build_wrapper(cfg, default_args=None):
    return build_from_cfg(cfg, WRAPPERS, default_args)
