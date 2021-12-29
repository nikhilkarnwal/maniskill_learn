from logging import info
from typing import ItemsView
from h5py import File
from argparse import Action
import glob
import os.path as osp
from random import shuffle

import h5py
from h5py._hl.selections2 import select_read
from imitation.algorithms.base import AnyTransitions
from imitation.data.types import Trajectory, TrajectoryWithRew
import numpy as np
from pynvml.nvml import nvmlShutdown

from mani_skill_learn.utils.data import (dict_to_seq, recursive_init_dict_array, map_func_to_dict_array,
                                         store_dict_array_to_h5,
                                         sample_element_in_dict_array, assign_single_element_in_dict_array, is_seq_of)
from mani_skill_learn.utils.data.converter import to_np, to_torch
from mani_skill_learn.utils.fileio import load_h5s_as_list_dict_array, load, check_md5sum
from mani_skill_learn.utils.fileio.h5_utils import load_h5_as_dict_array
from mani_skill_learn.utils.meta.v2_utils import add_absorbing_state, add_action_for_absorbing_states
from .builder import REPLAYS


@REPLAYS.register_module()
class ReplayMemory:
    """
    Replay buffer uses dict-array as basic data structure, which can be easily saved as hdf5 file.

    See mani_skill_learn/utils/data/dict_array.py for more details.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = {}
        self.position = 0
        self.running_count = 0

    def __getitem__(self, key):
        return self.memory[key]

    def __len__(self):
        return min(self.running_count, self.capacity)

    def reset(self):
        self.memory = {}
        self.position = 0
        self.running_count = 0

    def initialize(self, **kwargs):
        self.memory = recursive_init_dict_array(
            self.memory, dict(kwargs), self.capacity, self.position)

    def push(self, **kwargs):
        # assert not self.fixed, "Fix replay buffer does not support adding items!"
        self.initialize(**kwargs)
        assign_single_element_in_dict_array(
            self.memory, self.position, dict(kwargs))
        self.running_count += 1
        self.position = (self.position + 1) % self.capacity

    def push_batch(self, **kwargs):
        # assert not self.fixed, "Fix replay buffer does not support adding items!"
        kwargs = dict(kwargs)
        keys, values = dict_to_seq(kwargs)
        batch_size = len(
            list(filter(lambda v: not isinstance(v, dict), values))[0])
        for i in range(batch_size):
            self.push(**sample_element_in_dict_array(kwargs, i))

    def sample(self, batch_size):
        batch_idx = np.random.randint(low=0, high=len(self), size=batch_size)
        return sample_element_in_dict_array(self.memory, batch_idx)

    def tail_mean(self, num):
        def func(_, __, ___): return np.mean(_[___ - __:___])
        return map_func_to_dict_array(self.memory, func, num, len(self))

    def get_all(self):
        return sample_element_in_dict_array(self.memory, slice(0, len(self)))

    def to_h5(self, file, with_traj_index=False):
        from h5py import File
        data = self.get_all()
        if with_traj_index:
            data = {'traj_0': data}
        if isinstance(file, str):
            with File(file, 'w') as f:
                store_dict_array_to_h5(data, f)
        else:
            store_dict_array_to_h5(data, file)

    def restore(self, init_buffers, replicate_init_buffer=1, num_trajs_per_demo_file=-1):
        buffer_keys = ['obs', 'actions', 'next_obs', 'rewards', 'dones']
        if isinstance(init_buffers, str):
            init_buffers = [init_buffers]
        if is_seq_of(init_buffers, str):
            init_buffers = [load_h5s_as_list_dict_array(
                _) for _ in init_buffers]
        if isinstance(init_buffers, dict):
            init_buffers = [init_buffers]

        print('Num of datasets', len(init_buffers))
        for _ in range(replicate_init_buffer):
            cnt = 0
            for init_buffer in init_buffers:
                for item in init_buffer:
                    if cnt >= num_trajs_per_demo_file and num_trajs_per_demo_file != -1:
                        break
                    item = {key: item[key] for key in buffer_keys}
                    self.push_batch(**item)
                    cnt += 1
        print(
            f'Num of buffers {len(init_buffers)}, Total steps {self.running_count}')


@REPLAYS.register_module()
class ReplayDisk(ReplayMemory):
    """

    """

    def __init__(self, capacity, keys=None):
        super(ReplayDisk, self).__init__(capacity)
        self.keys = ['obs', 'actions', 'next_obs',
                     'rewards', 'dones'] if keys is None else keys
        self.h5_files = []
        self.h5_size = []
        self.h5_idx = 0
        self.idx_in_h5 = 0

        self.memory_begin_index = 0

    def restore(self, init_buffers, replicate_init_buffer=1, num_trajs_per_demo_file=-1):
        assert num_trajs_per_demo_file == - \
            1, "For chunked dataset, we only support loading all trajectories"
        assert replicate_init_buffer == 1, "Disk replay does not need to be replicated."
        if not (isinstance(init_buffers, str) and osp.exists(init_buffers) and osp.isdir(init_buffers)):
            print(f'{init_buffers} does not exist or is not a folder!')
            exit(-1)
        else:
            if not osp.exists(osp.join(init_buffers, 'index.pkl')):
                print(f'the index.pkl file should be under {init_buffers}!')
                exit(-1)
            num_files, file_size, file_md5 = load(
                osp.join(init_buffers, 'index.pkl'))
            h5_files = [osp.abspath(_) for _ in glob.glob(
                osp.join(init_buffers, '*.h5'))]
            print(f'{num_files} of file in index, {len(h5_files)} files in dataset!')
            if len(h5_files) != num_files:
                print('Wrong index file!')
                exit(0)
            else:
                for name in h5_files:
                    from mani_skill_learn.utils.data import get_one_shape
                    self.h5_files.append(h5py.File(name, 'r'))
                    length = get_one_shape(self.h5_files[-1])[0]
                    index = eval(osp.basename(name).split('.')
                                 [0].split('_')[-1])
                    assert file_size[index] == length
                    assert check_md5sum(name, file_md5[index])
                    self.h5_size.append(file_size[index])
        shuffle(self.h5_files)
        self.h5_idx = 0
        self.idx_in_h5 = 0
        self._update_buffer()

    def _get_h5(self):
        if self.idx_in_h5 < self.h5_size[self.h5_idx]:
            return self.h5_files[self.h5_idx]
        elif self.h5_idx < len(self.h5_files) - 1:
            self.h5_idx += 1
        else:
            shuffle(self.h5_files)
            self.h5_idx = 0
        self.idx_in_h5 = 0
        return self.h5_files[self.h5_idx]

    def _update_buffer(self, batch_size=None):
        if self.running_count < self.capacity:
            num_to_add = self.capacity
        elif self.capacity - self.memory_begin_index < batch_size:
            num_to_add = self.memory_begin_index
        else:
            return
        self.memory_begin_index = 0
        while num_to_add > 0:
            h5 = self._get_h5()
            num_item = min(self.h5_size[self.h5_idx] -
                           self.idx_in_h5, num_to_add)
            item = sample_element_in_dict_array(
                h5, slice(self.idx_in_h5, self.idx_in_h5 + num_item))
            self.push_batch(**item)
            num_to_add -= num_item
            self.idx_in_h5 += num_item
        index = list(range(self.capacity))
        shuffle(index)
        assign_single_element_in_dict_array(self.memory, index, self.memory)

    def sample(self, batch_size):
        assert self.capacity % batch_size == 0
        self._update_buffer(batch_size)
        batch_idx = slice(self.memory_begin_index,
                          self.memory_begin_index + batch_size)
        self.memory_begin_index += batch_size
        return sample_element_in_dict_array(self.memory, batch_idx)


@REPLAYS.register_module()
class TrajReplay:
    """
    Replay buffer uses dict-array as basic data structure, which can be easily saved as hdf5 file.

    See mani_skill_learn/utils/data/dict_array.py for more details.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.reset()

    def __getitem__(self, key):
        return self.memory[key]

    def __len__(self):
        return len(self.main_memory)

    def reset(self):
        self.main_memory = np.array([])
        self.memory = {'obs': {}, 'actions': np.zeros(5)}
        self.position = 0
        self.running_count = 0
        self.processed = False

    def push(self, traj):
        self.main_memory = np.append(self.main_memory, traj)
        self.running_count += 1
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch_idx = np.random.randint(low=0, high=len(self), size=batch_size)
        return self.main_memory[batch_idx]

    def tail_mean(self, num):
        def func(_, __, ___): return np.mean(_[___ - __:___])
        return map_func_to_dict_array(self.main_memory, func, num, len(self))

    def get_all(self):
        return self.main_memory

    def restore(self, init_buffers, replicate_init_buffer=1, num_trajs_per_demo_file=-1):
        buffer_keys = ['obs', 'actions', 'next_obs', 'rewards', 'dones']
        if isinstance(init_buffers, str):
            init_buffers = [init_buffers]
        if isinstance(init_buffers, dict):
            init_buffers = [init_buffers]

        print('Num of datasets', len(init_buffers))
        for _ in range(replicate_init_buffer):
            cnt = 0
            for init_buffer in init_buffers:
                trajs = File(init_buffer, 'r')
                # print(trajs.keys())
                for key in trajs.keys():
                    all_item = load_h5_as_dict_array(trajs[key])
                    item = {buf_key:all_item[buf_key] for buf_key in buffer_keys}
                    self.push(item)
                    cnt += 1
                # if self.running_count >=3000:
                #     break
        print(
            f'Num of buffers {len(init_buffers)}, Total steps {self.running_count}')

    def process(self, backbone, device):
        if self.processed:
            return
        self.processed = True
        for i in range(self.main_memory.shape[0]):
            sampled_batch = dict(
                obs=self.main_memory[i]['obs'], next_obs=self.main_memory[i]['next_obs'])
            sampled_batch = to_torch(
                sampled_batch, device=device, dtype='float32')
            final_obs = to_np(backbone(sampled_batch['obs'])[1])
            final_next_obs = to_np(backbone(sampled_batch['next_obs'])[1])
            self.main_memory[i]['obs'] = np.append(final_obs,[final_next_obs[-1]],axis=0)
            self.main_memory[i] = TrajectoryWithRew(obs=self.main_memory[i]['obs'],
                                                    acts=self.main_memory[i]['actions'],
                                                    rews=self.main_memory[i]['rewards'], infos=None, terminal=True)

        print(f'Processed-{self.main_memory.shape} trajs using backbone')

@REPLAYS.register_module()
class TrajReplayState(TrajReplay):

    def __init__(self, capacity):
        super().__init__(capacity)

    def process(self, backbone = None, device = None):
        if self.processed:
            return
        self.processed = True
        for i in range(self.main_memory.shape[0]):
            final_obs = self.main_memory[i]['obs']
            final_next_obs = self.main_memory[i]['next_obs']
            self.main_memory[i]['obs'] = np.append(final_obs,[final_next_obs[-1]],axis=0)
            self.main_memory[i] = TrajectoryWithRew(obs=self.main_memory[i]['obs'],
                                                    acts=self.main_memory[i]['actions'],
                                                    rews=self.main_memory[i]['rewards'], infos=None, terminal=True)

        print(f'Processed-{self.main_memory.shape} trajs using backbone')

@REPLAYS.register_module()
class TrajReplayStateABS(TrajReplay):

    def __init__(self, capacity, horizon):
        super().__init__(capacity);
        self.horizon = horizon

    def process(self, backbone = None, device = None):
        if self.processed:
            return
        self.processed = True
        for i in range(self.main_memory.shape[0]):
            final_obs = add_absorbing_state(self.main_memory[i]['obs'], self.horizon+1)
            final_actions = add_action_for_absorbing_states(self.main_memory[i]['actions'], self.horizon)
            final_rewards = np.ones(self.main_memory[i]['obs'].shape[0]+1, dtype=self.main_memory[i]['rewards'].dtype)
            final_rewards[:self.main_memory[i]['rewards'].shape[0]]=self.main_memory[i]['rewards']

            self.main_memory[i] = TrajectoryWithRew(obs=final_obs,
                                                    acts=final_actions,
                                                    rews=final_rewards, infos=None, terminal=False)

        print(f'Processed-{self.main_memory.shape} trajs using backbone')