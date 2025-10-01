import numpy as np
import torch


class ReplayBuffer:
    """
    HAPPO ReplayBuffer for heterogeneous agents with different action dimensions.
    Extended to support MACTN with global state storage for intrinsic reward computation.
    """
    def __init__(self, args):
        self.N = args.N
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim = getattr(args, 'action_dim', 5)
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.all_agents = getattr(args, 'agents', [f'agent_{i}' for i in range(self.N)])
        self.use_drnd = getattr(args, 'use_drnd', False)

        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()

    def _empty(self, *shape):
        """Convenience: allocate float32 Numpy array."""
        return np.empty(shape, dtype=np.float32)

    def reset_buffer(self):
        """Allocate a fresh buffer for a new batch of episodes."""
        self.buffer = {
            'obs': self._empty(self.batch_size, self.episode_limit, self.N, self.obs_dim),
            's': self._empty(self.batch_size, self.episode_limit + 1, self.state_dim),
            'a': self._empty(self.batch_size, self.episode_limit, self.N, self.action_dim),
            'a_logprob': self._empty(self.batch_size, self.episode_limit, self.N),
            'done': self._empty(self.batch_size, self.episode_limit, self.N),
        }
        
        # 根据是否使用 DRND 分别存储外在/内在奖励和价值
        if self.use_drnd:
            self.buffer['r_ext'] = self._empty(self.batch_size, self.episode_limit, self.N)
            self.buffer['r_int'] = self._empty(self.batch_size, self.episode_limit, self.N)
            self.buffer['v_ext'] = self._empty(self.batch_size, self.episode_limit + 1, self.N)
            self.buffer['v_int'] = self._empty(self.batch_size, self.episode_limit + 1, self.N)
            self.buffer['next_s'] = self._empty(self.batch_size, self.episode_limit, self.state_dim)
        else:
            self.buffer['r'] = self._empty(self.batch_size, self.episode_limit, self.N)
            self.buffer['v'] = self._empty(self.batch_size, self.episode_limit + 1, self.N)
        
        self.episode_num = 0

    
    def store_transition(self, episode_step, obs_n, s, v_n, a_dict, a_logprob_n, r_n, done_n, 
                        next_s=None, r_ext_n=None, r_int_n=None, v_ext_n=None, v_int_n=None):
        """
        Store one timestep for every agent in the current episode.
        Args:
            episode_step: current step in episode
            obs_n: (N, obs_dim) - padded observations
            s: (state_dim,) - global state
            v_n: (N,) - value estimates
            a_dict: {agent_id: action_array} - actions
            a_logprob_n: (N,) - log probabilities
            r_n: (N,) - rewards (已包含内在奖励)
            done_n: (N,) - done flags
            next_s: (state_dim,) - next global state (用于MACTN训练)
            r_int_n: (N,) - intrinsic rewards (用于统计)
        """
        """
        存储转换
        Args:
            v_n: 如果不使用DRND，是 (N,)；如果使用DRND，则传入 v_ext_n 和 v_int_n
            r_n: 如果不使用DRND，是组合奖励；如果使用DRND，则传入 r_ext_n 和 r_int_n
        """
        idx = self.episode_num
        
        self.buffer['obs'][idx, episode_step] = obs_n
        self.buffer['s'][idx, episode_step] = s
        self.buffer['done'][idx, episode_step] = done_n
        
        # 存储动作
        for i, agent_id in enumerate(self.all_agents):
            if agent_id in a_dict:
                self.buffer['a'][idx, episode_step, i] = a_dict[agent_id]
        
        # 存储对数概率
        if a_logprob_n is not None:
            self.buffer['a_logprob'][idx, episode_step] = a_logprob_n
        
        # ✅ 根据模式存储
        if self.use_drnd:
            self.buffer['r_ext'][idx, episode_step] = r_ext_n
            self.buffer['r_int'][idx, episode_step] = r_int_n
            self.buffer['v_ext'][idx, episode_step] = v_ext_n
            self.buffer['v_int'][idx, episode_step] = v_int_n
            if next_s is not None:
                self.buffer['next_s'][idx, episode_step] = next_s
        else:
            self.buffer['r'][idx, episode_step] = r_n
            self.buffer['v'][idx, episode_step] = v_n

    def store_last_value(self, episode_step, v_n, v_ext_n=None, v_int_n=None):
        """存储最后的价值估计"""
        if self.use_drnd:
            self.buffer['v_ext'][self.episode_num, episode_step] = v_ext_n
            self.buffer['v_int'][self.episode_num, episode_step] = v_int_n
        else:
            self.buffer['v'][self.episode_num, episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        """Convert the whole buffer to torch tensors."""
        batch = {}
        
        for k, v in self.buffer.items():
            batch[k] = torch.tensor(v, dtype=torch.float32)
        
        return batch