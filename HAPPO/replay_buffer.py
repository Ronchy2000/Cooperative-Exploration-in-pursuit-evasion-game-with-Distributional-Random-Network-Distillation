import numpy as np
import torch


class ReplayBuffer:
    """
    HAPPO ReplayBuffer for heterogeneous agents with different action dimensions.
    """
    def __init__(self, args):
        self.N = args.N                          # number of agents
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.action_dim_n = getattr(args, 'action_dim_n', [args.action_dim] * self.N)  # 支持异构动作
        self.episode_limit = args.episode_limit
        self.batch_size = args.batch_size
        self.all_agents = getattr(args, 'agents', [f'agent_{i}' for i in range(self.N)])

        self.episode_num = 0
        self.buffer = None
        self.reset_buffer()

    def _empty(self, *shape):
        """Convenience: allocate float32 Numpy array."""
        return np.empty(shape, dtype=np.float32)

    def reset_buffer(self):
        """Allocate a fresh buffer for a new batch of episodes."""
        self.buffer = {
            # observations: (B, T, N, obs_dim)
            'obs_n': self._empty(self.batch_size, self.episode_limit, self.N, self.obs_dim),
            
            # global state: (B, T, state_dim)
            's': self._empty(self.batch_size, self.episode_limit, self.state_dim),
            
            # value estimates: (B, T + 1, N)
            'v_n': self._empty(self.batch_size, self.episode_limit + 1, self.N),
            
            # actions: 字典形式存储，每个智能体单独存储
            'a_n': {
                agent_id: self._empty(self.batch_size, self.episode_limit, action_dim)
                for agent_id, action_dim in zip(self.all_agents, self.action_dim_n)
            },
            
            # old log‑probs: (B, T, N)
            'a_logprob_n': self._empty(self.batch_size, self.episode_limit, self.N),
            
            # rewards: (B, T, N)
            'r_n': self._empty(self.batch_size, self.episode_limit, self.N),
            
            # done flags: (B, T, N)
            'done_n': self._empty(self.batch_size, self.episode_limit, self.N),
        }
        self.episode_num = 0

    def store_transition(self, episode_step, obs_n, s, v_n, a_dict, a_logprob_n, r_n, done_n):
        """
        Store one timestep for every agent in the current episode.
        Args:
            a_dict: 字典形式的动作 {agent_id: action_array}
            其他参数保持不变
        """
        idx = self.episode_num
        self.buffer['obs_n'][idx, episode_step] = obs_n
        self.buffer['s'][idx, episode_step] = s
        self.buffer['v_n'][idx, episode_step] = v_n
        
        # 存储每个智能体的动作
        for agent_id, action in a_dict.items():
            self.buffer['a_n'][agent_id][idx, episode_step] = action
            
        self.buffer['a_logprob_n'][idx, episode_step] = a_logprob_n
        self.buffer['r_n'][idx, episode_step] = r_n
        self.buffer['done_n'][idx, episode_step] = done_n

    def store_last_value(self, episode_step, v_n):
        """After an episode ends, store the bootstrap value for V(s_{T})."""
        self.buffer['v_n'][self.episode_num, episode_step] = v_n
        self.episode_num += 1

    def get_training_data(self):
        """Convert the whole buffer to torch tensors."""
        batch = {}
        
        # 转换非动作数据
        for k, v in self.buffer.items():
            if k != 'a_n':
                batch[k] = torch.tensor(v, dtype=torch.float32)
        
        # 转换动作数据
        batch['a_n'] = {
            agent_id: torch.tensor(actions, dtype=torch.float32)
            for agent_id, actions in self.buffer['a_n'].items()
        }
        
        return batch