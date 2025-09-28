import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import *
import numpy as np
import os

# 从原始HAPPO文件导入基础组件
from HAPPO import (
    net_init, orthogonal_init, Actor, Actor_discrete, Critic, Agent, 
    huber_loss, tanh_action_sample
)

# 导入DRND模块
from drnd_exploration import MultiAgentDRND

class HAPPO_MPE_with_DRND:
    """
    集成DRND探索的HAPPO算法
    
    主要改进：
    1. 集成多智能体DRND探索模块
    2. 在训练过程中计算和使用内在奖励
    3. 更新预测器网络
    """
    
    def __init__(self, args):
        self.args = args
        self.env = None
        self.N = args.N
        self.action_dim = args.action_dim
        self.obs_dim = args.obs_dim
        self.state_dim = args.state_dim
        self.episode_limit = args.episode_limit
        self.rnn_hidden_dim = getattr(args, 'rnn_hidden_dim', 64)

        # 异构智能体相关
        self.all_agents = []
        self.agent_index = {}
        self.agents = {}
        self.dim_info = {}
        
        # 训练参数
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr
        self.gamma = args.gamma
        self.lamda = args.lamda
        self.epsilon = args.epsilon
        self.K_epochs = args.K_epochs
        self.entropy_coef = args.entropy_coef
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.use_rnn = args.use_rnn
        self.add_agent_id = args.add_agent_id
        self.use_value_clip = args.use_value_clip
        self.device = args.device
        
        # ⭐ DRND相关参数
        self.use_drnd = getattr(args, 'use_drnd', True)
        self.drnd_update_freq = getattr(args, 'drnd_update_freq', 1)  # DRND更新频率
        
        # 异构构建标志
        self._hetero_built = False
        self._drnd_initialized = False
        
        # 配置tricks
        self.trick = {
            'adv_norm': self.use_adv_norm,
            'orthogonal_init': getattr(args, 'use_orthogonal_init', True),
            'adam_eps': self.set_adam_eps,
            'lr_decay': self.use_lr_decay,
            'ValueClip': self.use_value_clip,
            'feature_norm': False,
            'LayerNorm': False,
            'huber_loss': False,
        }

    def _build_hetero_if_needed(self):
        """构建异构智能体（与原HAPPO相同）"""
        if self._hetero_built:
            return
        if not self.all_agents or len(self.all_agents) == 0:
            return

        # 获取每个智能体的观察维度
        if self.env is not None:
            obs_dims = self.env.get_obs_dims() if hasattr(self.env, 'get_obs_dims') else {}
        else:
            obs_dims = {}

        # 构建dim_info
        for agent_id in self.all_agents:
            if agent_id in obs_dims:
                obs_dim = obs_dims[agent_id]
            else:
                obs_dim = self.obs_dim
            self.dim_info[agent_id] = [obs_dim, self.action_dim]

        # 创建异构智能体
        for agent_id in self.all_agents:
            obs_dim = self.dim_info[agent_id][0]
            if self.add_agent_id:
                obs_dim += self.N
                
            self.agents[agent_id] = Agent(
                obs_dim=obs_dim,
                action_dim=self.action_dim,
                dim_info=self.dim_info,
                actor_lr=self.lr,
                critic_lr=self.lr,
                is_continue=True,
                device=self.device,
                trick=self.trick
            )

        # 设置agent索引
        self.agent_index = {agent_id: idx for idx, agent_id in enumerate(self.all_agents)}
        self._hetero_built = True
        
        # ⭐ 初始化DRND模块
        if self.use_drnd and not self._drnd_initialized:
            self._init_drnd()

    def _init_drnd(self):
        """初始化DRND探索模块"""
        if not self.all_agents:
            print("[DRND] 警告：智能体列表为空，无法初始化DRND")
            return
            
        self.drnd = MultiAgentDRND(self.args, self.all_agents)
        self._drnd_initialized = True
        print(f"[DRND] 多智能体DRND探索模块初始化完成，智能体: {self.all_agents}")

    def reset_rnn_hidden(self):
        """保持原有接口"""
        pass

    def choose_action(self, obs_dict, evaluate):
        """选择动作（与原HAPPO相同）"""
        self._build_hetero_if_needed()
        with torch.no_grad():
            active_agents = list(obs_dict.keys())
            actions_dict = {}
            logprobs_dict = {}
            
            for agent_id in active_agents:
                obs = torch.tensor(obs_dict[agent_id], dtype=torch.float32).unsqueeze(0).to(self.device)
                if self.add_agent_id:
                    idx = self.agent_index.get(agent_id, 0)
                    one_hot = torch.eye(self.N)[idx].unsqueeze(0).to(self.device)
                    actor_input = torch.cat([obs, one_hot], dim=-1)
                else:
                    actor_input = obs
                    
                mean, std = self.agents[agent_id].actor(actor_input)
                dist = Normal(mean, std)
                
                if evaluate:
                    a = torch.tanh(mean)
                    actions_dict[agent_id] = a.squeeze(0).cpu().numpy()
                else:
                    a, logp = tanh_action_sample(dist)
                    actions_dict[agent_id] = a.squeeze(0).cpu().numpy()
                    logprobs_dict[agent_id] = logp.squeeze(0).item()
                    
            if evaluate:
                return actions_dict, None
            return actions_dict, logprobs_dict

    def get_value(self, s):
        """获取价值函数（与原HAPPO相同）"""
        self._build_hetero_if_needed()
        with torch.no_grad():
            values = []
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            for agent_id in self.all_agents:
                v = self.agents[agent_id].critic(s_tensor)
                values.append(v.squeeze(-1))
                
            v_n = torch.stack(values, dim=0)
            return v_n.cpu().numpy().flatten()

    def train(self, replay_buffer, total_steps):
        """
        ⭐ 训练函数：集成DRND探索
        主要改进：
        1. 计算内在奖励
        2. 更新预测器网络
        3. 将内在奖励加入总奖励
        """
        self._build_hetero_if_needed()
        batch = replay_buffer.get_training_data()

        # ⭐ DRND步骤1：计算内在奖励
        intrinsic_rewards = None
        if self.use_drnd and self._drnd_initialized:
            with torch.no_grad():
                # ⭐ 修复：确保状态和奖励的时间维度匹配
                # 检查实际的形状
                print(f"[DRND] 调试信息:")
                print(f"  - batch['s']完整形状: {batch['s'].shape}")
                print(f"  - batch['r_n']形状: {batch['r_n'].shape}")
                print(f"  - batch['v_n']形状: {batch['v_n'].shape}")
                
                # 根据实际情况调整：使用前episode_limit个状态，与奖励匹配
                if batch['s'].shape[1] == batch['r_n'].shape[1] + 1:
                    # 如果状态比奖励多1个时间步，去掉最后一个
                    global_states = batch['s'][:, :-1]  # (batch_size, episode_limit, state_dim)
                elif batch['s'].shape[1] == batch['r_n'].shape[1]:
                    # 如果状态和奖励时间步相同，直接使用
                    global_states = batch['s']  # (batch_size, episode_limit, state_dim)
                else:
                    # 其他情况，尝试取最小长度
                    min_len = min(batch['s'].shape[1], batch['r_n'].shape[1])
                    global_states = batch['s'][:, :min_len]
                    print(f"[DRND] 警告：状态和奖励维度不匹配，使用最小长度: {min_len}")
                
                print(f"  - 调整后global_states形状: {global_states.shape}")
                
                # 计算内在奖励
                intrinsic_rewards = self.drnd.get_intrinsic_rewards_for_batch(global_states)
                intrinsic_rewards = intrinsic_rewards.squeeze(-1)  # (batch_size, episode_limit)
                
                print(f"  - intrinsic_rewards形状: {intrinsic_rewards.shape}")
                print(f"[DRND] 内在奖励统计 - 均值: {intrinsic_rewards.mean().item():.4f}, "
                      f"标准差: {intrinsic_rewards.std().item():.4f}, "
                      f"最大值: {intrinsic_rewards.max().item():.4f}")
            
            # DRND步骤2：更新预测器网络
            if total_steps % self.drnd_update_freq == 0:
                # 随机采样一些状态来更新预测器
                batch_size, episode_limit, state_dim = global_states.shape
                total_states = batch_size * episode_limit
                sample_size = min(256, total_states)
                
                sample_indices = np.random.choice(
                    total_states, 
                    size=sample_size,
                    replace=False
                )
                
                # 使用reshape而不是view，并确保tensor连续
                sampled_states = global_states.contiguous().reshape(-1, state_dim)[sample_indices]
                predictor_loss = self.drnd.update_predictor(sampled_states)
                
                if total_steps % (self.drnd_update_freq * 100) == 0:
                    print(f"[DRND] 步骤 {total_steps}: 预测器损失 = {predictor_loss:.6f}")

        # ⭐ DRND步骤3：将内在奖励加入环境奖励
        if intrinsic_rewards is not None:
            # ⭐ 修复：确保维度匹配并截取相应长度
            reward_time_len = batch['r_n'].shape[1]
            intrinsic_time_len = intrinsic_rewards.shape[1]
            
            if intrinsic_time_len != reward_time_len:
                # 如果长度不匹配，取最小长度
                min_len = min(intrinsic_time_len, reward_time_len)
                intrinsic_rewards_trimmed = intrinsic_rewards[:, :min_len]
                env_rewards_trimmed = batch['r_n'][:, :min_len]
                print(f"[DRND] 维度修正：截取到长度 {min_len}")
            else:
                intrinsic_rewards_trimmed = intrinsic_rewards
                env_rewards_trimmed = batch['r_n']
            
            # 扩展内在奖励到所有智能体
            expanded_intrinsic_rewards = intrinsic_rewards_trimmed.unsqueeze(-1).expand(-1, -1, self.N)
            print(f"[DRND] 扩展后内在奖励形状: {expanded_intrinsic_rewards.shape}")
            print(f"[DRND] 环境奖励形状: {env_rewards_trimmed.shape}")
            
            # 加入到环境奖励中
            total_rewards = env_rewards_trimmed + expanded_intrinsic_rewards
            print(f"[DRND] 奖励组合 - 环境奖励均值: {env_rewards_trimmed.mean().item():.4f}, "
                  f"内在奖励均值: {expanded_intrinsic_rewards.mean().item():.4f}, "
                  f"总奖励均值: {total_rewards.mean().item():.4f}")
            
            # ⭐ 重要：替换batch中的奖励为修正后的总奖励
            if intrinsic_time_len != reward_time_len:
                # 如果长度不同，需要调整其他相关tensor
                batch['r_n'] = total_rewards
                # 同时需要调整其他相关的tensor长度
                actual_len = total_rewards.shape[1]
                batch['v_n'] = batch['v_n'][:, :actual_len+1]  # 价值函数比奖励多一个时间步
                batch['done_n'] = batch['done_n'][:, :actual_len]
                batch['obs_n'] = batch['obs_n'][:, :actual_len]
                for agent_id in batch['a_n']:
                    batch['a_n'][agent_id] = batch['a_n'][agent_id][:, :actual_len]
                batch['a_logprob_n'] = batch['a_logprob_n'][:, :actual_len]
                print(f"[DRND] 已调整所有tensor长度为: {actual_len}")
        else:
            total_rewards = batch['r_n']

        # 使用总奖励（环境+内在）计算GAE优势
        adv = []
        gae = 0
        with torch.no_grad():
            deltas = total_rewards + self.gamma * batch['v_n'][:, 1:] * (1 - batch['done_n']) - batch['v_n'][:, :-1]
            for t in reversed(range(self.episode_limit)):
                gae = deltas[:, t] + self.gamma * self.lamda * gae
                adv.insert(0, gae)
            adv = torch.stack(adv, dim=1)
            v_target = adv + batch['v_n'][:, :-1]
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # HAPPO训练逻辑（与原代码相同）
        shuffled_agents = np.random.permutation(self.all_agents).tolist()
        factor = torch.ones((self.batch_size, self.episode_limit, 1), device=adv.device)
        
        eps = 1e-6
        a_n_raw_dict = {}
        for agent_id in self.all_agents:
            a_agent = batch['a_n'][agent_id]
            a_n_raw_dict[agent_id] = torch.atanh(torch.clamp(a_agent, -1 + eps, 1 - eps))
        
        for agent_id in shuffled_agents:
            idx = self.agent_index[agent_id]
            agent_obs_dim = self.dim_info[agent_id][0]
            
            obs_agent_padded = batch['obs_n'][:, :, idx, :]
            obs_agent = obs_agent_padded[:, :, :agent_obs_dim]
            a_agent_raw = a_n_raw_dict[agent_id]
            
            if self.add_agent_id:
                one_hot = torch.eye(self.N, device=adv.device)[idx].view(1, 1, -1).repeat(self.batch_size, self.episode_limit, 1)
                actor_input = torch.cat([obs_agent, one_hot], dim=-1)
            else:
                actor_input = obs_agent
                
            with torch.no_grad():
                mean_old, std_old = self.agents[agent_id].actor(actor_input)
                dist_old = Normal(mean_old, std_old)
                log_prob_old = dist_old.log_prob(a_agent_raw).sum(-1, keepdim=True)
                log_prob_old -= (2 * (np.log(2) - a_agent_raw - F.softplus(-2 * a_agent_raw))).sum(-1, keepdim=True)
            
            for _ in range(self.K_epochs):
                for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                    factor_batch = factor[index]
                    actor_input_batch = actor_input[index]
                    a_agent_raw_batch = a_agent_raw[index]
                    adv_batch = adv[index, :, idx].unsqueeze(-1)
                    v_target_batch = v_target[index, :, idx].unsqueeze(-1)
                    
                    mean, std = self.agents[agent_id].actor(actor_input_batch)
                    dist = Normal(mean, std)
                    
                    log_prob = dist.log_prob(a_agent_raw_batch).sum(-1, keepdim=True)
                    log_prob -= (2 * (np.log(2) - a_agent_raw_batch - F.softplus(-2 * a_agent_raw_batch))).sum(-1, keepdim=True)
                    
                    log_prob_old_batch = log_prob_old[index]
                    ratios = torch.exp(log_prob - log_prob_old_batch)
                    
                    surr1 = ratios * adv_batch * factor_batch
                    surr2 = torch.clamp(ratios, 1-self.epsilon, 1+self.epsilon) * adv_batch * factor_batch
                    actor_loss = -torch.min(surr1, surr2).mean()
                    
                    entropy = dist.entropy().sum(-1, keepdim=True).mean()
                    actor_loss -= self.entropy_coef * entropy
                    
                    self.agents[agent_id].update_actor(actor_loss)
                    
                    # critic值计算
                    s_global_batch = batch['s'][index]  # (mini_batch_size, episode_limit, state_dim)
                    
                    # 将批次数据重塑为 (batch_size * episode_limit, state_dim)
                    s_global_reshaped = s_global_batch.contiguous().reshape(-1, s_global_batch.size(-1))
                    v_now = self.agents[agent_id].critic(s_global_reshaped)
                    # 重塑回 (batch_size, episode_limit, 1)
                    v_now = v_now.reshape(len(index), self.episode_limit, 1)
                    
                    if self.use_value_clip:
                        v_old = batch['v_n'][index, :-1, idx].unsqueeze(-1).detach()
                        v_clipped = v_old + torch.clamp(v_now - v_old, -self.epsilon, self.epsilon)
                        critic_loss1 = (v_now - v_target_batch)**2
                        critic_loss2 = (v_clipped - v_target_batch)**2
                        critic_loss = torch.max(critic_loss1, critic_loss2).mean()
                    else:
                        critic_loss = ((v_now - v_target_batch)**2).mean()
                    
                    self.agents[agent_id].update_critic(critic_loss)
            
            with torch.no_grad():
                mean_new, std_new = self.agents[agent_id].actor(actor_input)
                dist_new = Normal(mean_new, std_new)
                log_prob_new = dist_new.log_prob(a_agent_raw).sum(-1, keepdim=True)
                log_prob_new -= (2 * (np.log(2) - a_agent_raw - F.softplus(-2 * a_agent_raw))).sum(-1, keepdim=True)
                
                factor = factor * torch.exp(log_prob_new - log_prob_old).detach()
        
        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        """学习率衰减（与原HAPPO相同）"""
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for agent_id in self.all_agents:
            for p in self.agents[agent_id].actor_optimizer.param_groups:
                p['lr'] = lr_now
            for p in self.agents[agent_id].critic_optimizer.param_groups:
                p['lr'] = lr_now

    def save_model(self, env_name, number, seed, total_steps, time_stamp):
        """保存模型（包括DRND网络）"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        timestamp_dir = os.path.join(models_dir, time_stamp)
        os.makedirs(timestamp_dir, exist_ok=True)
        
        # 保存HAPPO模型
        actor_path = os.path.join(timestamp_dir, f"HAPPO_DRND_actor_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
        
        self._build_hetero_if_needed()
        save_obj = {
            'format': 'hetero_per_agent_actor_v2_with_drnd',
            'agents': self.all_agents,
            'actor_state_dict_by_agent': {aid: self.agents[aid].actor.state_dict() for aid in self.all_agents}
        }
        torch.save(save_obj, actor_path)
        
        # ⭐ 保存DRND网络
        if self.use_drnd and self._drnd_initialized:
            drnd_path = os.path.join(timestamp_dir, f"DRND_networks_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
            self.drnd.save_networks(drnd_path)

    def load_model(self, env_name, number, seed, step, timestamp=None):
        """加载模型（包括DRND网络）"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        
        if timestamp:
            timestamp_dir = os.path.join(models_dir, timestamp)
        else:
            timestamp_dir = models_dir
            
        # 加载HAPPO模型
        actor_path = os.path.join(timestamp_dir, f"HAPPO_DRND_actor_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth")
        
        data = torch.load(actor_path)
        self._build_hetero_if_needed()
        
        if isinstance(data, dict) and 'actor_state_dict_by_agent' in data:
            for aid, state in data['actor_state_dict_by_agent'].items():
                if aid in self.agents:
                    self.agents[aid].actor.load_state_dict(state)
            print("加载HAPPO+DRND异构模型:", actor_path)
        else:
            for aid in self.agents:
                self.agents[aid].actor.load_state_dict(data)
            print("加载HAPPO+DRND共享模型并复制到所有智能体:", actor_path)
        
        # ⭐ 加载DRND网络
        if self.use_drnd and self._drnd_initialized:
            drnd_path = os.path.join(timestamp_dir, f"DRND_networks_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth")
            if os.path.exists(drnd_path):
                self.drnd.load_networks(drnd_path)
            else:
                print(f"[DRND] 警告：DRND网络文件不存在: {drnd_path}")

    # 为了兼容性，保留旧的接口
    @property
    def actor_by_agent(self):
        return {aid: agent.actor for aid, agent in self.agents.items()}
    
    @property
    def critic_by_agent(self):
        return {aid: agent.critic for aid, agent in self.agents.items()}