import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import *
import numpy as np
import os

# 网络初始化函数
def net_init(m, gain=None, use_relu=True):
    '''网络初始化'''
    use_orthogonal = True
    init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_]
    activate_function = ['tanh', 'relu', 'leaky_relu']
    gain = gain if gain is not None else nn.init.calculate_gain(activate_function[use_relu])
    
    init_method[use_orthogonal](m.weight, gain=gain)
    nn.init.constant_(m.bias, 0)

# Trick 8: orthogonal initialization (保持原有接口)
def orthogonal_init(layer, gain=1.0):
    for name, param in layer.named_parameters():
        if 'bias' in name:
            nn.init.constant_(param, 0)
        elif 'weight' in name:
            nn.init.orthogonal_(param, gain=gain)

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128, trick=None):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.mean_layer = nn.Linear(hidden_2, action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        self.trick = trick if trick is not None else {}
        
        # 使用 orthogonal_init
        if self.trick.get('orthogonal_init', False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.mean_layer, gain=0.01)

    def forward(self, x):
        if self.trick.get('feature_norm', False):
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l1(x))
        if self.trick.get('LayerNorm', False):
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l2(x))
        if self.trick.get('LayerNorm', False):
            x = F.layer_norm(x, x.size()[1:])

        mean = torch.tanh(self.mean_layer(x))
        log_std = self.log_std.expand_as(mean)
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)

        return mean, std

class Actor_discrete(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_1=128, hidden_2=128, trick=None):
        super(Actor_discrete, self).__init__()
        self.l1 = nn.Linear(obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, action_dim)

        self.trick = trick if trick is not None else {}
        
        if self.trick.get('orthogonal_init', False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3, gain=0.01)

    def forward(self, obs):
        if self.trick.get('feature_norm', False):
            obs = F.layer_norm(obs, obs.size()[1:])
        x = F.relu(self.l1(obs))
        if self.trick.get('LayerNorm', False):
            x = F.layer_norm(x, x.size()[1:])
        x = F.relu(self.l2(x))
        if self.trick.get('LayerNorm', False):
            x = F.layer_norm(x, x.size()[1:])
        a_prob = torch.softmax(self.l3(x), dim=1)
        return a_prob

class Critic(nn.Module):
    def __init__(self, dim_info, hidden_1=128, hidden_2=128, trick=None):
        super(Critic, self).__init__()
        # 使用原始观察维度计算全局状态维度
        self.dim_info = dim_info
        self.agent_obs_dims = [val[0] for val in dim_info.values()]
        global_obs_dim = sum(self.agent_obs_dims)
        
        self.l1 = nn.Linear(global_obs_dim, hidden_1)
        self.l2 = nn.Linear(hidden_1, hidden_2)
        self.l3 = nn.Linear(hidden_2, 1)
        
        self.trick = trick if trick is not None else {}
        
        if self.trick.get('orthogonal_init', False):
            net_init(self.l1)
            net_init(self.l2)
            net_init(self.l3)

    def forward(self, s):
        # s应该是全局状态张量 (batch_size, global_state_dim)
        if isinstance(s, (list, tuple)) and len(s) == 1:
            s = s[0]
        elif isinstance(s, dict):
            # 如果输入是字典，按照dim_info的顺序拼接
            s_parts = []
            for agent_id, (obs_dim, _) in self.dim_info.items():
                if agent_id in s:
                    s_parts.append(s[agent_id])
                else:
                    # 为缺失的智能体填充零
                    batch_size = next(iter(s.values())).shape[0]
                    s_parts.append(torch.zeros(batch_size, obs_dim, device=next(iter(s.values())).device))
            s = torch.cat(s_parts, dim=1)
            
        if self.trick.get('feature_norm', False):
            s = F.layer_norm(s, s.size()[1:])
        
        q = F.relu(self.l1(s))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])
        q = F.relu(self.l2(q))
        if self.trick.get('LayerNorm', False):
            q = F.layer_norm(q, q.size()[1:])
        q = self.l3(q)

        return q

class CriticDual(nn.Module):
    """双头 Critic：分别预测外在价值和内在价值"""
    def __init__(self, dim_info, hidden_1=128, hidden_2=128, trick=None):
        super(CriticDual, self).__init__()
        # 使用原始观察维度计算全局状态维度
        self.dim_info = dim_info
        self.agent_obs_dims = [val[0] for val in dim_info.values()]
        global_obs_dim = sum(self.agent_obs_dims)
        
        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(global_obs_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU()
        )
        
        # 外在价值头
        self.ext_head = nn.Linear(hidden_2, 1)
        
        # 内在价值头
        self.int_head = nn.Linear(hidden_2, 1)
        
        self.trick = trick if trick is not None else {}
        
        # 初始化
        if self.trick.get('orthogonal_init', False):
            for layer in self.shared:
                if isinstance(layer, nn.Linear):
                    net_init(layer)
            net_init(self.ext_head)
            net_init(self.int_head)

    def forward(self, s):
        """
        Args:
            s: 全局状态 (batch_size, global_state_dim)
        Returns:
            v_ext: 外在价值 (batch_size, 1)
            v_int: 内在价值 (batch_size, 1)
        """
        # 处理输入
        if isinstance(s, (list, tuple)) and len(s) == 1:
            s = s[0]
        elif isinstance(s, dict):
            s_parts = []
            for agent_id, (obs_dim, _) in self.dim_info.items():
                if agent_id in s:
                    s_parts.append(s[agent_id])
                else:
                    batch_size = next(iter(s.values())).shape[0]
                    s_parts.append(torch.zeros(batch_size, obs_dim, device=next(iter(s.values())).device))
            s = torch.cat(s_parts, dim=1)
        
        if self.trick.get('feature_norm', False):
            s = F.layer_norm(s, s.size()[1:])
        
        # 共享特征
        features = self.shared(s)
        
        # 分别输出外在和内在价值
        v_ext = self.ext_head(features)
        v_int = self.int_head(features)
        
        return v_ext, v_int

class DRNDModel(nn.Module):
    """Distributional Random Network Distillation for MPE environments"""
    def __init__(self, obs_dim, hidden_dim=128, feature_dim=64, num_target=5, device='cpu'):
        super(DRNDModel, self).__init__()
        self.obs_dim = obs_dim
        self.num_target = num_target
        self.device = device
        
        # Predictor network (will be trained)
        self.predictor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        ).to(device)
        
        # Multiple target networks (fixed, randomly initialized)
        self.target_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(obs_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            ).to(device) for _ in range(num_target)
        ])
        
        # Initialize networks
        for p in self.predictor.modules():
            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                nn.init.constant_(p.bias, 0)
        
        # Freeze target networks
        for target_net in self.target_networks:
            for p in target_net.modules():
                if isinstance(p, nn.Linear):
                    nn.init.orthogonal_(p.weight, np.sqrt(2))
                    nn.init.constant_(p.bias, 0)
            for param in target_net.parameters():
                param.requires_grad = False
    
    def forward(self, obs):
        """
        Args:
            obs: (batch_size, obs_dim)
        Returns:
            predict_feature: (batch_size, feature_dim)
            target_features: (num_target, batch_size, feature_dim)
        """
        predict_feature = self.predictor(obs)
        
        target_features = torch.stack([
            target_net(obs) for target_net in self.target_networks
        ], dim=0)  # (num_target, batch_size, feature_dim)
        
        return predict_feature, target_features

class MACTNModel(nn.Module):
    """
    Multi-Agent Collaborative Target Networks (MACTN)
    符合DRND框架，但用N个agent的协同靶网络替代N个随机靶网络
    核心设计：
    - N个协同靶网络（每个agent一个，固定不变）
    - 1个全局预测器（可训练）
    - 所有网络输入全局状态s（解决维度不一致问题）
    """
    def __init__(self, state_dim, num_agents, hidden_dim=128, feature_dim=64, device='cpu'):
        super(MACTNModel, self).__init__()
        self.state_dim = state_dim  # 全局状态维度
        self.num_agents = num_agents  # agent数量（对应DRND的N个target）
        self.feature_dim = feature_dim
        self.device = device
        
        # 全局预测器（可训练，输入全局状态s）
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        ).to(device)
        
        # 协同靶网络集群（N个，每个对应一个agent，固定不变）
        # 每个靶网络输入全局状态s，输出特征向量
        self.collaborative_targets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, feature_dim)
            ).to(device) for _ in range(num_agents)
        ])
        
        # 初始化predictor
        for p in self.predictor.modules():
            if isinstance(p, nn.Linear):
                nn.init.orthogonal_(p.weight, np.sqrt(2))
                nn.init.constant_(p.bias, 0)
        
        # 初始化并冻结协同靶网络
        for target_net in self.collaborative_targets:
            for p in target_net.modules():
                if isinstance(p, nn.Linear):
                    nn.init.orthogonal_(p.weight, np.sqrt(2))
                    nn.init.constant_(p.bias, 0)
            # 冻结参数（固定不变）
            for param in target_net.parameters():
                param.requires_grad = False
    
    def forward(self, global_state):
        """
        Args:
            global_state: (batch_size, state_dim) - 全局状态s
        Returns:
            predict_feature: (batch_size, feature_dim) - 预测器输出
            target_features: (num_agents, batch_size, feature_dim) - N个靶网络输出
        """
        # 全局预测器输出
        predict_feature = self.predictor(global_state)
        
        # N个协同靶网络输出（每个对应一个agent的随机认知视角）
        target_features = torch.stack([
            target_net(global_state) for target_net in self.collaborative_targets
        ], dim=0)  # (num_agents, batch_size, feature_dim)
        
        return predict_feature, target_features
    
    def compute_intrinsic_reward(self, global_state, alpha=0.9):
        """
        计算DRND双阶段内在奖励（完全复用DRND公式）
        Args:
            global_state: (batch_size, state_dim) - 全局状态s
            alpha: 早期/后期探索权重（0.9为DRND最优值）
        Returns:
            intrinsic_reward: (batch_size,) - 内在奖励
            stats: dict - 用于分析的统计量
        """
        with torch.no_grad():
            predict_feature, target_features = self.forward(global_state)
            
            # 计算靶网络分布的统计量
            mu = torch.mean(target_features, dim=0)
            B2 = torch.mean(target_features**2, dim=0)
            
            # b₁ = ||f_θ(s) - μ(s)||²
            b1 = (predict_feature - mu).pow(2).sum(dim=1)
            
            # b₂ = 伪计数估计
            numerator = torch.abs(predict_feature.pow(2) - mu.pow(2))
            denominator = B2 - mu.pow(2) + 1e-8
            variance_ratio = torch.clamp(numerator / denominator, 1e-6, 1.0)
            b2 = torch.sqrt(variance_ratio).sum(dim=1)
            
            # 原始内在奖励
            intrinsic_reward_raw = alpha * b1 + (1 - alpha) * b2
            
            # 应用平方根缩放（RND常用技巧，防止数值过大）
            intrinsic_reward = torch.sqrt(intrinsic_reward_raw + 1e-8)
            # 添加裁剪步骤（DRND关键技术）
            intrinsic_reward = torch.clamp(intrinsic_reward, 0.0, 1.0)

            # 统计量
            stats = {
                'b1_mean': b1.mean().item(),
                'b2_mean': b2.mean().item(),
                'reward_raw_mean': intrinsic_reward_raw.mean().item(),
                'reward_scaled_mean': intrinsic_reward.mean().item(),
                'mu_norm': mu.norm(dim=1).mean().item(),
                'predictor_norm': predict_feature.norm(dim=1).mean().item()
            }
        
        return intrinsic_reward.cpu().numpy(), stats

class Agent:
    def __init__(self, obs_dim, action_dim, dim_info, actor_lr, critic_lr, is_continue, device, trick, use_dual_critic=False):
        """
        Args:
            use_dual_critic: 是否使用双头 Critic
        """
        if is_continue:
            self.actor = Actor(obs_dim, action_dim, trick=trick).to(device)
        else:
            self.actor = Actor_discrete(obs_dim, action_dim, trick=trick).to(device)
        
        # 根据是否使用 DRND 选择 Critic 类型
        if use_dual_critic:
            self.critic = CriticDual(dim_info, trick=trick).to(device)
        else:
            self.critic = Critic(dim_info, trick=trick).to(device)
        
        self.device = device
        self.use_dual_critic = use_dual_critic
        
        # 设置优化器
        if trick.get('adam_eps', False):
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr, eps=1e-5)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr, eps=1e-5)
        else:
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update_actor(self, loss):
        self.actor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()

    def update_critic(self, loss):
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def tanh_action_sample(dist):
    raw_action = dist.rsample()
    action = torch.tanh(raw_action)
    log_prob = dist.log_prob(raw_action).sum(-1)
    log_prob -= (2 * (np.log(2) - raw_action - F.softplus(-2 * raw_action))).sum(-1)
    return action, log_prob

class HAPPO_MPE:
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
        self.agents = {}  # 新的异构智能体字典
        self.dim_info = {}  # 存储每个智能体的维度信息
        
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
        
        # TensorBoard相关
        self.writer = None
        self.adversary_agents = []
        self.good_agents = []

        # 异构构建标志
        self._hetero_built = False
        
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

        # MACTN相关参数
        self.use_drnd = getattr(args, 'use_drnd', False)
        self.drnd_alpha = getattr(args, 'drnd_alpha', 0.9)  # DRND双阶段权重
        self.int_coef = getattr(args, 'int_coef', 1.0)  # 内在奖励系数
        self.drnd_lr = getattr(args, 'drnd_lr', 5e-4)  # MACTN predictor学习率
        self.drnd_update_proportion = getattr(args, 'drnd_update_proportion', 0.25)
        
        # MACTN模型（延迟初始化，等待state_dim确定）
        self.mactn = None
        self.mactn_optimizer = None

    def _build_hetero_if_needed(self):
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

        # 创建异构智能体（使用双 Critic）
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
                trick=self.trick,
                use_dual_critic=self.use_drnd  # DRND 模式使用双 Critic
            )
        
        # 初始化全局MACTN系统
        if self.use_drnd:
            self.mactn = MACTNModel(
                state_dim=self.state_dim,
                num_agents=self.N,
                hidden_dim=128,
                feature_dim=64,
                device=self.device
            )
            self.mactn_optimizer = torch.optim.Adam(
                self.mactn.predictor.parameters(),
                lr=self.drnd_lr,
                eps=1e-5
            )
            print(f"MACTN初始化完成（双Critic模式）: state_dim={self.state_dim}, num_agents={self.N}")
        
        self.agent_index = {agent_id: idx for idx, agent_id in enumerate(self.all_agents)}
        self._hetero_built = True

    def reset_rnn_hidden(self):
        """保持原有接口"""
        pass  # 新架构不使用RNN，保持接口兼容性

    def choose_action(self, obs_dict, evaluate):
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
        """获取价值估计（双 Critic 返回外在+内在）"""
        self._build_hetero_if_needed()
        with torch.no_grad():
            values_ext = []
            values_int = []
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device)
            
            for agent_id in self.all_agents:
                if self.use_drnd:
                    # 双 Critic
                    v_ext, v_int = self.agents[agent_id].critic(s_tensor)
                    values_ext.append(v_ext.squeeze(-1))
                    values_int.append(v_int.squeeze(-1))
                else:
                    # 单 Critic
                    v = self.agents[agent_id].critic(s_tensor)
                    values_ext.append(v.squeeze(-1))
            
            v_ext_n = torch.stack(values_ext, dim=0).cpu().numpy().flatten()
            
            if self.use_drnd:
                v_int_n = torch.stack(values_int, dim=0).cpu().numpy().flatten()
                return v_ext_n, v_int_n
            else:
                return v_ext_n, None

    def compute_intrinsic_rewards(self, global_state):
        """
        计算内在奖励（基于全局状态s）
        Args:
            global_state: numpy array (state_dim,) - 全局状态
        Returns:
            intrinsic_reward: float - 内在奖励
            stats: dict - 统计量
        """
        if not self.use_drnd or self.mactn is None:
            return 0.0, {}
        
        # 转换为tensor并添加batch维度
        s_tensor = torch.FloatTensor(global_state).unsqueeze(0).to(self.device)
        
        # 用MACTN计算内在奖励
        intrinsic_reward, stats = self.mactn.compute_intrinsic_reward(
            s_tensor, alpha=self.drnd_alpha
        )
        
        return intrinsic_reward[0], stats

    def train(self, replay_buffer, total_steps):
        self._build_hetero_if_needed()
        batch = replay_buffer.get_training_data()

        # ========== 第一步：训练MACTN predictor ==========
        mactn_loss_avg = 0
        if self.use_drnd and self.mactn is not None and 'next_s' in batch:
            next_states = batch['next_s'].to(self.device)
            next_states_flat = next_states.reshape(-1, self.state_dim)
            
            num_samples = next_states_flat.shape[0]
            sample_size = int(num_samples * self.drnd_update_proportion)
            indices = torch.randperm(num_samples)[:sample_size]
            sampled_states = next_states_flat[indices]
            
            predict_feature, target_features = self.mactn(sampled_states)
            num_sampled = predict_feature.shape[0]
            target_idx = torch.randint(0, self.N, (num_sampled,), device=self.device)
            selected_targets = target_features[target_idx, torch.arange(num_sampled), :]
            mactn_loss = F.mse_loss(predict_feature, selected_targets.detach())
            
            self.mactn_optimizer.zero_grad()
            mactn_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.mactn.predictor.parameters(), 0.5)
            self.mactn_optimizer.step()
            
            mactn_loss_avg = mactn_loss.item()
        
        # ========== 第二步：分别计算外在和内在 GAE ==========
        with torch.no_grad():
            if self.use_drnd:
                # 外在 GAE（episodic，受 done 影响）
                r_ext = batch['r_ext'].to(self.device)
                v_ext = batch['v_ext'].to(self.device)
                done = batch['done'].to(self.device)
                
                ext_adv = []
                gae_ext = 0
                for t in reversed(range(self.episode_limit)):
                    delta_ext = r_ext[:, t] + self.gamma * v_ext[:, t+1] * (1 - done[:, t]) - v_ext[:, t]
                    gae_ext = delta_ext + self.gamma * self.lamda * (1 - done[:, t]) * gae_ext
                    ext_adv.insert(0, gae_ext)
                ext_adv = torch.stack(ext_adv, dim=1)
                v_ext_target = ext_adv + v_ext[:, :-1]
                
                # 内在 GAE（non-episodic，不受 done 影响）
                r_int = batch['r_int'].to(self.device)
                v_int = batch['v_int'].to(self.device)
                
                int_adv = []
                gae_int = 0
                int_gamma = getattr(self.args, 'int_gamma', 0.99)
                for t in reversed(range(self.episode_limit)):
                    delta_int = r_int[:, t] + int_gamma * v_int[:, t+1] - v_int[:, t]  # ✅ 无 (1-done)
                    gae_int = delta_int + int_gamma * self.lamda * gae_int  # ✅ 无 (1-done)
                    int_adv.insert(0, gae_int)
                int_adv = torch.stack(int_adv, dim=1)
                v_int_target = int_adv + v_int[:, :-1]
                
                # 组合 advantage（而非组合 reward）
                int_coef = getattr(self.args, 'int_coef', 0.01)
                adv = ext_adv + int_coef * int_adv
                
                if self.use_adv_norm:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-5)
            else:
                # 原始单 Critic 逻辑
                rs = batch['r'].to(self.device)
                vs = batch['v'].to(self.device)
                dones = batch['done'].to(self.device)
                
                adv = []
                gae = 0
                for t in reversed(range(self.episode_limit)):
                    delta = rs[:, t] + self.gamma * vs[:, t+1] * (1 - dones[:, t]) - vs[:, t]
                    gae = delta + self.gamma * self.lamda * (1 - dones[:, t]) * gae
                    adv.insert(0, gae)
                adv = torch.stack(adv, dim=1)
                v_target = adv + vs[:, :-1]
                
                if self.use_adv_norm:
                    adv = (adv - adv.mean()) / (adv.std() + 1e-5)

        # ========== 第三步：HAPPO 更新（修改 Critic 部分） ==========
        shuffled_agents = np.random.permutation(self.all_agents).tolist()
        factor = torch.ones((self.batch_size, self.episode_limit, 1), device=adv.device)
        critic_losses = {}
        
        eps = 1e-6
        a_n_raw = torch.atanh(torch.clamp(batch['a'].to(self.device), -1 + eps, 1 - eps))
        
        for agent_id in shuffled_agents:
            idx = self.agent_index[agent_id]
            agent_obs_dim = self.dim_info[agent_id][0]
            
            obs_agent_padded = batch['obs'][:, :, idx, :]
            obs_agent = obs_agent_padded[:, :, :agent_obs_dim]
            a_agent_raw = a_n_raw[:, :, idx, :]
            
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
            
            agent_critic_losses = []
            
            for _ in range(self.K_epochs):
                for index in BatchSampler(SequentialSampler(range(self.batch_size)), self.mini_batch_size, False):
                    factor_batch = factor[index]
                    actor_input_batch = actor_input[index]
                    a_agent_raw_batch = a_agent_raw[index]
                    adv_batch = adv[index, :, idx].unsqueeze(-1)
                    
                    # Actor 更新
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
                    
                    # Critic 更新（双头）
                    s_global_batch = batch['s'][index, :-1, :]
                    s_global_reshaped = s_global_batch.reshape(-1, s_global_batch.size(-1))
                    
                    if self.use_drnd:
                        # 双 Critic
                        v_ext_now, v_int_now = self.agents[agent_id].critic(s_global_reshaped)
                        v_ext_now = v_ext_now.view(len(index), self.episode_limit, 1)
                        v_int_now = v_int_now.view(len(index), self.episode_limit, 1)
                        
                        v_ext_target_batch = v_ext_target[index, :, idx].unsqueeze(-1)
                        v_int_target_batch = v_int_target[index, :, idx].unsqueeze(-1)
                        
                        # 分别计算损失
                        critic_ext_loss = ((v_ext_now - v_ext_target_batch) ** 2).mean()
                        critic_int_loss = ((v_int_now - v_int_target_batch) ** 2).mean()
                        critic_loss = critic_ext_loss + critic_int_loss
                    else:
                        # 单 Critic
                        v_now = self.agents[agent_id].critic(s_global_reshaped)
                        v_now = v_now.view(len(index), self.episode_limit, 1)
                        v_target_batch = v_target[index, :, idx].unsqueeze(-1)
                        critic_loss = ((v_now - v_target_batch) ** 2).mean()
                    
                    agent_critic_losses.append(critic_loss.item())
                    self.agents[agent_id].update_critic(critic_loss)
            
            if agent_critic_losses:
                critic_losses[agent_id] = sum(agent_critic_losses) / len(agent_critic_losses)
            
            # 更新保护因子
            with torch.no_grad():
                mean_new, std_new = self.agents[agent_id].actor(actor_input)
                dist_new = Normal(mean_new, std_new)
                log_prob_new = dist_new.log_prob(a_agent_raw).sum(-1, keepdim=True)
                log_prob_new -= (2 * (np.log(2) - a_agent_raw - F.softplus(-2 * a_agent_raw))).sum(-1, keepdim=True)
                factor = factor * torch.exp(log_prob_new - log_prob_old).detach()
        
        if self.use_lr_decay:
            self.lr_decay(total_steps)
        
        critic_losses['mactn_loss'] = mactn_loss_avg
        return critic_losses

    def lr_decay(self, total_steps):
        lr_now = self.lr * (1 - total_steps / self.max_train_steps)
        for agent_id in self.all_agents:
            for p in self.agents[agent_id].actor_optimizer.param_groups:
                p['lr'] = lr_now
            for p in self.agents[agent_id].critic_optimizer.param_groups:
                p['lr'] = lr_now

    def save_model(self, env_name, number, seed, total_steps, time_stamp):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        timestamp_dir = os.path.join(models_dir, time_stamp)
        os.makedirs(timestamp_dir, exist_ok=True)
        
        actor_path = os.path.join(timestamp_dir, f"HAPPO_actor_env_{env_name}_number_{number}_seed_{seed}_step_{int(total_steps / 1000)}k.pth")
        
        self._build_hetero_if_needed()
        # 添加调试信息
        # print(f"保存模型到: {actor_path}")
        # print(f"智能体列表: {self.all_agents}")
        # print(f"构建的智能体: {list(self.agents.keys())}")
        save_obj = {
            'format': 'hetero_per_agent_actor_v2',
            'agents': self.all_agents,
            'actor_state_dict_by_agent': {aid: self.agents[aid].actor.state_dict() for aid in self.all_agents}
        }
        torch.save(save_obj, actor_path)

    def load_model(self, env_name, number, seed, step, timestamp=None):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        
        if timestamp:
            timestamp_dir = os.path.join(models_dir, timestamp)
        else:
            timestamp_dir = models_dir
            
        actor_path = os.path.join(timestamp_dir, f"HAPPO_actor_env_{env_name}_number_{number}_seed_{seed}_step_{step}k.pth")
        
        data = torch.load(actor_path)
        self._build_hetero_if_needed()
        
        if isinstance(data, dict) and 'actor_state_dict_by_agent' in data:
            for aid, state in data['actor_state_dict_by_agent'].items():
                if aid in self.agents:
                    self.agents[aid].actor.load_state_dict(state)
            print("加载异构模型:", actor_path)
        else:
            for aid in self.agents:
                self.agents[aid].actor.load_state_dict(data)
            print("加载共享模型并复制到所有智能体:", actor_path)

    # 为了兼容性，保留旧的接口
    @property
    def actor_by_agent(self):
        return {aid: agent.actor for aid, agent in self.agents.items()}
    
    @property
    def critic_by_agent(self):
        return {aid: agent.critic for aid, agent in self.agents.items()}