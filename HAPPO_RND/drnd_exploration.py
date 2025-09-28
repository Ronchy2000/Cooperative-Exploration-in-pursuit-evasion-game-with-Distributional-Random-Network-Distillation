import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class MLP(nn.Module):
    """通用全连接网络（靶网络、predictor均可复用）"""
    def __init__(self, input_dim, output_dim, hidden_dim=128, use_orthogonal_init=True):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        if use_orthogonal_init:
            self._init_weights()
    
    def _init_weights(self):
        """正交初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class MultiAgentDRND:
    """
    多智能体DRND探索模块
    核心思想：将RND推广到多智能体场景
    - 每个agent有一个固定的target random network
    - 有一个全局的predictor network
    - 基于全局状态进行内在奖励计算
    """
    
    def __init__(self, args, agent_ids):
        """
        初始化多智能体DRND
        
        Args:
            args: 包含配置参数的对象
            agent_ids: 智能体ID列表
        """
        self.args = args
        self.agent_ids = agent_ids
        self.num_agents = len(agent_ids)
        self.device = getattr(args, 'device', 'cpu')
        
        # 网络参数
        self.state_dim = args.state_dim  # 全局状态维度（所有agent观察拼接）
        self.target_output_dim = getattr(args, 'drnd_output_dim', 64)  # 靶网络输出维度
        self.hidden_dim = getattr(args, 'drnd_hidden_dim', 128)
        
        # DRND参数
        self.alpha = getattr(args, 'drnd_alpha', 0.9)  # 双阶段探索权重
        self.intrinsic_reward_coeff = getattr(args, 'intrinsic_reward_coeff', 1.0)  # 内在奖励系数
        
        # 初始化网络
        self._init_networks()
        
        # 优化器
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(), 
            lr=getattr(args, 'drnd_lr', 3e-4)
        )
        
        print(f"[DRND] 初始化完成:")
        print(f"  - 智能体数量: {self.num_agents}")
        print(f"  - 全局状态维度: {self.state_dim}")
        print(f"  - 靶网络输出维度: {self.target_output_dim}")
        print(f"  - 双阶段权重α: {self.alpha}")
        print(f"  - 内在奖励系数: {self.intrinsic_reward_coeff}")
    
    def _init_networks(self):
        """初始化网络：每个agent一个固定靶网络 + 一个全局预测网络"""
        
        # 1. 协同靶网络集群（每个agent对应一个固定的random target network）
        self.target_networks = {}
        for agent_id in self.agent_ids:
            target_net = MLP(
                input_dim=self.state_dim,
                output_dim=self.target_output_dim,
                hidden_dim=self.hidden_dim
            ).to(self.device)
            
            # ⭐ 关键步骤：固定靶网络参数，不进行训练
            for param in target_net.parameters():
                param.requires_grad = False
            
            self.target_networks[agent_id] = target_net
            print(f"[DRND] 为{agent_id}创建固定靶网络")
        
        # 2. 全局预测器（可训练，拟合所有靶网络的分布）
        self.predictor = MLP(
            input_dim=self.state_dim,
            output_dim=self.target_output_dim,
            hidden_dim=self.hidden_dim
        ).to(self.device)
        
        print(f"[DRND] 创建全局预测器，输入维度: {self.state_dim}")
    
    def compute_target_statistics(self, global_state):
        """
        计算靶网络分布的统计量
        
        Args:
            global_state: 全局状态张量 (batch_size, state_dim)
            
        Returns:
            mu: 均值 (batch_size, target_output_dim)
            B2: 二阶矩 (batch_size, target_output_dim)
        """
        target_outputs = []
        
        # 获取所有靶网络的输出
        with torch.no_grad():
            for agent_id in self.agent_ids:
                target_output = self.target_networks[agent_id](global_state)
                target_outputs.append(target_output)
        
        # 计算分布统计量
        target_outputs = torch.stack(target_outputs, dim=0)  # (num_agents, batch_size, target_output_dim)
        
        # ⭐ 关键计算：DRND的核心统计量
        mu = torch.mean(target_outputs, dim=0)  # 均值
        B2 = torch.mean(target_outputs ** 2, dim=0)  # 二阶矩
        
        return mu, B2
    
    def compute_intrinsic_reward(self, global_state):
        """
        计算DRND内在奖励
        
        Args:
            global_state: 全局状态张量 (batch_size, state_dim)
            
        Returns:
            intrinsic_reward: 内在奖励 (batch_size, 1)
        """
        if isinstance(global_state, np.ndarray):
            global_state = torch.tensor(global_state, dtype=torch.float32).to(self.device)
        
        if len(global_state.shape) == 1:
            global_state = global_state.unsqueeze(0)
        
        # 1. 计算靶网络分布统计量
        mu, B2 = self.compute_target_statistics(global_state)
        
        # 2. 计算预测器输出
        f_theta = self.predictor(global_state)
        
        # 3. ⭐ DRND双阶段内在奖励计算
        # 第一阶段：预测误差（早期均匀探索）
        b1 = torch.norm(f_theta - mu, p=2, dim=1, keepdim=True) ** 2
        
        # 第二阶段：伪计数估计（后期靶向探索）
        numerator = torch.norm(f_theta, p=2, dim=1, keepdim=True) ** 2 - torch.norm(mu, p=2, dim=1, keepdim=True) ** 2
        denominator = torch.sum(B2, dim=1, keepdim=True) - torch.norm(mu, p=2, dim=1, keepdim=True) ** 2
        
        # 避免除零
        denominator = torch.clamp(denominator, min=1e-8)
        b2 = torch.sqrt(torch.clamp(numerator / denominator, min=0, max=100))  # 限制最大值避免爆炸
        
        # ⭐ 最终内在奖励：双阶段加权
        intrinsic_reward = self.alpha * b1 + (1 - self.alpha) * b2
        
        return intrinsic_reward * self.intrinsic_reward_coeff
    
    def update_predictor(self, global_state):
        """
        更新预测器网络
        
        Args:
            global_state: 全局状态张量 (batch_size, state_dim)
            
        Returns:
            loss: 预测器损失值
        """
        if isinstance(global_state, np.ndarray):
            global_state = torch.tensor(global_state, dtype=torch.float32).to(self.device)
        
        # ⭐ DRND关键训练步骤：随机采样靶网络输出作为目标
        with torch.no_grad():
            # 随机选择一个靶网络的输出作为训练目标
            random_agent_id = np.random.choice(self.agent_ids)
            target_output = self.target_networks[random_agent_id](global_state)
        
        # 预测器输出
        predictor_output = self.predictor(global_state)
        
        # ⭐ DRND标准损失：MSE loss
        loss = F.mse_loss(predictor_output, target_output)
        
        # 更新预测器
        self.predictor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 0.5)  # 梯度裁剪
        self.predictor_optimizer.step()
        
        return loss.item()
    
    def get_intrinsic_rewards_for_batch(self, batch_global_states):
        """
        为一批全局状态计算内在奖励
        
        Args:
            batch_global_states: (batch_size, episode_limit, state_dim)
            
        Returns:
            intrinsic_rewards: (batch_size, episode_limit, 1)
        """
        batch_size, episode_limit, state_dim = batch_global_states.shape
        intrinsic_rewards = []
        
        for t in range(episode_limit):
            states_t = batch_global_states[:, t, :]  # (batch_size, state_dim)
            reward_t = self.compute_intrinsic_reward(states_t)  # (batch_size, 1)
            intrinsic_rewards.append(reward_t)
        
        return torch.stack(intrinsic_rewards, dim=1)  # (batch_size, episode_limit, 1)
    
    def save_networks(self, save_path):
        """保存DRND网络"""
        torch.save({
            'predictor_state_dict': self.predictor.state_dict(),
            'target_networks_state_dict': {
                agent_id: net.state_dict() 
                for agent_id, net in self.target_networks.items()
            },
            'predictor_optimizer_state_dict': self.predictor_optimizer.state_dict(),
            'args': {
                'state_dim': self.state_dim,
                'target_output_dim': self.target_output_dim,
                'hidden_dim': self.hidden_dim,
                'alpha': self.alpha,
                'intrinsic_reward_coeff': self.intrinsic_reward_coeff
            }
        }, save_path)
        print(f"[DRND] 网络已保存到: {save_path}")
    
    def load_networks(self, load_path):
        """加载DRND网络"""
        checkpoint = torch.load(load_path, map_location=self.device)
        
        self.predictor.load_state_dict(checkpoint['predictor_state_dict'])
        for agent_id, state_dict in checkpoint['target_networks_state_dict'].items():
            if agent_id in self.target_networks:
                self.target_networks[agent_id].load_state_dict(state_dict)
        
        self.predictor_optimizer.load_state_dict(checkpoint['predictor_optimizer_state_dict'])
        print(f"[DRND] 网络已从 {load_path} 加载")