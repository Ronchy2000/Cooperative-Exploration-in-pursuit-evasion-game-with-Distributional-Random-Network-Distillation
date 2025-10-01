import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling, RunningMeanStd
from replay_buffer import ReplayBuffer
from HAPPO_DRND import HAPPO_MPE

from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3
from envs import simple_tag_env_v4_partially
import os
from datetime import datetime
import csv

def get_env(env_name, num_good, num_adversaries, num_obstacles, ep_len=25, render_mode="None", seed=None):
    """create environment and get observation and action dimension of each agent in this environment"""
    new_env = None
    if env_name == 'simple_adversary_v3':
        new_env = simple_adversary_v3.parallel_env(max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_spread_v3':
        new_env = simple_spread_v3.parallel_env(max_cycles=ep_len, render_mode=render_mode, continuous_actions=True)
    if env_name == 'simple_tag_v3':
        new_env = simple_tag_v3.parallel_env(render_mode = render_mode, num_good=num_good, num_adversaries=num_adversaries, num_obstacles=num_obstacles, max_cycles=ep_len, continuous_actions=True)
    if env_name == 'simple_tag_env_v4_partially':
        new_env = simple_tag_env_v4_partially.parallel_env(render_mode = render_mode, num_good=num_good, num_adversaries=num_adversaries, num_obstacles=num_obstacles, max_cycles=ep_len, continuous_actions=True)
    
    # 使用reset时处理None种子
    if seed is not None:
        new_env.reset(seed=seed)
    else:
        new_env.reset()

    _dim_info = {}
    action_bound = {}
    for agent_id in new_env.agents:
        print("agent_id:", agent_id)
        _dim_info[agent_id] = []
        action_bound[agent_id] = []
        _dim_info[agent_id].append(new_env.observation_space(agent_id).shape[0])
        _dim_info[agent_id].append(new_env.action_space(agent_id).shape[0])
        action_bound[agent_id].append(new_env.action_space(agent_id).low)
        action_bound[agent_id].append(new_env.action_space(agent_id).high)
    print("_dim_info:", _dim_info)
    print("action_bound:", action_bound)
    return new_env, _dim_info, action_bound

class Runner_MAPPO_MPE:
    def __init__(self, args, num_good, num_adversaries, num_obstacles, seed, env_name):
        self.args = args
        self.env_name = env_name
        self.num_good = num_good
        self.num_obstacles = num_obstacles
        self.num_adversaries = num_adversaries
        self.seed = seed
        self.number = num_good + num_adversaries
        
        # Set random seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        # 创建训练开始时间戳
        self.timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M')
        print(f"训练开始时间戳: {self.timestamp}")
        
        # 初始化TensorBoard
        self.writer = None
        if args.use_tensorboard:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            log_dir = os.path.join(current_dir, "logs", "tensorboard_logs", f"{env_name}_{self.timestamp}")
            os.makedirs(log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir)
            print(f"TensorBoard日志保存到: {log_dir}")
            print("启动TensorBoard命令: tensorboard --logdir=logs/tensorboard_logs")

        # Create env
        self.env, self.dim_info, _ = get_env(env_name=env_name, num_good=self.num_good, num_adversaries=self.num_adversaries, num_obstacles= self.num_obstacles, ep_len=self.args.episode_limit)
        print("self.env.agents", self.env.agents)

        # 根据智能体ID区分追捕者和逃跑者
        self.adversary_agents = []
        self.good_agents = []
        
        for agent_id in self.env.agents:
            if agent_id.startswith('adversary_'):
                self.adversary_agents.append(agent_id)
            else:
                self.good_agents.append(agent_id)
        
        print(f"追捕者智能体: {self.adversary_agents}")
        print(f"逃跑者智能体: {self.good_agents}")
        
        self.args.agents = self.env.agents
        self.args.N = len(self.env.agents)
        self.args.obs_dim_n = [self.env.observation_space(i).shape[0] for i in self.args.agents]
        self.args.action_dim_n = [self.env.action_space(i).shape[0] for i in self.args.agents]
        
        # 为了处理异构观察，我们使用最大观察维度
        self.max_obs_dim = max(self.args.obs_dim_n)
        self.args.obs_dim = self.max_obs_dim  # 使用最大观察维度
        self.args.action_dim = self.args.action_dim_n[0]
        
        # 添加这一行来修复动作维度不匹配问题
        self.args.act_dim = self.args.action_dim
        
        # 计算全局状态维度（所有智能体观察的总和）
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("max_obs_dim={}".format(self.max_obs_dim))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))

        # Create agent with heterogeneous support
        self.agent_n = HAPPO_MPE(self.args)
        self.agent_n.env = self.env
        self.agent_n.all_agents = [agent_id for agent_id in self.env.agents]
        self.agent_n.dim_info = self.dim_info  # 设置维度信息
        
        # 传递TensorBoard writer给agent
        if self.writer:
            self.agent_n.writer = self.writer
            self.agent_n.adversary_agents = self.adversary_agents
            self.agent_n.good_agents = self.good_agents
        
        # 为环境添加get_obs_dims方法
        def get_obs_dims():
            return {agent_id: self.env.observation_space(agent_id).shape[0] for agent_id in self.env.agents}
        self.env.get_obs_dims = get_obs_dims
        
        self.replay_buffer = ReplayBuffer(self.args)

        self.evaluate_rewards = [] # Total_Reward
        self.evaluate_adversary_rewards = []  # 追捕者总奖励
        self.evaluate_adversary_avg_rewards = []  # 追捕者平均奖励
        self.evaluate_good_rewards = []       # 逃跑者奖励
        self.evaluate_individual_adversary_rewards = {agent_id: [] for agent_id in self.adversary_agents} # 为每个追捕者单独记录奖励
        
        self.total_steps = 0
        
        # 外在奖励归一化（可选）
        if self.args.use_reward_norm:
            print("------use reward norm (extrinsic)------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling (extrinsic)------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)
        elif self.args.use_manual_reward_scale:
            print(f"------use manual reward scaling (factor: {self.args.reward_scale_factor})------")
            self.manual_reward_scaler = lambda r: r * self.args.reward_scale_factor
        
        # MACTN初始化
        self.use_drnd = getattr(args, 'use_drnd', False)
        if self.use_drnd:
            from normalization import RewardForwardFilter
            print("=" * 60)
            print("启用MACTN探索机制（双Critic + Non-episodic）")
            print(f"  - 全局状态维度: {self.args.state_dim}")
            print(f"  - 协同靶网络数量: {self.args.N}")
            print(f"  - 内在奖励系数: {args.int_coef}")
            print(f"  - DRND alpha: {args.drnd_alpha}")
            print(f"  - 内在gamma: {getattr(args, 'int_gamma', 0.99)}")
            print("=" * 60)
            
            # 内在奖励归一化器（运行时标准差）
            self.int_reward_rms = RunningMeanStd(shape=())
            self.int_reward_forward_filter = RewardForwardFilter(gamma=getattr(args, 'int_gamma', 0.99))
        
    def _pad_obs_to_max_dim(self, obs_dict):
        """将不同维度的观察填充到最大维度 - 仅用于replay buffer存储"""
        obs_list = []
        for agent_id in self.agent_n.all_agents:
            if agent_id in obs_dict:
                obs = obs_dict[agent_id]
                # 填充到最大维度
                if len(obs) < self.max_obs_dim:
                    padded_obs = np.zeros(self.max_obs_dim)
                    padded_obs[:len(obs)] = obs
                    obs_list.append(padded_obs)
                else:
                    obs_list.append(obs)
            else:
                # 对于已经完成的智能体，使用零填充
                obs_list.append(np.zeros(self.max_obs_dim))
        return np.array(obs_list)

    def _get_global_state(self, obs_dict):
        """获取全局状态（保持原始维度）- 用于Critic网络和MACTN"""
        state_parts = []
        for agent_id in self.agent_n.all_agents:
            if agent_id in obs_dict:
                state_parts.append(obs_dict[agent_id])
            else:
                # 对于已完成的智能体，使用原始观察维度的零填充
                agent_obs_dim = self.dim_info[agent_id][0]
                state_parts.append(np.zeros(agent_obs_dim))
        return np.concatenate(state_parts)

    def run(self):
        evaluate_num = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            _, _, _, _, episode_steps = self.run_episode_mpe(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                # 训练并获取损失
                critic_losses = self.agent_n.train(self.replay_buffer, self.total_steps)
                # 记录损失到TensorBoard
                if self.writer and critic_losses:
                    self.log_critic_losses(critic_losses, self.total_steps)
                
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.save_rewards_to_csv()
        # 关闭TensorBoard writer
        if self.writer:
            self.writer.close()
        self.env.close()

    def log_critic_losses(self, critic_losses, total_steps):
        """记录Critic损失和MACTN损失到TensorBoard"""
        if not self.writer or not critic_losses:
            return
        
        # 记录MACTN损失
        if 'mactn_loss' in critic_losses:
            self.writer.add_scalar('Loss/MACTN_Predictor', critic_losses['mactn_loss'], total_steps)
        
        # 记录每个智能体的Critic损失
        for agent_id, loss_value in critic_losses.items():
            if agent_id != 'mactn_loss':
                self.writer.add_scalar(f'Loss/Critic/{agent_id}', loss_value, total_steps)
        
        # 分别计算追捕者和逃跑者的平均损失
        adversary_losses = [critic_losses[agent_id] for agent_id in self.adversary_agents if agent_id in critic_losses]
        good_losses = [critic_losses[agent_id] for agent_id in self.good_agents if agent_id in critic_losses]
        
        if adversary_losses:
            avg_adversary_loss = sum(adversary_losses) / len(adversary_losses)
            self.writer.add_scalar('Loss/Critic/Adversaries_Average', avg_adversary_loss, total_steps)
        
        if good_losses:
            avg_good_loss = sum(good_losses) / len(good_losses)
            self.writer.add_scalar('Loss/Critic/Good_Agents_Average', avg_good_loss, total_steps)
        
        # 记录所有智能体的平均损失
        all_losses = [v for k, v in critic_losses.items() if k != 'mactn_loss']
        if all_losses:
            avg_all_loss = sum(all_losses) / len(all_losses)
            self.writer.add_scalar('Loss/Critic/All_Average', avg_all_loss, total_steps)
    
    def evaluate_policy(self):
        evaluate_reward = 0
        evaluate_adversary_reward = 0  # 追捕者总奖励
        evaluate_good_reward = 0       # 逃跑者总奖励
        individual_adversary_rewards = {agent_id: 0 for agent_id in self.adversary_agents} # 为每个追捕者单独统计
        for _ in range(self.args.evaluate_times):
            episode_reward, adversary_reward, good_reward, individual_rewards, _ = self.run_episode_mpe(evaluate=True)
            evaluate_reward += episode_reward
            evaluate_adversary_reward += adversary_reward
            evaluate_good_reward += good_reward
            # 累加每个追捕者的奖励
            for agent_id in self.adversary_agents:
                individual_adversary_rewards[agent_id] += individual_rewards.get(agent_id, 0)

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        evaluate_adversary_reward = evaluate_adversary_reward / self.args.evaluate_times
        evaluate_good_reward = evaluate_good_reward / self.args.evaluate_times
        evaluate_adversary_avg_reward = evaluate_adversary_reward / len(self.adversary_agents) # 计算追捕者平均奖励（总奖励除以追捕者数量）
        # 计算每个追捕者的平均奖励
        for agent_id in self.adversary_agents:
            individual_adversary_rewards[agent_id] = individual_adversary_rewards[agent_id] / self.args.evaluate_times
        
        # 记录评估奖励到TensorBoard
        if self.writer:
            self.writer.add_scalar('Evaluation/Total_Reward', evaluate_reward, self.total_steps)
            self.writer.add_scalar('Evaluation/Adversary_Total_Reward', evaluate_adversary_reward, self.total_steps)
            self.writer.add_scalar('Evaluation/Adversary_Average_Reward', evaluate_adversary_avg_reward, self.total_steps)
            self.writer.add_scalar('Evaluation/Good_Reward', evaluate_good_reward, self.total_steps)
        
        # 保存奖励数据
        self.evaluate_rewards.append(evaluate_reward)
        self.evaluate_adversary_rewards.append(evaluate_adversary_reward)
        self.evaluate_adversary_avg_rewards.append(evaluate_adversary_avg_reward)
        self.evaluate_good_rewards.append(evaluate_good_reward)

        # 保存每个追捕者的奖励
        for agent_id in self.adversary_agents:
            self.evaluate_individual_adversary_rewards[agent_id].append(individual_adversary_rewards[agent_id])
        print("total_steps:{} \t total_reward:{:.2f} \t adversary_total:{:.2f} \t adversary_avg:{:.2f} \t good_reward:{:.2f}".format(
            self.total_steps, evaluate_reward, evaluate_adversary_reward, evaluate_adversary_avg_reward, evaluate_good_reward))

        # 打印每个追捕者的奖励
        for agent_id in self.adversary_agents:
            print(f"  {agent_id}: {individual_adversary_rewards[agent_id]:.2f}")
        
        # Save the model
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps, self.timestamp)

    def run_episode_mpe(self, evaluate=False):
        episode_reward = 0
        episode_adversary_reward = 0
        episode_good_reward = 0
        individual_adversary_rewards = {agent_id: 0 for agent_id in self.adversary_agents}
        episode_intrinsic_reward_raw = 0
        episode_intrinsic_reward_normalized = 0

        obs_dict, _ = self.env.reset()
        done_dict = {agent_id: False for agent_id in self.agent_n.all_agents}
        
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.reset_rnn_hidden()
        
        # 收集本 episode 的所有内在奖励（用于归一化）
        episode_int_rewards = []
            
        for episode_step in range(self.args.episode_limit):
            # 选择动作
            actions_dict, logprobs_dict = self.agent_n.choose_action(obs_dict, evaluate=evaluate)
            obs_n = self._pad_obs_to_max_dim(obs_dict)
            s = self._get_global_state(obs_dict)
            
            # 获取双价值估计
            if self.use_drnd:
                v_ext_n, v_int_n = self.agent_n.get_value(s)
            else:
                v_n, _ = self.agent_n.get_value(s)
            
            # 环境步进
            next_obs_dict, rewards_dict, terminated_dict, truncated_dict, _ = self.env.step(actions_dict)
            
            # 计算内在奖励（原始值）
            intrinsic_reward_raw = 0.0
            int_reward_stats = {}
            
            if not evaluate and self.use_drnd:
                next_s = self._get_global_state(next_obs_dict)
                intrinsic_reward_raw, int_reward_stats = self.agent_n.compute_intrinsic_rewards(next_s)
                episode_int_rewards.append(intrinsic_reward_raw)
                episode_intrinsic_reward_raw += intrinsic_reward_raw
            
            # 环境奖励统计
            step_adversary_reward = sum([rewards_dict.get(agent_id, 0) for agent_id in self.adversary_agents])
            step_good_reward = sum([rewards_dict.get(agent_id, 0) for agent_id in self.good_agents])
            step_total_reward = step_adversary_reward + step_good_reward
            
            for agent_id in self.adversary_agents:
                individual_adversary_rewards[agent_id] += rewards_dict.get(agent_id, 0)
            
            episode_reward += step_total_reward
            episode_adversary_reward += step_adversary_reward
            episode_good_reward += step_good_reward

            # 更新done
            for agent_id in terminated_dict:
                if terminated_dict[agent_id] or truncated_dict[agent_id]:
                    done_dict[agent_id] = True
            done = all(done_dict.values()) or len(self.env.agents) == 0
            done_n = np.array([done] * self.args.N)

            if not evaluate:
                # 构建动作和奖励数组
                a_dict = {}
                
                if self.use_drnd:
                    r_ext_n = np.zeros(self.args.N)
                    r_int_n = np.zeros(self.args.N)
                else:
                    r_n = np.zeros(self.args.N)
                
                for i, agent_id in enumerate(self.agent_n.all_agents):
                    if agent_id in actions_dict:
                        a_dict[agent_id] = actions_dict[agent_id]
                    else:
                        agent_action_dim = self.dim_info[agent_id][1]
                        a_dict[agent_id] = np.zeros(agent_action_dim)
                    
                    # 分别存储外在和内在奖励
                    if agent_id in rewards_dict:
                        ext_reward = rewards_dict[agent_id]
                        
                        if self.use_drnd:
                            # 内在奖励平均分配（暂时不归一化）
                            int_reward_share = intrinsic_reward_raw / self.args.N
                            r_ext_n[i] = ext_reward
                            r_int_n[i] = int_reward_share
                        else:
                            r_n[i] = ext_reward
                
                # 处理logprobs
                if logprobs_dict is not None:
                    a_logprob_n = np.zeros(self.args.N)
                    for i, agent_id in enumerate(self.agent_n.all_agents):
                        if agent_id in logprobs_dict:
                            a_logprob_n[i] = logprobs_dict[agent_id]
                else:
                    a_logprob_n = None
                
                # 只对外在奖励归一化（可选）
                if not self.use_drnd:
                    if self.args.use_reward_norm:
                        r_n = self.reward_norm(r_n)
                    elif self.args.use_reward_scaling:
                        r_n = self.reward_scaling(r_n)

                # 获取next全局状态
                next_s = self._get_global_state(next_obs_dict)
                
                # 存储转换
                if self.use_drnd:
                    self.replay_buffer.store_transition(
                        episode_step, obs_n, s, None, a_dict, a_logprob_n, None, done_n,
                        next_s=next_s,
                        r_ext_n=r_ext_n,
                        r_int_n=r_int_n,
                        v_ext_n=v_ext_n,
                        v_int_n=v_int_n
                    )
                else:
                    self.replay_buffer.store_transition(
                        episode_step, obs_n, s, v_n, a_dict, a_logprob_n, r_n, done_n
                    )

            obs_dict = next_obs_dict
            if done:
                break

        if not evaluate:
            obs_n = self._pad_obs_to_max_dim(obs_dict)
            s = self._get_global_state(obs_dict)
            
            if self.use_drnd:
                # 先归一化本 episode 的内在奖励（在 store_last_value 之前）
                if episode_int_rewards:
                    # 转换为 numpy 数组
                    episode_int_rewards_np = np.array(episode_int_rewards)
                    
                    # 使用 forward filter 更新运行时统计
                    discounted_int_reward = self.int_reward_forward_filter.update(episode_int_rewards_np)
                    
                    # 更新运行时均值和标准差
                    mean_int = np.mean(discounted_int_reward)
                    std_int = np.std(discounted_int_reward)
                    self.int_reward_rms.update_from_moments(mean_int, std_int ** 2, len(discounted_int_reward))
                    
                    # 获取当前 episode 的索引
                    current_episode_idx = self.replay_buffer.episode_num
                    
                    # 归一化已存储的内在奖励
                    for step in range(len(episode_int_rewards)):
                        # 使用运行时标准差归一化
                        self.replay_buffer.buffer['r_int'][current_episode_idx, step, :] /= np.sqrt(self.int_reward_rms.var + 1e-8)
                    
                    # 统计已归一化的奖励
                    episode_intrinsic_reward_normalized = self.replay_buffer.buffer['r_int'][current_episode_idx, :len(episode_int_rewards), :].sum()
                    
                    # TensorBoard 记录
                    if self.writer:
                        self.writer.add_scalar('Reward/Intrinsic_Raw', episode_intrinsic_reward_raw, self.total_steps)
                        self.writer.add_scalar('Reward/Intrinsic_Normalized', episode_intrinsic_reward_normalized, self.total_steps)
                        self.writer.add_scalar('Reward/Intrinsic_Std', np.sqrt(self.int_reward_rms.var), self.total_steps)
                
                # 获取双价值估计并存储
                v_ext_n, v_int_n = self.agent_n.get_value(s)
                self.replay_buffer.store_last_value(episode_step + 1, None, v_ext_n, v_int_n)
            else:
                v_n, _ = self.agent_n.get_value(s)
                self.replay_buffer.store_last_value(episode_step + 1, v_n)

        return episode_reward, episode_adversary_reward, episode_good_reward, individual_adversary_rewards, episode_step + 1

    def save_rewards_to_csv(self):
        """保存详细的分类评估奖励数据到CSV文件"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # 保存详细的分类奖励数据
        filename = os.path.join(data_dir, f"happo_mactn_rewards_{self.env_name}_n{self.number}_s{self.seed}_{self.timestamp}.csv")
        
        steps = [i * self.args.evaluate_freq for i in range(len(self.evaluate_rewards))]
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 创建表头
            header = ['Steps', 'Total_Reward', 'Adversary_Total', 'Adversary_Avg', 'Good_Reward']
            # 为每个追捕者添加单独的列
            for agent_id in self.adversary_agents:
                header.append(f'{agent_id}_Reward')
            
            writer.writerow(header)
            
            # 写入数据
            for i, step in enumerate(steps):
                row = [
                    step,
                    self.evaluate_rewards[i],
                    self.evaluate_adversary_rewards[i],
                    self.evaluate_adversary_avg_rewards[i],
                    self.evaluate_good_rewards[i]
                ]
                # 添加每个追捕者的奖励
                for agent_id in self.adversary_agents:
                    row.append(self.evaluate_individual_adversary_rewards[agent_id][i])
                
                writer.writerow(row)
        
        print(f"详细评估奖励数据已保存到 {filename}")
        print(f"保存的数据包含: 总奖励、追捕者总奖励、追捕者平均奖励、逃跑者奖励，以及{len(self.adversary_agents)}个追捕者的单独奖励")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for HAPPO with MACTN in MPE environment")
    parser.add_argument("--device", type=str, default='cpu', help="training device")
    parser.add_argument("--max_train_steps", type=int, default=int(5e6), help="Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=1000, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64, help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="PPO update epochs")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")

    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--use_manual_reward_scale", type=bool, default=False, help="Use manual reward scaling with fixed factor")
    parser.add_argument("--reward_scale_factor", type=float, default=1/6.3, help="Manual reward scaling factor")

    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False, help="Whether to add agent_id")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip")
    parser.add_argument("--act_dim", type=float, default=5, help="Act_dimension")
    parser.add_argument("--number", type=int, default=1, help="Experiment number")
    parser.add_argument("--use_tensorboard", type=bool, default=True, help="Whether to use TensorBoard for logging")

    # MACTN-specific arguments
    parser.add_argument("--use_drnd", type=bool, default=True, help="Whether to use MACTN exploration")
    parser.add_argument("--drnd_alpha", type=float, default=0.9, help="DRND alpha")
    parser.add_argument("--drnd_lr", type=float, default=5e-4, help="Learning rate for MACTN predictor")
    parser.add_argument("--drnd_update_proportion", type=float, default=0.25, help="Proportion of samples for MACTN update")
    parser.add_argument("--int_coef", type=float, default=0.01, help="Coefficient for intrinsic reward")
    parser.add_argument("--int_gamma", type=float, default=0.99, help="Discount factor for intrinsic rewards (non-episodic)")

    args = parser.parse_args()
    runner = Runner_MAPPO_MPE(args, num_good=1, num_adversaries=3, num_obstacles=0, seed=23, env_name="simple_tag_v3")
    runner.run()