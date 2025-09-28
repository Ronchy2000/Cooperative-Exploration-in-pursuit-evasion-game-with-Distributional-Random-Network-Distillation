import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from HAPPO_DRND import HAPPO_MPE_with_DRND

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

class Runner_HAPPO_DRND_MPE:
    """⭐ 集成DRND探索的HAPPO训练器"""
    
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
        self.args.obs_dim = self.max_obs_dim
        self.args.action_dim = self.args.action_dim_n[0]
        self.args.act_dim = self.args.action_dim
        
        # 计算全局状态维度（所有智能体观察的总和）
        self.args.state_dim = np.sum(self.args.obs_dim_n)
        
        print("observation_space=", self.env.observation_space)
        print("obs_dim_n={}".format(self.args.obs_dim_n))
        print("max_obs_dim={}".format(self.max_obs_dim))
        print("action_space=", self.env.action_space)
        print("action_dim_n={}".format(self.args.action_dim_n))
        print(f"⭐ 全局状态维度: {self.args.state_dim}")

        # ⭐ Create agent with DRND
        self.agent_n = HAPPO_MPE_with_DRND(self.args)
        self.agent_n.env = self.env
        self.agent_n.all_agents = [agent_id for agent_id in self.env.agents]
        self.agent_n.dim_info = self.dim_info
        
        # 为环境添加get_obs_dims方法
        def get_obs_dims():
            return {agent_id: self.env.observation_space(agent_id).shape[0] for agent_id in self.env.agents}
        self.env.get_obs_dims = get_obs_dims
        
        self.replay_buffer = ReplayBuffer(self.args)

        # 奖励记录
        self.evaluate_rewards = []
        self.evaluate_adversary_rewards = []
        self.evaluate_adversary_avg_rewards = []
        self.evaluate_good_rewards = []
        self.evaluate_individual_adversary_rewards = {agent_id: [] for agent_id in self.adversary_agents}
        
        # ⭐ DRND相关记录
        self.intrinsic_reward_history = []
        self.exploration_coverage = []  # 探索覆盖率记录
        
        self.total_steps = 0
        
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)
        
        # ⭐ 初始化tensorboard（可选）
        if getattr(args, 'use_tensorboard', False):
            log_dir = f"runs/HAPPO_DRND_{env_name}_{self.timestamp}"
            self.writer = SummaryWriter(log_dir)
            print(f"Tensorboard日志目录: {log_dir}")
        else:
            self.writer = None

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
        """⭐ 获取全局状态（保持原始维度）- 用于DRND和Critic网络"""
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
        """主训练循环"""
        evaluate_num = -1
        while self.total_steps < self.args.max_train_steps:
            if self.total_steps // self.args.evaluate_freq > evaluate_num:
                self.evaluate_policy()
                evaluate_num += 1

            _, _, _, _, episode_steps = self.run_episode_mpe(evaluate=False)
            self.total_steps += episode_steps

            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)
                self.replay_buffer.reset_buffer()

        self.evaluate_policy()
        self.save_rewards_to_csv()
        if self.writer:
            self.writer.close()
        self.env.close()

    def evaluate_policy(self):
        """评估策略性能"""
        evaluate_reward = 0
        evaluate_adversary_reward = 0
        evaluate_good_reward = 0
        individual_adversary_rewards = {agent_id: 0 for agent_id in self.adversary_agents}
        
        # ⭐ DRND评估指标
        total_intrinsic_reward = 0
        exploration_states = set()  # 用于估计探索覆盖率
        
        for _ in range(self.args.evaluate_times):
            episode_reward, adversary_reward, good_reward, individual_rewards, intrinsic_reward, visited_states = self.run_episode_mpe(evaluate=True, collect_exploration_metrics=True)
            evaluate_reward += episode_reward
            evaluate_adversary_reward += adversary_reward
            evaluate_good_reward += good_reward
            total_intrinsic_reward += intrinsic_reward
            exploration_states.update(visited_states)
            
            # 累加每个追捕者的奖励
            for agent_id in self.adversary_agents:
                individual_adversary_rewards[agent_id] += individual_rewards.get(agent_id, 0)

        evaluate_reward = evaluate_reward / self.args.evaluate_times
        evaluate_adversary_reward = evaluate_adversary_reward / self.args.evaluate_times
        evaluate_good_reward = evaluate_good_reward / self.args.evaluate_times
        evaluate_adversary_avg_reward = evaluate_adversary_reward / len(self.adversary_agents)
        avg_intrinsic_reward = total_intrinsic_reward / self.args.evaluate_times
        exploration_coverage = len(exploration_states)
        
        # 计算每个追捕者的平均奖励
        for agent_id in self.adversary_agents:
            individual_adversary_rewards[agent_id] = individual_adversary_rewards[agent_id] / self.args.evaluate_times
        
        # 保存奖励数据
        self.evaluate_rewards.append(evaluate_reward)
        self.evaluate_adversary_rewards.append(evaluate_adversary_reward)
        self.evaluate_adversary_avg_rewards.append(evaluate_adversary_avg_reward)
        self.evaluate_good_rewards.append(evaluate_good_reward)
        
        # ⭐ 保存DRND指标
        self.intrinsic_reward_history.append(avg_intrinsic_reward)
        self.exploration_coverage.append(exploration_coverage)

        # 保存每个追捕者的奖励
        for agent_id in self.adversary_agents:
            self.evaluate_individual_adversary_rewards[agent_id].append(individual_adversary_rewards[agent_id])
        
        print("total_steps:{} \t total_reward:{:.2f} \t adversary_total:{:.2f} \t adversary_avg:{:.2f} \t good_reward:{:.2f}".format(
            self.total_steps, evaluate_reward, evaluate_adversary_reward, evaluate_adversary_avg_reward, evaluate_good_reward))
        print(f"⭐ DRND指标 - 内在奖励:{avg_intrinsic_reward:.4f} \t 探索覆盖:{exploration_coverage}")

        # 打印每个追捕者的奖励
        for agent_id in self.adversary_agents:
            print(f"  {agent_id}: {individual_adversary_rewards[agent_id]:.2f}")
        
        # ⭐ Tensorboard记录
        if self.writer:
            self.writer.add_scalar('Eval/Total_Reward', evaluate_reward, self.total_steps)
            self.writer.add_scalar('Eval/Adversary_Reward', evaluate_adversary_reward, self.total_steps)
            self.writer.add_scalar('Eval/Good_Reward', evaluate_good_reward, self.total_steps)
            self.writer.add_scalar('DRND/Intrinsic_Reward', avg_intrinsic_reward, self.total_steps)
            self.writer.add_scalar('DRND/Exploration_Coverage', exploration_coverage, self.total_steps)
        
        # Save the model
        self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps, self.timestamp)

    def run_episode_mpe(self, evaluate=False, collect_exploration_metrics=False):
        """运行一个episode"""
        episode_reward = 0
        episode_adversary_reward = 0
        episode_good_reward = 0
        individual_adversary_rewards = {agent_id: 0 for agent_id in self.adversary_agents}
        
        # ⭐ DRND指标
        episode_intrinsic_reward = 0
        visited_states = set()

        obs_dict, _ = self.env.reset()
        done_dict = {agent_id: False for agent_id in self.agent_n.all_agents}
        
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()
        if self.args.use_rnn:
            self.agent_n.reset_rnn_hidden()
            
        for episode_step in range(self.args.episode_limit):
            # 选择动作
            actions_dict, logprobs_dict = self.agent_n.choose_action(obs_dict, evaluate=evaluate)
            
            # 将观察转换为填充后的数组（所有智能体使用相同维度）
            obs_n = self._pad_obs_to_max_dim(obs_dict)
            
            # ⭐ 获取全局状态（用于DRND）
            s = self._get_global_state(obs_dict)
            v_n = self.agent_n.get_value(s)
            
            # ⭐ 计算内在奖励（如果启用DRND且不在评估模式）
            step_intrinsic_reward = 0
            if not evaluate and self.agent_n.use_drnd and self.agent_n._drnd_initialized:
                intrinsic_reward_tensor = self.agent_n.drnd.compute_intrinsic_reward(s)
                step_intrinsic_reward = intrinsic_reward_tensor.item()
                episode_intrinsic_reward += step_intrinsic_reward
            
            # ⭐ 收集探索指标（评估时）
            if collect_exploration_metrics:
                # 将状态转换为可哈希的元组（用于集合）
                state_hash = tuple(np.round(s, 2))  # 四舍五入以减少状态空间
                visited_states.add(state_hash)
            
            # 环境步进
            next_obs_dict, rewards_dict, terminated_dict, truncated_dict, _ = self.env.step(actions_dict)
            
            # 分别计算不同类型智能体的奖励
            step_adversary_reward = sum([rewards_dict.get(agent_id, 0) for agent_id in self.adversary_agents])
            step_good_reward = sum([rewards_dict.get(agent_id, 0) for agent_id in self.good_agents])
            step_total_reward = step_adversary_reward + step_good_reward
            
            # 为每个追捕者单独累加奖励
            for agent_id in self.adversary_agents:
                individual_adversary_rewards[agent_id] += rewards_dict.get(agent_id, 0)
            
            episode_reward += step_total_reward
            episode_adversary_reward += step_adversary_reward
            episode_good_reward += step_good_reward

            # 更新done标志
            for agent_id in terminated_dict:
                if terminated_dict[agent_id] or truncated_dict[agent_id]:
                    done_dict[agent_id] = True
                    
            done = all(done_dict.values()) or len(self.env.agents) == 0
            done_n = np.array([done] * self.args.N)

            if not evaluate:
                # 使用字典存储动作而不是numpy数组
                a_dict = {}
                r_n = np.zeros(self.args.N)
                
                # 填充实际值
                for i, agent_id in enumerate(self.agent_n.all_agents):
                    if agent_id in actions_dict:
                        a_dict[agent_id] = actions_dict[agent_id]
                    else:
                        # 为缺失的智能体创建零动作
                        agent_action_dim = self.dim_info[agent_id][1]
                        a_dict[agent_id] = np.zeros(agent_action_dim)
                        
                    if agent_id in rewards_dict:
                        r_n[i] = rewards_dict[agent_id]
                        
                # 处理logprobs
                if logprobs_dict is not None:
                    a_logprob_n = np.zeros(self.args.N)
                    for i, agent_id in enumerate(self.agent_n.all_agents):
                        if agent_id in logprobs_dict:
                            a_logprob_n[i] = logprobs_dict[agent_id]
                else:
                    a_logprob_n = None
                    
                # 应用奖励归一化
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                elif self.args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # 存储转换 - 使用字典形式的动作
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_dict, a_logprob_n, r_n, done_n)

            obs_dict = next_obs_dict
            if done:
                break

        if not evaluate:
            # 存储最后的值
            obs_n = self._pad_obs_to_max_dim(obs_dict)
            s = self._get_global_state(obs_dict)
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        if collect_exploration_metrics:
            return episode_reward, episode_adversary_reward, episode_good_reward, individual_adversary_rewards, episode_intrinsic_reward, visited_states
        else:
            return episode_reward, episode_adversary_reward, episode_good_reward, individual_adversary_rewards, episode_step + 1

    def save_rewards_to_csv(self):
        """⭐ 保存详细的分类评估奖励数据到CSV文件（包括DRND指标）"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # 保存详细的分类奖励数据
        filename = os.path.join(data_dir, f"happo_drnd_detailed_rewards_{self.env_name}_n{self.number}_s{self.seed}_{self.timestamp}.csv")
        
        steps = [i * self.args.evaluate_freq for i in range(len(self.evaluate_rewards))]
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # 创建表头
            header = ['Steps', 'Total_Reward', 'Adversary_Total', 'Adversary_Avg', 'Good_Reward', 'Intrinsic_Reward', 'Exploration_Coverage']
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
                    self.evaluate_good_rewards[i],
                    self.intrinsic_reward_history[i] if i < len(self.intrinsic_reward_history) else 0,
                    self.exploration_coverage[i] if i < len(self.exploration_coverage) else 0
                ]
                # 添加每个追捕者的奖励
                for agent_id in self.adversary_agents:
                    row.append(self.evaluate_individual_adversary_rewards[agent_id][i])
                
                writer.writerow(row)
        
        print(f"⭐ HAPPO+DRND详细评估奖励数据已保存到 {filename}")
        print(f"保存的数据包含: 总奖励、追捕者总奖励、追捕者平均奖励、逃跑者奖励、内在奖励、探索覆盖率，以及{len(self.adversary_agents)}个追捕者的单独奖励")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for HAPPO+DRND in MPE environment")
    parser.add_argument("--device", type=str, default='cpu', help="training device")
    parser.add_argument("--max_train_steps", type=int, default=int(1e5), help="Maximum number of training steps")
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
    
    # ⭐ DRND参数
    parser.add_argument("--use_drnd", type=bool, default=True, help="Whether to use DRND exploration")
    parser.add_argument("--drnd_output_dim", type=int, default=64, help="DRND target network output dimension")
    parser.add_argument("--drnd_hidden_dim", type=int, default=128, help="DRND network hidden dimension")
    parser.add_argument("--drnd_lr", type=float, default=3e-4, help="DRND predictor learning rate")
    parser.add_argument("--drnd_alpha", type=float, default=0.9, help="DRND dual-phase exploration weight")
    parser.add_argument("--intrinsic_reward_coeff", type=float, default=1.0, help="Intrinsic reward coefficient")
    parser.add_argument("--drnd_update_freq", type=int, default=1, help="DRND predictor update frequency")
    parser.add_argument("--use_tensorboard", type=bool, default=False, help="Whether to use tensorboard logging")

    args = parser.parse_args()
    
    print("⭐ 开始HAPPO+DRND训练")
    print(f"DRND参数:")
    print(f"  - 使用DRND: {args.use_drnd}")
    print(f"  - 靶网络输出维度: {args.drnd_output_dim}")
    print(f"  - 隐藏层维度: {args.drnd_hidden_dim}")
    print(f"  - 学习率: {args.drnd_lr}")
    print(f"  - 双阶段权重α: {args.drnd_alpha}")
    print(f"  - 内在奖励系数: {args.intrinsic_reward_coeff}")
    
    runner = Runner_HAPPO_DRND_MPE(args, num_good=1, num_adversaries=3, num_obstacles=0, seed=23, env_name="simple_tag_v3")
    runner.run()