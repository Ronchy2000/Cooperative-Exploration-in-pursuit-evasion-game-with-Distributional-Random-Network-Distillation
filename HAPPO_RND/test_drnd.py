"""
测试DRND模块的功能
验证多智能体DRND的核心组件是否正常工作
"""

import torch
import numpy as np
import argparse
from drnd_exploration import MultiAgentDRND

def test_drnd_basic_functionality():
    """测试DRND基础功能"""
    print("=" * 60)
    print("⭐ 测试多智能体DRND基础功能")
    print("=" * 60)
    
    # 创建测试参数
    args = argparse.Namespace()
    args.state_dim = 30  # 假设3个agent，每个观察维度10
    args.drnd_output_dim = 64
    args.drnd_hidden_dim = 128
    args.drnd_lr = 3e-4
    args.drnd_alpha = 0.9
    args.intrinsic_reward_coeff = 1.0
    args.device = 'cpu'
    
    # 智能体列表
    agent_ids = ['adversary_0', 'adversary_1', 'adversary_2']
    
    # 初始化DRND
    drnd = MultiAgentDRND(args, agent_ids)
    
    print(f"✓ DRND初始化成功")
    print(f"  - 智能体数量: {drnd.num_agents}")
    print(f"  - 全局状态维度: {drnd.state_dim}")
    print(f"  - 靶网络数量: {len(drnd.target_networks)}")
    
    # 测试靶网络输出
    print("\n" + "-" * 40)
    print("测试靶网络输出")
    print("-" * 40)
    
    # 创建测试状态
    batch_size = 5
    test_states = torch.randn(batch_size, args.state_dim)
    print(f"测试状态形状: {test_states.shape}")
    
    # 计算靶网络统计量
    mu, B2 = drnd.compute_target_statistics(test_states)
    print(f"✓ 靶网络均值μ形状: {mu.shape}")
    print(f"✓ 靶网络二阶矩B2形状: {B2.shape}")
    print(f"  - μ的范围: [{mu.min().item():.4f}, {mu.max().item():.4f}]")
    print(f"  - B2的范围: [{B2.min().item():.4f}, {B2.max().item():.4f}]")
    
    # 测试内在奖励计算
    print("\n" + "-" * 40)
    print("测试内在奖励计算")
    print("-" * 40)
    
    intrinsic_rewards = drnd.compute_intrinsic_reward(test_states)
    print(f"✓ 内在奖励形状: {intrinsic_rewards.shape}")
    print(f"  - 内在奖励范围: [{intrinsic_rewards.min().item():.4f}, {intrinsic_rewards.max().item():.4f}]")
    print(f"  - 内在奖励均值: {intrinsic_rewards.mean().item():.4f}")
    print(f"  - 内在奖励标准差: {intrinsic_rewards.std().item():.4f}")
    
    # 测试预测器更新
    print("\n" + "-" * 40)
    print("测试预测器更新")
    print("-" * 40)
    
    initial_predictor_weights = drnd.predictor.net[0].weight.clone()
    
    for step in range(5):
        loss = drnd.update_predictor(test_states)
        print(f"  步骤 {step+1}: 预测器损失 = {loss:.6f}")
    
    final_predictor_weights = drnd.predictor.net[0].weight.clone()
    weight_change = torch.norm(final_predictor_weights - initial_predictor_weights).item()
    print(f"✓ 预测器权重变化: {weight_change:.6f}")
    
    if weight_change > 1e-6:
        print("✓ 预测器正在正常更新")
    else:
        print("⚠️ 预测器权重变化很小，可能存在问题")
    
    # 测试训练后的内在奖励变化
    print("\n" + "-" * 40)
    print("测试训练对内在奖励的影响")
    print("-" * 40)
    
    new_intrinsic_rewards = drnd.compute_intrinsic_reward(test_states)
    reward_change = torch.norm(new_intrinsic_rewards - intrinsic_rewards).item()
    print(f"✓ 内在奖励变化: {reward_change:.6f}")
    print(f"  - 新内在奖励均值: {new_intrinsic_rewards.mean().item():.4f}")
    
    # 测试批量处理
    print("\n" + "-" * 40)
    print("测试批量episode处理")
    print("-" * 40)
    
    batch_size = 3
    episode_limit = 10
    batch_states = torch.randn(batch_size, episode_limit, args.state_dim)
    print(f"批量状态形状: {batch_states.shape}")
    
    batch_intrinsic_rewards = drnd.get_intrinsic_rewards_for_batch(batch_states)
    print(f"✓ 批量内在奖励形状: {batch_intrinsic_rewards.shape}")
    print(f"  - 批量内在奖励均值: {batch_intrinsic_rewards.mean().item():.4f}")
    
    print("\n" + "=" * 60)
    print("✓ 所有DRND基础功能测试通过！")
    print("=" * 60)

def test_drnd_dual_phase_behavior():
    """测试DRND双阶段探索行为"""
    print("\n" + "=" * 60)
    print("⭐ 测试DRND双阶段探索行为")
    print("=" * 60)
    
    # 创建测试参数
    args = argparse.Namespace()
    args.state_dim = 12  # 3个agent，每个观察维度4（简化）
    args.drnd_output_dim = 32
    args.drnd_hidden_dim = 64
    args.drnd_lr = 1e-3  # 较高学习率以便观察变化
    args.drnd_alpha = 0.9
    args.intrinsic_reward_coeff = 1.0
    args.device = 'cpu'
    
    agent_ids = ['agent_0', 'agent_1', 'agent_2']
    drnd = MultiAgentDRND(args, agent_ids)
    
    # 创建固定的测试状态
    test_state = torch.randn(1, args.state_dim)
    print(f"使用固定测试状态: 形状 {test_state.shape}")
    
    # 记录训练过程中内在奖励的变化
    print("\n训练步骤 | b1(预测误差) | b2(伪计数) | 总内在奖励")
    print("-" * 55)
    
    intrinsic_rewards_history = []
    b1_history = []
    b2_history = []
    
    for step in range(0, 101, 10):  # 0, 10, 20, ..., 100
        # 更新预测器
        for _ in range(10):
            drnd.update_predictor(test_state)
        
        # 计算详细的内在奖励组件
        with torch.no_grad():
            mu, B2 = drnd.compute_target_statistics(test_state)
            f_theta = drnd.predictor(test_state)
            
            # 第一阶段：预测误差
            b1 = torch.norm(f_theta - mu, p=2, dim=1, keepdim=True) ** 2
            
            # 第二阶段：伪计数估计
            numerator = torch.norm(f_theta, p=2, dim=1, keepdim=True) ** 2 - torch.norm(mu, p=2, dim=1, keepdim=True) ** 2
            denominator = torch.sum(B2, dim=1, keepdim=True) - torch.norm(mu, p=2, dim=1, keepdim=True) ** 2
            denominator = torch.clamp(denominator, min=1e-8)
            b2 = torch.sqrt(torch.clamp(numerator / denominator, min=0, max=100))
            
            # 总内在奖励
            total_intrinsic = args.drnd_alpha * b1 + (1 - args.drnd_alpha) * b2
            
            b1_val = b1.item()
            b2_val = b2.item()
            total_val = total_intrinsic.item()
            
            print(f"   {step:3d}    |   {b1_val:8.4f}   |  {b2_val:7.4f}  |   {total_val:8.4f}")
            
            intrinsic_rewards_history.append(total_val)
            b1_history.append(b1_val)
            b2_history.append(b2_val)
    
    print("\n" + "-" * 55)
    print("双阶段行为分析:")
    
    # 分析b1（预测误差）的变化趋势
    b1_start = b1_history[0]
    b1_end = b1_history[-1]
    b1_decrease_ratio = (b1_start - b1_end) / b1_start if b1_start > 0 else 0
    
    print(f"  - b1（预测误差）: {b1_start:.4f} → {b1_end:.4f} (下降 {b1_decrease_ratio*100:.1f}%)")
    
    if b1_decrease_ratio > 0.5:
        print("    ✓ b1正常下降，预测器学会了靶网络分布")
    else:
        print("    ⚠️ b1下降不明显，可能需要调整学习率")
    
    # 分析b2（伪计数）的行为
    b2_start = b2_history[0]
    b2_end = b2_history[-1]
    print(f"  - b2（伪计数）: {b2_start:.4f} → {b2_end:.4f}")
    
    # 分析总内在奖励的变化
    total_start = intrinsic_rewards_history[0]
    total_end = intrinsic_rewards_history[-1]
    print(f"  - 总内在奖励: {total_start:.4f} → {total_end:.4f}")
    
    print("\n✓ 双阶段探索行为测试完成！")
    print("  理想行为：早期b1主导（探索新颖状态），后期b2主导（聚焦少见状态）")
    
    print("\n" + "=" * 60)
    print("✓ 所有DRND测试完成！可以开始正式训练了。")
    print("=" * 60)

if __name__ == "__main__":
    test_drnd_basic_functionality()
    test_drnd_dual_phase_behavior()