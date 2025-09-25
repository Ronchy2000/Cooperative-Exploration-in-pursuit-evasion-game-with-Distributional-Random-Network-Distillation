import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import argparse
import re

def plot_rewards(file_path=None, data_dir=None, show=True, save=True):
    """
    Plot training reward curves for detailed reward data
    
    Parameters:
    file_path: Specified CSV file path. If None, use the latest CSV file
    data_dir: Directory containing CSV files
    show: Whether to display the chart
    save: Whether to save the chart as PNG file
    """
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # If data directory not specified, use default data subdirectory
    if data_dir is None:
        data_dir = os.path.join(current_dir)
    elif not os.path.isabs(data_dir):
        # If a relative path is provided, convert to absolute path
        data_dir = os.path.join(current_dir, data_dir)
    
    # Confirm directory exists
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # If file not specified, find the latest CSV file
    if file_path is None:
        # 支持自动查找最新的happo详细奖励文件
        csv_files = glob.glob(os.path.join(data_dir, "happo_detailed_rewards_*.csv"))
        if not csv_files:
            print(f"Error: No HAPPO detailed reward CSV files found in directory {data_dir}")
            return
        file_path = max(csv_files, key=os.path.getctime)  # Select the latest file
    elif not os.path.isabs(file_path):
        # If a relative path is provided, convert to absolute path
        file_path = os.path.join(current_dir, file_path)
    
    print(f"Loading data file: {file_path}")
    
    # Read CSV data
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return
    
    # Extract filename information
    filename = os.path.basename(file_path)
    parts = filename.split('_')
    
    algorithm = "HAPPO"
    
    # 从文件名中提取环境名和种子值
    env_match = re.search(r"happo_detailed_rewards_(.+?)_n\d+_s(\d+)_", filename)
    if env_match:
        env_name = env_match.group(1)
        seed = env_match.group(2)
    else:
        env_name = "unknown_env"
        seed = "unknown"
    
    # 检查数据列是否存在
    required_columns = ['Steps', 'Total_Reward', 'Adversary_Total', 'Adversary_Avg', 'Good_Reward']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        print(f"Error: Missing columns in CSV: {missing_columns}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Create subplots for better visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Total Reward
    ax1.plot(df['Steps'], df['Total_Reward'], marker='o', linestyle='-', markersize=3, linewidth=2, alpha=0.8, color='blue')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Total Reward (All Agents)')
    ax1.grid(True, linestyle='--', alpha=0.3)
    
    # Plot 2: Adversary vs Good Rewards
    ax2.plot(df['Steps'], df['Adversary_Total'], marker='s', linestyle='-', markersize=3, linewidth=2, alpha=0.8, color='red', label='Adversary Total')
    ax2.plot(df['Steps'], df['Good_Reward'], marker='^', linestyle='-', markersize=3, linewidth=2, alpha=0.8, color='green', label='Good Agent')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Reward')
    ax2.set_title('Adversary vs Good Agent Rewards')
    ax2.grid(True, linestyle='--', alpha=0.3)
    ax2.legend()
    
    # Plot 3: Adversary Total vs Average
    ax3.plot(df['Steps'], df['Adversary_Total'], marker='s', linestyle='-', markersize=3, linewidth=2, alpha=0.8, color='red', label='Adversary Total')
    ax3.plot(df['Steps'], df['Adversary_Avg'], marker='d', linestyle='-', markersize=3, linewidth=2, alpha=0.8, color='orange', label='Adversary Average')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('Reward')
    ax3.set_title('Adversary Total vs Average Reward')
    ax3.grid(True, linestyle='--', alpha=0.3)
    ax3.legend()
    
    # Plot 4: Individual Adversary Rewards (if available)
    individual_columns = [col for col in df.columns if col.startswith('adversary_') and col.endswith('_Reward')]
    if individual_columns:
        colors = ['purple', 'brown', 'pink', 'gray', 'olive']  # 预定义颜色
        for i, col in enumerate(individual_columns):
            color = colors[i % len(colors)]
            agent_name = col.replace('_Reward', '')
            ax4.plot(df['Steps'], df[col], marker='o', linestyle='-', markersize=3, 
                    linewidth=2, alpha=0.8, color=color, label=agent_name)
        ax4.set_xlabel('Training Steps')
        ax4.set_ylabel('Reward')
        ax4.set_title('Individual Adversary Rewards')
        ax4.grid(True, linestyle='--', alpha=0.3)
        ax4.legend()
    else:
        # 如果没有个体奖励数据，显示总体统计
        ax4.text(0.5, 0.5, 'No individual adversary\nreward data available', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Individual Adversary Analysis')
    
    # Add overall title
    fig.suptitle(f'{algorithm} Training Analysis | Env: {env_name} | Seed: {seed}', 
                fontsize=16, fontweight='bold')
    
    # Add performance statistics
    if len(df) > 1:
        # 计算统计信息
        total_final = df['Total_Reward'].tail(5).mean()
        adversary_final = df['Adversary_Total'].tail(5).mean()
        good_final = df['Good_Reward'].tail(5).mean()
        
        stats_text = (f'Final Performance (last 5 episodes avg):\n'
                     f'Total: {total_final:.2f}\n'
                     f'Adversary: {adversary_final:.2f}\n'
                     f'Good: {good_final:.2f}')
        
        fig.text(0.02, 0.02, stats_text, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    
    # Save chart
    if save:
        plots_dir = os.path.join(current_dir)
        os.makedirs(plots_dir, exist_ok=True)
        plt_filename = os.path.join(plots_dir, f"{algorithm.lower()}_detailed_analysis_{env_name}_s{seed}.png")
        plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {plt_filename}")
    
    # Display chart
    if show:
        plt.show()

def plot_simple_reward(file_path=None, data_dir=None, show=True, save=True, reward_column='Total_Reward'):
    """
    Plot a simple single reward curve
    
    Parameters:
    file_path: Specified CSV file path
    data_dir: Directory containing CSV files
    show: Whether to display the chart
    save: Whether to save the chart
    reward_column: Which reward column to plot
    """
    # Get the absolute path of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if data_dir is None:
        data_dir = os.path.join(current_dir)
    elif not os.path.isabs(data_dir):
        data_dir = os.path.join(current_dir, data_dir)
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    if file_path is None:
        csv_files = glob.glob(os.path.join(data_dir, "happo_detailed_rewards_*.csv"))
        if not csv_files:
            print(f"Error: No HAPPO detailed reward CSV files found in directory {data_dir}")
            return
        file_path = max(csv_files, key=os.path.getctime)
    elif not os.path.isabs(file_path):
        file_path = os.path.join(current_dir, file_path)
    
    print(f"Loading data file: {file_path}")
    
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return
    
    if reward_column not in df.columns:
        print(f"Error: Column '{reward_column}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Extract filename information
    filename = os.path.basename(file_path)
    algorithm = "HAPPO"
    
    env_match = re.search(r"happo_detailed_rewards_(.+?)_n\d+_s(\d+)_", filename)
    if env_match:
        env_name = env_match.group(1)
        seed = env_match.group(2)
    else:
        env_name = "unknown_env"
        seed = "unknown"
    
    # Create simple plot
    plt.figure(figsize=(12, 8))
    plt.plot(df['Steps'], df[reward_column], marker='o', linestyle='-', markersize=3, linewidth=2, alpha=0.8)
    plt.xlabel('Training Steps', fontsize=12)
    plt.ylabel(f'{reward_column.replace("_", " ")}', fontsize=12)
    
    plt.title(f'{algorithm} Learning Curve | {reward_column.replace("_", " ")} | Env: {env_name} | Seed: {seed}', 
              fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    # Add statistics
    if len(df) > 1:
        avg_reward = df[reward_column].mean()
        plt.axhline(y=avg_reward, color='red', linestyle='--', alpha=0.6, linewidth=2,
                   label=f'Average: {avg_reward:.2f}')
        
        if len(df) >= 5:
            window_size = min(max(len(df) // 10, 3), 10)
            df['MA'] = df[reward_column].rolling(window=window_size, min_periods=1).mean()
            plt.plot(df['Steps'], df['MA'], color='green', linestyle='-', linewidth=2,
                    alpha=0.7, label=f'Moving Average ({window_size} points)')
        
        final_rewards = df[reward_column].tail(5).mean()
        plt.axhline(y=final_rewards, color='purple', linestyle=':', alpha=0.6, linewidth=2,
                   label=f'Final Performance: {final_rewards:.2f}')
    
    if len(df) > 1:
        max_reward = df[reward_column].max()
        min_reward = df[reward_column].min()
        std_reward = df[reward_column].std()
        
        stats_text = f'Max: {max_reward:.2f}\nMin: {min_reward:.2f}\nStd: {std_reward:.2f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='top', fontsize=10)
    
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()
    
    if save:
        plots_dir = os.path.join(current_dir)
        os.makedirs(plots_dir, exist_ok=True)
        plt_filename = os.path.join(plots_dir, f"{algorithm.lower()}_{reward_column.lower()}_{env_name}_s{seed}.png")
        plt.savefig(plt_filename, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {plt_filename}")
    
    if show:
        plt.show()

if __name__ == "__main__":
    # 在这里设置文件名和时间戳相关参数，方便修改
    DEFAULT_FILE_PATTERN = "happo_detailed_rewards_simple_tag_v3_n1_s23_2025-09-25_11-24.csv"  # 你可以修改这个模式
    DEFAULT_ENV_NAME = "simple_tag_v3"      # 你可以修改默认环境名
    DEFAULT_SEED = "23"                                   # 你可以修改默认种子
    DEFAULT_TIMESTAMP = "2025-09-25_11-24"                     # 你可以修改默认时间戳模式
    
    parser = argparse.ArgumentParser(description="Plot HAPPO detailed training reward curves")
    parser.add_argument("--file", type=str, default=None, 
                       help="CSV file path to plot. If not specified, use the latest CSV file")
    parser.add_argument("--data_dir", type=str, default=None,
                       help="Directory containing CSV data files")
    parser.add_argument("--no-show", action="store_false", dest="show",
                       help="Do not display the chart")
    parser.add_argument("--no-save", action="store_false", dest="save",
                       help="Do not save the chart")
    parser.add_argument("--mode", type=str, choices=["detailed", "simple"], default="detailed",
                       help="Plot mode: 'detailed' for multi-subplot analysis, 'simple' for single curve")
    parser.add_argument("--reward", type=str, default="Total_Reward",
                       choices=["Total_Reward", "Adversary_Total", "Adversary_Avg", "Good_Reward"],
                       help="Which reward to plot in simple mode")
    
    args = parser.parse_args()
    
    # 你可以在这里修改具体的文件查找逻辑
    if args.file is None and args.data_dir is not None:
        # 构建特定的文件模式，方便你自定义
        specific_pattern = f"happo_detailed_rewards_{DEFAULT_ENV_NAME}_n1_s{DEFAULT_SEED}_{DEFAULT_TIMESTAMP}.csv"
        print(f"Looking for files matching pattern: {specific_pattern}")
    
    if args.mode == "detailed":
        plot_rewards(args.file, args.data_dir, args.show, args.save)
    else:
        plot_simple_reward(args.file, args.data_dir, args.show, args.save, args.reward)