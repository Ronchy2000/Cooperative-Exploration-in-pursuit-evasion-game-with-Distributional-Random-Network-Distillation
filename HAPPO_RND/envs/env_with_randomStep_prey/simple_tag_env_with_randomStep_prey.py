# from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3

# from pettingzoo.mpe.simple_tag_v3 import raw_env

import numpy as np
import gymnasium
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .custom_agent_dynamics_with_randomStep_prey import CustomAgent, CustomWorld # Ronchy自定义的World类，用于测试自定义的智能体动力学模型

import pygame  #Ronchy: 用于渲染动画环境

# 用于检测包围条件
# 由于智能体出现共线，该方法会报错，遂移除，使用本地写的方法检测
# from scipy.spatial import ConvexHull, Delaunay

'''
继承 raw_env, 修改部分功能。
'''

class Custom_raw_env(SimpleEnv, EzPickle):
    def __init__(
        self,
        num_good=1,
        num_adversaries=3,
        num_obstacles=2,
        max_cycles=50,
        continuous_actions=False,
        render_mode=None,
        dynamic_rescaling=False,
        world_size = 2.5, # Ronchy添加 world_size参数 ,地图大小 world_size x world_size
    ):
        EzPickle.__init__(
            self,
            num_good=num_good,
            num_adversaries=num_adversaries,
            num_obstacles=num_obstacles,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            render_mode=render_mode,
        )
        scenario = Scenario()
        world = scenario.make_world(num_good, num_adversaries, num_obstacles, _world_size = world_size) # Ronchy添加 world_size参数
        SimpleEnv.__init__(
            self,
            scenario=scenario,
            world=world,
            render_mode=render_mode,
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
            dynamic_rescaling=dynamic_rescaling,
        )
        self.world_size = world_size  # Ronchy添加 world_size参数, 地图大小 world_size x world_size
        self.metadata["name"] = "simple_tag_v3"
        # Ronchy添加轨迹记录
        self.history_positions = {agent.name: [] for agent in world.agents}
        # self.max_history_length = 500  # 最大轨迹长度

        # 重载 simple_env.py中的代码
        pygame.font.init()
        self.game_font = pygame.font.SysFont('arial', 16)  # 使用系统字体

        self.max_force = 1.0  # 最大力
        self.capture_threshold = self.world_size * 0.2 # 围捕阈值: 使用世界大小的20%作为默认捕获范围
        self.local_ratio = 0.4  # 全局奖励和局部奖励的比例 local*个体 + (1-local)*全局共享
        # 重载continuous_actions空间
        # set spaces
        self.action_spaces = dict()
        self.observation_spaces = dict()
        state_dim = 0
        for agent in self.world.agents:  #每个智能体都有自己的观测空间和动作空间
            if agent.movable: 
                if self.continuous_actions == True:
                    space_dim = self.world.dim_p  # dim_p: default 2  -> position dimensionality  
                elif self.continuous_actions == False:
                    space_dim = self.world.dim_p * 2 + 1  # default: 5  # 1个维度表示静止，4个维度表示4个方向的运动，离散值
            else:
                space_dim = 1 # 1个维度表示静止
            # 通信动作
            if agent.silent == False:  #Scenario类中默认为True
                if self.continuous_actions:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c
            obs_dim = len(self.scenario.observation(agent, self.world))  # 观测维度， Scenario类中 observation()创建观测空间
            state_dim += obs_dim  # 所有智能体的观测空间累积，就是 状态空间维数

            if self.continuous_actions: # continuous actions
                self.action_spaces[agent.name] = gymnasium.spaces.Box(
                    low= -1.0, high=1.0, shape=(space_dim,), dtype=np.float32 # 限制在[-1,1]之间，这个是控制输入的限幅
                )
            else:  # discrete actions
                self.action_spaces[agent.name] = gymnasium.spaces.Discrete(space_dim)
            # 定义单个agent的观测空间
            self.observation_spaces[agent.name] = gymnasium.spaces.Box(
                low = -np.float32(np.inf), # 最低限制
                high = +np.float32(np.inf), # 最高限制
                shape = (obs_dim,),
                dtype = np.float32,
            )
        # 定义多智能体状态空间 公用1个状态空间。
        self.state_space = gymnasium.spaces.Box(
            low = -np.float32(np.inf),
            high = +np.float32(np.inf),
            shape = (state_dim,),
            dtype = np.float32,
        )


    def reset(self, seed=None, options=None):
        # 重置环境状态并清空轨迹记录
        super().reset(seed=seed, options=options)
        # 清空轨迹
        self.history_positions = {agent.name: [] for agent in self.world.agents}
        # 重置捕获标志
        self.world.is_captured = False
        self.world.is_surrounded = False
        self.world.episode_steps = 0  # 每个episode 重置步数计数器

    def reset_world(self, world, np_random):
        # 清除历史轨迹
        self.history_positions = {agent.name: [] for agent in self.world.agents}  
        world.is_captured = False
        world.is_surrounded = False
        world.episode_steps = 0  # 每个episode 重置步数计数器

        # 调用Scenario的reset_world方法
        super().scenario.reset_world(world, np_random)
    
    """
    rewrite `_execute_world_step` method in:
        simple_env <- class SimpleEnv()
    """
    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            action = self.current_actions[i]
            scenario_action = []
            mdim = self.world.dim_p if self.continuous_actions else self.world.dim_p * 2 + 1  # 连续 2，离散 5
            # print(f"_execute_world_step : mdim:{mdim}") # mdim: 2
            if agent.movable: # default: True  # 且为  policy_agents
                if self.continuous_actions:
                    scenario_action.append(action[0:mdim])  # phisical action  mdim: 2  ,此处的Scenario_action是二维列表了.[action[0:2], acrionp[2:]],[[物理动作]，[通信动作]]
                    action = action[mdim:] # communication action
                else:
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent: # default: True
                scenario_action.append(action)
            # set action for agent  action_spaces[agent.name]已经被划分成 scenario_action和action了，所以此处action_spaces[agent.name]不再使用
            self._set_action(scenario_action, agent, self.action_spaces[agent.name], time = None)

        self.world.step() #core.py  在world实例中 执行动力学

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    # set env action for a particular agent
    # self._set_action(scenario_action, agent, self.action_spaces[agent.name])
    # scenario_action 物理动作实参
    def _set_action(self, action, agent, action_space, time=None):
        """
        pettiongzoo中的 agent的动作 被分为 action.u 和 action.c 两部分,
        分别代表physical action和communication action。
        默认值：
        action维数为5, 第0位没有用
        第1,2位表示x方向加速和减速
        第3,4位表示y方向加速和减速
        """
        #此处是指agent.action = agent.action.u -> scenarios_action + agent.action.c -> communication_action
        agent.action.u = np.zeros(self.world.dim_p) # default:2  phisiacal action, 加速度的维数
        agent.action.c = np.zeros(self.world.dim_c) # default:2  communication action的维数
        if agent.movable:
            agent.action.u = np.zeros(self.world.dim_p) #  default:2
            if self.continuous_actions:
                # Process continuous action as in OpenAI MPE
                # Note: this ordering preserves the same movement direction as in the discrete case
                # print("_set_action: action",action)
                agent.action.u[0] = action[0][0] # Force in x direction
                agent.action.u[1] = action[0][1]  # Force in y direction
            else:
                # process discrete action
                if action[0] == 1:
                    agent.action.u[0] = -1.0
                if action[0] == 2:
                    agent.action.u[0] = +1.0
                if action[0] == 3:
                    agent.action.u[1] = -1.0
                if action[0] == 4:
                    agent.action.u[1] = +1.0
        
        # Ronchy 添加力限幅 
        agent.action.u = np.clip(agent.action.u, -self.max_force, self.max_force) #self.max_force = 1.0  # 根据需求调整最大力

        #     # Ronchy 修改加速度逻辑
        #     sensitivity = 1.0  # default: 5.0
        #     if agent.accel is not None:
        #         sensitivity = agent.accel
        #     agent.action.u *= sensitivity
        #     action = action[1:]

        # if not agent.silent:  # 默认为True，这里被跳过。
        #     # communication action
        #     if self.continuous_actions:
        #         agent.action.c = action[0]
        #     else:
        #         agent.action.c = np.zeros(self.world.dim_c)
        #         agent.action.c[action[0]] = 1.0
        #     action = action[1:]
        # make sure we used all elements of action
        # assert len(action) == 0 # Ronchy: 保证action被完全使用。如果 action 数组不为空，说明有动作维度没有被正确处理，程序会抛出 AssertionError

    """
    rewrite step method in: 
        simple_env <- class SimpleEnv()

        simple_tag_env.step(action)   # 在算法上直接调用env.step(action)即可
            -> _execute_world_step()  
            -> _set_action(action) # 把合外力变成加速度。（原版是 乘以 sensitivity; 即agent.accel）
            -> world.step()  # 调用 core.py 中的 World.step()  # 实现agent 的动力学
        -simple_tag_env.step() ：
            - 环境层面的步进
            - 处理动作的接收和预处理
            - 管理奖励、状态转换等高层逻辑
            - 处理轨迹记录、终止条件等
        - core.py 中的 World.step() ：  # 需要在Scenario类中重载
            - 物理引擎层面的步进
            - 实现具体的动力学计算
            - 处理力的应用和状态积分
            - 更新物理状态（位置、速度等）
    """ 
    def step(self, action):   # 环境层面的步进
        # print("Using rewrited step method.")
        #  如果有任何智能体的 terminated 状态为 True，它们将从 self.env.agents 中移除
        if ( 
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()

            # Ronchy记录轨迹
            for agent in self.world.agents:
                self.history_positions[agent.name].append(agent.state.p_pos.copy())
                # if len(self.history_positions[agent.name]) > self.max_history_length: # 限制轨迹长度
                #     self.history_positions[agent.name].pop(0)

            self.steps += 1
            # 检查围捕条件
            is_captured, capture_info = self.check_capture_condition(threshold=self.capture_threshold)  #围捕标志——半径
            # 设置围捕状态以供全局奖励函数使用
            self.world.is_captured = is_captured
             # 如果被围捕，设置终止状态
            """
            主要问题是在 step 方法中使用了 self.agents 而不是 self.possible_agents。
            在 PettingZoo 中， self.agents 是当前活跃的智能体列表，而 self.possible_agents 包含所有可能的智能体，包括已经终止的智能体。
            """
            if is_captured:
                print("Captured!")
                for a in self.possible_agents:
                    self.terminations[a] = True
            # 如果达到最大步数，标记 truncation 为 True
            if self.steps >= self.max_cycles:
                for a in self.possible_agents: 
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
    """
    rewrite step method in: 
        simple_env <- class SimpleEnv()
    """ 
    # Ronchy: 重载render函数
    def enable_render(self, mode="human"):
        if not self.renderOn and mode == "human":
            self.screen = pygame.display.set_mode(self.screen.get_size())
            self.clock = pygame.time.Clock()
            self.renderOn = True
    
    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)

        # 添加事件处理, 解决windows渲染报错
        """
        Mac 上不需要特别处理事件是因为 macOS 的窗口管理系统和事件处理机制与 Windows 不同。Mac 系统有更好的窗口管理机制，即使不处理事件队列也不会导致程序无响应。
         这样的设计可以：
                1. 在 Windows 上避免无响应
                2. 在 Mac 上也能正常工作
                3. 提供更好的用户体验（比如正确响应窗口关闭按钮）
                所以建议保留这段事件处理代码，这样你的程序在任何平台上都能正常工作。
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type == pygame.WINDOWCLOSE:
                pygame.quit()
                return
        pygame.event.pump()  # 确保事件系统正常运行
        #--------
        self.draw()

         # Ronchy添加
        """
        根据调试结果，我们已经确认了主要问题：
            在将浮点数转换为整数像素坐标时产生的舍入误差导致了视觉上的间隙。
        """
        # self.debug_collision_rendering()

        if self.render_mode == "rgb_array":
            observation = np.array(pygame.surfarray.pixels3d(self.screen))
            return np.transpose(observation, axes=(1, 0, 2))
        elif self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return
    
    def draw(self):
        # 清空画布
        self.screen.fill((255, 255, 255))
        # 计算动态缩放
        all_poses = [entity.state.p_pos for entity in self.world.entities]
    #     cam_range = np.max(np.abs(np.array(all_poses)))
        cam_range = self.world_size  # 使用环境实际大小
        scaling_factor = 0.7 * self.original_cam_range / cam_range
        # 绘制坐标轴
        self.draw_grid_and_axes()
        # 在逃跑者位置绘制capture_threshold 圆圈
        for agent in self.scenario.good_agents(self.world):
            x, y = agent.state.p_pos
            y *= -1
            # 使用与实体相同的坐标转换逻辑
            x = (x / cam_range) * self.width // 2 * 0.9
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2
            
            # 创建透明surface来绘制捕获圈
            circle_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
            threshold_pixels = int(self.capture_threshold / cam_range * self.width // 2 * 0.9)  # 与实体渲染使用相同的缩放逻辑
            pygame.draw.circle(circle_surface, (0, 200, 0, 50), (int(x+0.5), int(y+0.5)), int(threshold_pixels+0.5), 2)  # 最后一个参数2是线宽
            self.screen.blit(circle_surface, (0, 0))
    
        # 绘制轨迹
        for agent in self.world.agents:
            if len(self.history_positions[agent.name]) >= 2:
                points = []
                for pos in self.history_positions[agent.name]:
                   x, y = pos
                   y *= -1
                   x = (x / cam_range) * self.width // 2 * 0.9
                   y = (y / cam_range) * self.height // 2 * 0.9
                   x += self.width // 2
                   y += self.height // 2
                   points.append((int(x), int(y)))
               
                # 绘制渐变轨迹
                for i in range(len(points) - 1):
                    alpha = int(255 * (i + 1) / len(points))
                    color = (0, 0, 255, alpha) if agent.adversary else (255, 0, 0, alpha)
                    line_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                    pygame.draw.line(line_surface, color, points[i], points[i + 1], 4)  # 最后一位是线宽
                    self.screen.blit(line_surface, (0, 0))
        # 绘制实体
        for entity in self.world.entities:
            x, y = entity.state.p_pos
            y *= -1
            x = (x / cam_range) * self.width // 2 * 0.9
            y = (y / cam_range) * self.height // 2 * 0.9
            x += self.width // 2
            y += self.height // 2


            # radius = entity.size * 140 * scaling_factor
            # # 修改为：根据世界到屏幕的实际转换比例计算
            world_to_screen_scale = (self.width / (2 * self.world_size)) * 0.9
            radius = entity.size * world_to_screen_scale

            if isinstance(entity, Agent):
             # 设置透明度：例如，transparent_alpha=128 (半透明)
                transparent_alpha = 200  # 透明度，范围从0（完全透明）到255（完全不透明）
                color = (0, 0, 255, transparent_alpha) if entity.adversary else (255, 0, 0, transparent_alpha)
                # 创建透明度支持的Surface
                agent_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

                # 添加 +0.5 进行四舍五入,修复视觉效果
                pygame.draw.circle(agent_surface, color, (int(x+0.5), int(y+0.5)), int(radius+0.5))

                self.screen.blit(agent_surface, (0, 0))

                pygame.draw.circle(self.screen, (255, 255, 255), (int(x+0.5), int(y+0.5)), int(radius+0.5), 1) # 绘制边框
            else:  # Landmark
                pygame.draw.circle(self.screen, (128, 128, 128), (int(x+0.5), int(y+0.5)), int(radius+0.5))
        pygame.display.flip()
    
    """绘制坐标轴"""
    def draw_grid_and_axes(self):
        cam_range = self.world_size  # 使用环境实际大小
        # 计算屏幕边界位置
        margin = 40  # 边距
        plot_width = self.width - 2 * margin
        plot_height = self.height - 2 * margin
     
        # 绘制边框
        pygame.draw.rect(self.screen, (0, 0, 0), 
                        (margin, margin, plot_width, plot_height), 1)
     
        # 绘制网格线
        grid_size = 0.5  # 网格间隔
        for x in np.arange(-self.world_size, self.world_size + grid_size, grid_size):
            screen_x = int((x + self.world_size) / (2 * self.world_size) * plot_width + margin)
            pygame.draw.line(self.screen, (220, 220, 220),
                            (screen_x, margin),
                            (screen_x, margin + plot_height), 1)
            # 绘制刻度
            if abs(x) % 1.0 < 0.01:  # 整数位置
                pygame.draw.line(self.screen, (0, 0, 0),
                               (screen_x, margin + plot_height),
                               (screen_x, margin + plot_height + 5), 1)
                text = self.game_font.render(f"{x:.0f}", True, (0, 0, 0))
                self.screen.blit(text, (screen_x - 5, margin + plot_height + 10))
     
        for y in np.arange(-self.world_size, self.world_size + grid_size, grid_size):
            screen_y = int((-y + self.world_size) / (2 * self.world_size) * plot_height + margin)
            pygame.draw.line(self.screen, (220, 220, 220),
                            (margin, screen_y),
                            (margin + plot_width, screen_y), 1)
            # 绘制刻度
            if abs(y) % 1.0 < 0.01:  # 整数位置
                pygame.draw.line(self.screen, (0, 0, 0),
                               (margin - 5, screen_y),
                               (margin, screen_y), 1)
                text = self.game_font.render(f"{y:.0f}", True, (0, 0, 0))
                text_rect = text.get_rect()
                self.screen.blit(text, (margin - 25, screen_y - 8))


    # debug测试渲染函数
    def debug_collision_rendering(self):
        """测试碰撞检测与渲染的一致性"""
        # 创建两个测试智能体（恰好接触的情况）
        test_agent1 = CustomAgent(is_scripted=False)
        test_agent2 = CustomAgent(is_scripted=False)

        # 设置基本属性
        test_agent1.size = self.world_size * 0.1  # 与正常智能体相同的尺寸
        test_agent2.size = self.world_size * 0.1
        test_agent1.adversary = True
        test_agent2.adversary = True

        # 设置状态
        test_agent1.state = self.world.agents[0].state.__class__()
        test_agent2.state = self.world.agents[0].state.__class__()

        # 设置位置 - 恰好接触的情况
        test_agent1.state.p_pos = np.array([0.0, 0.0])
        distance = test_agent1.size + test_agent2.size  # 恰好接触的距离
        test_agent2.state.p_pos = np.array([distance, 0.0])

        # 检查物理检测中是否接触
        delta_pos = test_agent1.state.p_pos - test_agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = test_agent1.size + test_agent2.size
        is_collision = dist < dist_min

        # 计算与主游戏区域相同的渲染比例
        cam_range = self.world_size
        scaling_factor = 0.7 * self.original_cam_range / cam_range

        # ======= 使用与draw函数完全相同的渲染逻辑 =======
        # 1. 计算半径 - 与draw函数一致
        debug_radius = test_agent1.size * 140 * scaling_factor
        # 2. 计算像素距离 - 与draw函数一致的转换方法
        pixel_distance = distance * 140 * scaling_factor

        # 输出调试信息
        print(f"=== 碰撞检测调试 ===")
        print(f"Agent1 位置: {test_agent1.state.p_pos}")
        print(f"Agent2 位置: {test_agent2.state.p_pos}")
        print(f"Agent1 大小: {test_agent1.size}")
        print(f"Agent2 大小: {test_agent2.size}")
        print(f"两点距离: {dist:.4f}, 最小碰撞距离: {dist_min:.4f}")
        print(f"物理引擎判定碰撞: {is_collision}")
        print(f"像素坐标下的半径: {int(debug_radius)}px")
        print(f"像素坐标下的间距: {int(pixel_distance)}px")

        # 绘制调试视图
        debug_margin = 50
        debug_width = 300  # 增加宽度，便于显示更多信息
        debug_height = 200  # 增加高度，便于显示重叠测试

        # 绘制调试区背景
        debug_surface = pygame.Surface((debug_width, debug_height), pygame.SRCALPHA)
        debug_surface.fill((240, 240, 240, 220))
        self.screen.blit(debug_surface, (debug_margin, self.height - debug_margin - debug_height))

        # === 测试场景1: 恰好接触 ===
        center_x = debug_margin + debug_width // 4
        center_y = self.height - debug_margin - debug_height // 4 * 1

        # ======= 使用与draw函数完全相同的渲染方法 =======
        # 创建透明度支持的Surface - 与draw函数相同
        agent_surface1 = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(agent_surface1, (0, 0, 255, 200), (center_x+0.5, center_y+0.5), int(debug_radius+0.5))
        self.screen.blit(agent_surface1, (0, 0))
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x+0.5, center_y+0.5), int(debug_radius+0.5), 1)

        # 绘制第二个智能体
        second_x = center_x + int(pixel_distance)
        agent_surface2 = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(agent_surface2, (0, 0, 255, 200), (second_x+0.5, center_y+0.5), int(debug_radius+0.5))
        self.screen.blit(agent_surface2, (0, 0))
        pygame.draw.circle(self.screen, (255, 255, 255), (second_x+0.5, center_y+0.5), int(debug_radius+0.5), 1)

        # 添加测试场景1标题
        font = self.game_font
        title_text = font.render("Sce1:Touching", True, (0, 0, 0))
        self.screen.blit(title_text, (center_x - 50, center_y - debug_radius - 25))

        # === 测试场景2: 轻微重叠 ===
        # 创建两个额外的测试智能体
        test_agent3 = CustomAgent(is_scripted=False)
        test_agent4 = CustomAgent(is_scripted=False)

        # 设置基本属性
        test_agent3.size = test_agent1.size
        test_agent4.size = test_agent2.size
        test_agent3.adversary = True
        test_agent4.adversary = True

        # 设置状态
        test_agent3.state = self.world.agents[0].state.__class__()
        test_agent4.state = self.world.agents[0].state.__class__()

        # 设置位置 - 轻微重叠（距离比最小碰撞距离小5%）
        test_agent3.state.p_pos = np.array([0.0, 0.0])
        overlap_distance = (test_agent3.size + test_agent4.size) * 0.95  # 轻微重叠
        test_agent4.state.p_pos = np.array([overlap_distance, 0.0])

        # 检查物理碰撞状态
        delta_pos_overlap = test_agent3.state.p_pos - test_agent4.state.p_pos
        dist_overlap = np.sqrt(np.sum(np.square(delta_pos_overlap)))
        dist_min_overlap = test_agent3.size + test_agent4.size
        is_collision_overlap = dist_overlap < dist_min_overlap

        # 为重叠情况计算像素距离 - 与draw函数相同的计算方式
        pixel_distance_overlap = overlap_distance * 140 * scaling_factor

        # 绘制测试场景2
        center_x2 = debug_margin + debug_width // 4 * 3
        center_y2 = self.height - debug_margin - debug_height // 4 * 1

        # 使用相同的渲染方式绘制重叠场景
        agent_surface3 = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(agent_surface3, (255, 0, 0, 200), (center_x2+0.5, center_y2+0.5), int(debug_radius+0.5))
        self.screen.blit(agent_surface3, (0, 0))
        pygame.draw.circle(self.screen, (255, 255, 255), (center_x2+0.5, center_y2+0.5), int(debug_radius+0.5), 1)

        # 绘制第二个智能体（重叠测试）
        second_x2 = center_x2 + int(pixel_distance_overlap)
        agent_surface4 = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        pygame.draw.circle(agent_surface4, (255, 0, 0, 200), (second_x2+0.5, center_y2+0.5), int(debug_radius+0.5))
        self.screen.blit(agent_surface4, (0, 0))
        pygame.draw.circle(self.screen, (255, 255, 255), (second_x2+0.5, center_y2+0.5), int(debug_radius+0.5), 1)

        # 添加测试场景2标题
        title_text2 = font.render("Sce2:Overlap", True, (0, 0, 0))
        self.screen.blit(title_text2, (center_x2 - 50, center_y2 - debug_radius - 25))

        # === 显示测试结果 ===
        # 场景1信息
        distance_text = font.render(f"Dist={dist:.3f}, Min Collision={dist_min:.3f}", True, (0, 0, 0))
        collision_text = font.render(f"Collision: {is_collision}", True, (255, 0, 0) if is_collision else (0, 100, 0))

        result_y = self.height - debug_margin - debug_height + debug_height // 2
        self.screen.blit(distance_text, (debug_margin + 10, result_y + 10))
        self.screen.blit(collision_text, (debug_margin + 10, result_y + 30))

        # 场景2信息
        overlap_text = font.render(f"Overlap:Dist={dist_overlap:.3f}, Min Collision={dist_min_overlap:.3f}", True, (0, 0, 0))
        overlap_collision_text = font.render(f"Collision: {is_collision_overlap}", True, (255, 0, 0) if is_collision_overlap else (0, 100, 0))

        self.screen.blit(overlap_text, (debug_margin + debug_width // 2 + 10, result_y + 10))
        self.screen.blit(overlap_collision_text, (debug_margin + debug_width // 2 + 10, result_y + 30))

        # 打印舍入误差相关信息 - 有助于分析问题
        print(f"舍入信息: debug_radius={int(debug_radius)}, pixel_distance={int(pixel_distance)}")
        print(f"两个圆心之间的像素距离: {int(pixel_distance)}")
        print(f"两个半径之和的像素值: {int(debug_radius) * 2}")
        print(f"像素差异: {int(pixel_distance) - int(debug_radius) * 2}")

        # 输出更详细的比较信息
        print(f"=== 渲染比较 ===")
        print(f"调试区域半径计算: {test_agent1.size} * 140 * {scaling_factor} = {test_agent1.size * 140 * scaling_factor}")
        print(f"主渲染使用的半径计算: entity.size * 140 * scaling_factor")

        # 增加比较测试
        main_game_entity = self.world.agents[0]  # 获取一个主游戏中的智能体
        main_game_radius = main_game_entity.size * 140 * scaling_factor
        print(f"主游戏中智能体大小: {main_game_entity.size}, 计算半径: {main_game_radius}")
        print(f"调试智能体大小: {test_agent1.size}, 计算半径: {debug_radius}")
        print("===============================")

        # 额外：在调试区域底部显示舍入对比
        rounding_text = font.render(f"Rounding:pixel_dis={int(pixel_distance)}, 2*radius={int(debug_radius)*2}, diff={int(pixel_distance)-int(debug_radius)*2}", 
                                   True, (0, 0, 0))
        self.screen.blit(rounding_text, (debug_margin + 10, result_y + 60))

    def check_capture_condition(self,threshold = None): # agent.size = 0.075 if agent.adversary else 0.05
        """
        检查逃跑者是否被围捕。
        判断条件：
        1. 逃跑者被追捕者围成的区域包围
        2. 追捕者与逃跑者的平均距离小于阈值
        
        Args:
            threshold (float): 围捕者和逃跑者之间的最大允许平均距离。
        """
        if threshold is None:
            threshold = self.world_size * 0.2 # 使用世界大小的20%作为默认捕获范围
        agents = self.scenario.good_agents(self.world)  # 逃跑者
        adversaries = self.scenario.adversaries(self.world)  # 围捕者
        # 统一返回结构
        capture_info = {
            "surrounded": False,
            "avg_distance": None,
            "threshold": threshold
        }
        # 重置world状态
        self.world.is_surrounded = False
        self.world.is_captured = False
        self.world.triangle_areas = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "Sum_S": 0}
        # 假设只有一个逃跑者，计算所有围捕者与该逃跑者的距离
        for agent in agents:  
            # 计算所有追捕者与（单个）目标的距离
            distances = [np.linalg.norm(agent.state.p_pos - adv.state.p_pos) for adv in adversaries]
            avg_distance = np.mean(distances)
            capture_info["avg_distance"] = avg_distance
            # 条件1：平均距离必须小于阈值
            if avg_distance >= threshold:
                continue

            # 条件2：根据追捕者数量判断包围状态
            if len(adversaries) >= 3:  # 3个追捕者
                # 3个及以上追捕者：使用三角形面积法或方向检测法
                pursuers_pos = np.array([adv.state.p_pos for adv in adversaries])
                is_surrounded, triangle_areas = self._is_point_surrounded(pursuers_pos, agent.state.p_pos)
                capture_info["surrounded"] = is_surrounded
                # 保存三角形面积信息到world对象
                self.world.triangle_areas = {
                    "S1": triangle_areas[0],
                    "S2": triangle_areas[1],
                    "S3": triangle_areas[2],
                    "S4": triangle_areas[3],
                    "Sum_S": triangle_areas[4]
                }
                    
                # 更新world状态
                self.world.is_surrounded = is_surrounded
                if self.world.is_surrounded:
                    print(f"包围成功!")
                
                if not capture_info["surrounded"]:
                    continue

            elif len(adversaries) == 2: # 2个追捕者：检查是否形成半包围（两个追捕者在目标的相对两侧）
                pos1, pos2 = adversaries[0].state.p_pos, adversaries[1].state.p_pos
                target = agent.state.p_pos
                # 计算向量夹角
                vec1 = pos1 - target
                vec2 = pos2 - target
                dot_product = np.dot(vec1, vec2)
                norm1 = np.linalg.norm(vec1)
                norm2 = np.linalg.norm(vec2)
                # 如果夹角接近180度（两个追捕者在相对两侧），认为形成半包围 -> TODO: 待验证
                cos_angle = dot_product / (norm1 * norm2 + 1e-10)
                capture_info["surrounded"] = cos_angle < -0.5  # 夹角大于约120度
                if not capture_info["surrounded"]:
                    continue
            elif len(adversaries) == 1: # 1个追捕者：只需满足距离条件
                capture_info["surrounded"] = True
            
            # 同时满足条件，设置终止状态
            self.world.is_captured = True
            # 修复：不要在这里设置 terminations，而是在 step 方法中统一设置
            # for a in self.world.agents:
            #     self.terminations[a] = True
            return True, capture_info
        return False, capture_info

    # 包围的判定是错的！（怪不得成功率很低）
    def _is_point_surrounded(self, points, target):
        """
        适用于3个追捕者的情形。
        判断目标点是否被一组点"包围"
        使用三角形面积法判断目标是否在追捕者形成的三角形内部
        Args:
            points (np.array): 围捕者的位置数组，形状为 (n, 2)
            target (np.array): 目标点（逃跑者）的位置，形状为 (2,)
        Returns:
            bool: 如果目标被包围则返回True，否则返回False
        """
        # 如果追捕者小于3，无法形成包围
        if len(points) < 3:
            return False,[None,None,None,None,None]
        
        S1,S2,S3,S4 = self._calculate_triangle_areas(points, target)
        
        # 如果三角形面积过小，说明三点几乎共线，无法形成有效包围
        if len(points) == 3:
            if S4 < 1e-5:
                return False, [S1, S2, S3, S4, 0]  # 修复：返回完整元组
            
            sum_S = S1 + S2 + S3
            return (abs(sum_S - S4) < 1e-5), [S1,S2,S3,S4, sum_S]
        return False, [S1,S2,S3,S4,sum_S]

    def _calculate_triangle_areas(self, pursuers_pos, target_pos):
        """目前只适用于3个追捕者，计算三角形面积，用于判断包围状态"""
        def triangle_area(p1, p2, p3):
            return 0.5 * abs((p1[0]*(p2[1]-p3[1]) + p2[0]*(p3[1]-p1[1]) + p3[0]*(p1[1]-p2[1])))
        
        # 确保有足够的追捕者
        if len(pursuers_pos) < 3:
            return 0, 0, 0, 0
            
        # 计算三个追捕者形成的三角形面积
        S4 = triangle_area(pursuers_pos[0], pursuers_pos[1], pursuers_pos[2])
        
        # 计算目标与每对追捕者形成的三角形面积
        S1 = triangle_area(target_pos, pursuers_pos[0], pursuers_pos[1])
        S2 = triangle_area(target_pos, pursuers_pos[1], pursuers_pos[2])
        S3 = triangle_area(target_pos, pursuers_pos[2], pursuers_pos[0])
        
        return S1, S2, S3, S4
    

env = make_env(Custom_raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2, _world_size=2.5):
        # world = World() # core.py 中的World类
        world = CustomWorld() # Ronchy: 使用自定义的World类,重载了动力学逻辑
        # set any world properties first
        world.world_size =  _world_size # Ronchy添加世界大小
        world.capture_threshold = _world_size * 0.2  # Ronchy添加围捕半径
        
        world.is_captured = False  # Ronchy添加围捕成功标志
        world.is_surrounded = False  # Ronchy添加包围成功标志
        world.triangle_areas  = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "Sum_S": 0} # Ronchy添加 pre-eva形成的面积
        world.episode_steps = 0

        world.dim_c = 0  # Ronchy set 0, communication channel dimensionality,default 2
        world.dim_p = 2  # position dimensionality, default 2
        """
        time_step = 0.1  这个是在core.py中的World类中定义的,名称为 dt = 0.1
        agent的运动都在core.py中的World类中的step()方法中进行
        """
        world.dt = 0.1 # time_step, default 0.1
        world.damping = 0.2  # 阻尼系数 0.25是默认值
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        # add agents
        # world.agents = [Agent() for i in range(num_agents)]

        agent_adversaries = [CustomAgent(is_scripted= False) for i in range(num_adversaries)]
        agent_good_agents = [CustomAgent(is_scripted= True) for i in range(num_good)]
        world.agents = agent_adversaries + agent_good_agents

        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            base_size = _world_size * 0.05 # 基础大小为世界大小的10%
            # agent.size = 0.25 if agent.adversary else 0.15  # 智能体的半径，判断是否碰撞的界定
            agent.size = base_size if agent.adversary else base_size*0.6  # 智能体的半径，判断是否碰撞的界定
            agent.initial_mass = 1.6 if agent.adversary else 0.8  # 智能体的质量 kg
            agent.accel = None # 不使用该参数
            # agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.max_speed = 1.0 if agent.adversary else 1.0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = 0.2
            landmark.boundary = False
        return world

    def reset_world(self, world, np_random):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = (
                np.array([0.35, 0.85, 0.35])
                if not agent.adversary
                else np.array([0.85, 0.35, 0.35])
            )
            # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        
        # set random initial states
        for agent in world.agents:
            # agent.state.p_pos = np_random.uniform(-1, +1, world.dim_p) # default: 2
            agent.state.p_pos = np_random.uniform(-world.world_size * 0.8, +world.world_size * 0.8, world.dim_p) #  # 留出20%边界
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            if not landmark.boundary:
                # landmark.state.p_pos = np_random.uniform(-0.9, +0.9, world.dim_p) # default: 1.8
                landmark.state.p_pos = np_random.uniform(-world.world_size * 0.8, +world.world_size * 0.8, world.dim_p) # 留出20%边界
                landmark.state.p_vel = np.zeros(world.dim_p)

    def benchmark_data(self, agent, world):
        # returns data for benchmarking purposes
        if agent.adversary:
            collisions = 0
            for a in self.good_agents(world):
                if self.is_collision(a, agent):
                    collisions += 1
            return collisions
        else:
            return 0

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist < dist_min else False

    # return all agents that are not adversaries
    def good_agents(self, world):
        return [agent for agent in world.agents if not agent.adversary]

    # return all adversarial agents
    def adversaries(self, world):
        return [agent for agent in world.agents if agent.adversary]

    def reward(self, agent, world):  # main_reward 也是一个数值，而不是元组
        # Agents are rewarded based on minimum agent distance to each landmark
        main_reward = (
            self.adversary_reward(agent, world)
            if agent.adversary
            else self.agent_reward(agent, world)
        )
        # print(f"main_reward{main_reward}")
        return main_reward

    # 逃跑者reward设置
    def agent_reward(self, agent, world):
        # Agents are negatively rewarded if caught by adversaries
        rew = 0
        return rew

    # 围捕者reward设置
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        agents = self.good_agents(world)
          
        for target in agents:  # 通常只有一个逃跑者
            # 1. 距离奖励 - 鼓励追捕者接近逃跑者
            distance = np.linalg.norm(agent.state.p_pos - target.state.p_pos)
            # 归一化距离系数
            dist_factor = distance / world.world_size
            # # 距离奖励：使用非线性奖励，让接近时奖励增长更快——错误的！
            # distance_reward = 0.5 * (dist_factor ** 2)  # 使用二次函数放大近距离奖励  #如果超出了边界，离逃跑者越远，奖励越高
            if distance > world.capture_threshold:
                approach_reward = -dist_factor
                rew += approach_reward
            else:
                rew += 1.0  # 当距离小于阈值时，给予最大奖励
            # 3. 碰撞奖励 - 如果碰撞到逃跑者则给予高额奖励
            if self.is_collision(agent, target):
                rew += 1.0
        # 4. 边界惩罚 - 防止智能体靠近边界或离开环境（同时在底层环境限制世界范围）
        def bound(x):
            boundary_start = world.world_size * 0.9 
            full_boundary = world.world_size
            if x < boundary_start:
                return 0
            if x < full_boundary:
                return (x - boundary_start) * 15
            return min(np.exp(2 * x - 2 * full_boundary), 10)
        # ==== 必须添加实际边界计算 ====
        for p in range(world.dim_p):  # 遍历每个坐标轴 (x, y)
            x = abs(agent.state.p_pos[p])  # 获取坐标绝对值
            rew -= bound(x)  # 应用边界惩罚函数
        return rew
    
    def global_reward(self, world):
        """
        计算全局奖励，鼓励追捕者合作追捕逃避者
        简化版：专注于形成包围和接近目标
        """
        global_rew = 0.0
        agents = self.good_agents(world)      # 逃跑者
        adversaries = self.adversaries(world)  # 围捕者
        # 如果没有逃跑者或者追捕者，直接返回0奖励
        if not agents or not adversaries:
            return 0.0
        
        # 获取全局设置的围捕阈值
        capture_threshold = world.capture_threshold if hasattr(world, 'env') else world.world_size * 0.2
        # 安全地获取捕获信息
        distances, pursuers_pos, target_pos = None, None, None
        # 已有围捕信息，直接使用
        target = agents[0] # 获取第一个逃跑者
        target_pos = target.state.p_pos
        # target_speed = np.linalg.norm(target.state.p_vel)
        pursuers_pos = np.array([adv.state.p_pos for adv in adversaries])
        distances = np.array([np.linalg.norm(p - target_pos) for p in pursuers_pos])
        # 计算平均距离和归一化
        mean_distance = np.mean(distances)
        min_distance = np.min(distances)

        # 检查目标是否被包围（使用手动实现的方法，而非ConvexHull）
        is_surrounded = world.is_surrounded
        # 成功围捕的条件：目标被包围且至少一个追捕者足够近
        is_captured = world.is_captured
        
        # 奖励分配 - 简化为三个基本阶段
        # 1. 已成功围捕 - 给予最高奖励
        if is_captured or (hasattr(world, 'is_captured') and world.is_captured):
            global_rew += 100.0
            return global_rew
        
        # 2. 已形成包围但距离不够近 - 给予中等奖励并鼓励接近
        if is_surrounded:
            # 基础包围奖励
            global_rew += 5.0
            # 接近奖励
            proximity_reward = 3.0 * (1.0 - min_distance / capture_threshold)
            global_rew += proximity_reward
            # return global_rew

        # 2.2 is_surrounded = False - >鼓励包围
        if is_surrounded == False and (mean_distance > capture_threshold * 2):
            S1, S2, S3, S4, Sum_S = [world.triangle_areas[key] for key in ["S1", "S2", "S3", "S4", "Sum_S"]]
            # 面积差异越小，越接近包围
            encircle_progress = -  ((1/len(pursuers_pos)) * np.log(Sum_S - S4 + 1))
            global_rew += 0.02 * encircle_progress
            # return global_rew
        
        # 3. 未形成包围 - 给予基础接近奖励
        approach_reward = 1.0 * (1.0 - (mean_distance / world.world_size))
        global_rew += approach_reward
        # 添加时间压力
        time_pressure = -0.01 * world.episode_steps  # 随着时间推移增加负奖励
        # if world.episode_steps == 1: #test print
        #     print("-----world.episode_steps-----")
        global_rew += time_pressure

        return global_rew
    
    # 待进一步修改
    def observation(self, agent, world):  # 返回值，自动适配智能体的观测空间维数
        """
            智能体及地标的观测空间
            TODO:需要按需重载。
        """
        # 1. 自身状态归一化
        norm_self_vel = agent.state.p_vel / agent.max_speed
        norm_self_pos = agent.state.p_pos / world.world_size
        # 2. 障碍物信息
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:
                relative_pos = (entity.state.p_pos - agent.state.p_pos) / world.world_size
                entity_pos.append(relative_pos)
        # 3. 其他智能体信息 - 这部分是通信网络能够学习做出贡献的关键
        other_pos = []  # 保留相对位置信息
        # 只对追捕者，看逃跑者相关位置
        if agent.adversary:
            # 追捕者观察逃跑者信息
            for other in world.agents:
                if not other.adversary:  # 对于逃跑者
                    # 相对位置
                    relative_pos = (other.state.p_pos - agent.state.p_pos) / world.world_size
                    other_pos.append(relative_pos)
                    # 归一化速度
                    norm_vel = other.state.p_vel / other.max_speed
    
        # 4. 最终观测 - 基本版本，足够通信网络工作但不过于复杂
        return np.concatenate(
            [norm_self_vel]  # 自身速度
            + [norm_self_pos]  # 自身位置
            + entity_pos  # 障碍物位置
            + other_pos  # 其他智能体相对位置（针对追捕者只包含逃跑者）
        )

#=======================================================================
if __name__ =="__main__":
    print("Custom_raw_env",Custom_raw_env)
    # 创建测试环境
    # env = make_env(Custom_raw_env)
    # parallel_env = parallel_wrapper_fn(env) # 启用并行环境
    # parallel_env.reset()

    num_good = 1
    num_adversaries = 3
    num_obstacles = 0

        # 创建并初始化环境
    env = Custom_raw_env(
        num_good=num_good, 
        num_adversaries=num_adversaries, 
        num_obstacles=num_obstacles, 
        continuous_actions=True,  # 设置为 True 使用连续动作空间
        render_mode="None"
    )

    env.reset()

    # 打印环境和智能体信息
    print("环境初始化完成。")
    print(f"环境名称: {env.metadata['name']}")
    print(f"智能体数量: {len(env.agents)}")

   # 遍历每个智能体并打印其初始状态
    for agent_name in env.agents:
        # 获取当前智能体的观察空间和动作空间
        obs_space = env.observation_space(agent_name)
        action_space = env.action_space(agent_name)

        # 获取当前智能体的观测
        observation = env.observe(agent_name)

        # 获取当前智能体的动作空间范围（低和高值）
        action_low = action_space.low
        action_high = action_space.high

        # 打印信息
        print(f"\n==== {agent_name} ====")
        print(f"观测空间维度: {obs_space.shape}")
        print(f"动作空间维度: {action_space.shape}")
        print(f"动作空间的低值: {action_low}")
        print(f"动作空间的高值: {action_high}")

        # 打印智能体的初始观测
        print(f"初始观测: {observation}")
        
        # 如果你想测试环境的一个动作，可以给智能体一个随机动作，并打印
        random_action = action_space.sample()  # 从动作空间中采样一个随机动作
        print(f"随机选择的动作: {random_action}")