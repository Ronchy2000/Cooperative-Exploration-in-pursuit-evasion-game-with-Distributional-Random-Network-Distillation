# from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3

# from pettingzoo.mpe.simple_tag_v3 import raw_env
"""
原版的渲染有问题，主要集中在：

围捕圈内显示is_collision判断！！！！。。
在evaluate时，在is_collision中加入打印，确实会发现问题！
- 在物理世界中：当两个智能体相距 agent1.size + agent2.size 时发生碰撞
- 在渲染中：智能体看起来半径是 agent.size * 140 * scaling_factor 像素
问题在于这两种计算方式不成比例，导致即使物理上已经碰撞了，视觉上还有很大距离

"""


import numpy as np
import gymnasium
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .custom_agent_dynamics_with_stationary_prey import CustomAgent, CustomWorld # Ronchy自定义的World类，用于测试自定义的智能体动力学模型

import pygame  #Ronchy: 用于渲染动画环境

'''
继承 raw_env, 修改部分功能。
V3 特性：
- 捕获判断条件：2个及以上的进入阈值

V4_partially特性：
- 为追捕者加入观测范围！
- 猎物是静止的，不会移动

V5, 加入两阶段的观测
- 修改观测空间
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
        world_size = 1, # Ronchy添加 world_size参数 ,地图大小 world_size x world_size
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
        self.metadata["name"] = "simple_tag_env_with_stationary_v5_Partially_2Phase"
        # Ronchy添加轨迹记录
        self.history_positions = {agent.name: [] for agent in world.agents}
        # self.max_history_length = 500  # 最大轨迹长度

        # 重载 simple_env.py中的代码
        pygame.font.init()
        self.game_font = pygame.font.SysFont('arial', 16)  # 使用系统字体

        self.max_force = 1.0  # 最大力
        # self.capture_threshold = self.world_size * 0.2 # 围捕阈值: 使用世界大小的20%作为默认捕获范围
        self.local_ratio = 0.5  # 全局奖励和局部奖励的比例 local*个体 + (1-local)*全局共享
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
        # V5_2phase
        self.blackboard = {
            'm_target': 
            {
                'target_pos': None,
                'target_vel': None
            } # 目标位置，速度 信息
        }
        self.communication_phase = 0  # 重置为探索阶段

    def reset_world(self, world, np_random):
        # 清除历史轨迹
        self.history_positions = {agent.name: [] for agent in self.world.agents}  
        world.is_captured = False
        world.is_surrounded = False
        world.episode_steps = 0  # 每个episode 重置步数计数器
        # V5_2phase
        world.blackboard = {
            'm_target': 
            {
                'target_pos': None,
                'target_vel': None
            } # 目标位置，速度 信息
        }
        world.communication_phase = 0  # 重置为探索阶段

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
        self.world.episode_steps += 1  # Ronchy add更新步数计数

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
        
        # Ronchy 添加力限幅
        #self.max_force = 1.0  # 根据需求调整
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
            self.scenario.update_communication_phase(self.world)

            self._execute_world_step()

            # Ronchy记录轨迹
            for agent in self.world.agents:
                self.history_positions[agent.name].append(agent.state.p_pos.copy())
                # if len(self.history_positions[agent.name]) > self.max_history_length: # 限制轨迹长度
                #     self.history_positions[agent.name].pop(0)

            self.steps += 1
            # 检查围捕条件
            is_captured, capture_info = self.check_capture_condition(threshold=self.world.capture_threshold)  #围捕标志——半径
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
        # self.debug_observation_rendering()
        # self.test_agent_obstacle_contact()

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

        # 为每个追捕者绘制观测范围
        for agent in self.world.agents:
            if agent.adversary:
                x, y = agent.state.p_pos
                y *= -1
                x = (x / cam_range) * self.width // 2 * 0.9
                y = (y / cam_range) * self.height // 2 * 0.9
                x += self.width // 2
                y += self.height // 2
                
                # 创建透明surface来绘制观测范围
                obs_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
                obs_radius = int(self.world.observation_radius / cam_range * self.width // 2 * 0.9) # 与实体渲染使用相同的缩放逻辑
                pygame.draw.circle(obs_surface, (0, 0, 255, 30), (int(x+0.5), int(y+0.5)), int(obs_radius+0.5), 1)
                self.screen.blit(obs_surface, (0, 0))
        
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
        if self.render_mode == "human": # 在渲染模式为human时才调用flip
            pygame.display.flip()
    
    """绘制坐标轴"""
    def draw_grid_and_axes(self):
        cam_range = self.world_size  # 使用环境实际大小
        # 计算屏幕边界位置
        margin = 40  # 边距
        plot_width = self.width - 2 * margin
        plot_height = self.height - 2 * margin
     
        # 绘制边框 - 为了保证边框渲染正确，手动调整像素值
        '''
        这个问题与pygame中如何渲染线条有关。当绘制宽度为1像素的线条或矩形边框时，pygame会将线条放置在像素网格上，而不是像素之间。由于计算机图形中的坐标通常从像素的左上角开始，这可能导致某些边在某些环境下看起来"虚化"或不明显。
        '''
        pygame.draw.rect(self.screen, (0, 0, 0), 
                        (margin-1, margin-1, plot_width+1, plot_height+1), 1)

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
    def debug_observation_rendering(self):
        """测试观测范围渲染与实际观测逻辑的一致性"""
        # 创建测试舞台
        debug_margin = 50
        debug_width = 600
        debug_height = 400
        debug_surface = pygame.Surface((debug_width, debug_height), pygame.SRCALPHA)
        debug_surface.fill((240, 240, 240, 220))
        self.screen.blit(debug_surface, (debug_margin, self.height - debug_margin - debug_height))
        
        # 获取一个追捕者和逃跑者用于测试
        pursuer = None
        evader = None
        for agent in self.world.agents:
            if agent.adversary and pursuer is None:
                pursuer = agent
            elif not agent.adversary and evader is None:
                evader = agent
            if pursuer and evader:
                break
        
        if not pursuer or not evader:
            return
        
        # 获取逻辑中使用的参数值
        observation_radius = self.world.observation_radius
        pursuer_size = pursuer.size
        evader_size = evader.size
        
        # 计算实际距离
        actual_distance = np.linalg.norm(pursuer.state.p_pos - evader.state.p_pos)
        is_observed = actual_distance <= observation_radius
        is_collision = self.scenario.is_collision(pursuer, evader)
        
        # 计算渲染中使用的参数值
        cam_range = self.world_size
        world_to_screen_scale = (self.width / (2 * self.world_size)) * 0.9
        rendered_obs_radius = observation_radius * world_to_screen_scale
        rendered_pursuer_radius = pursuer_size * world_to_screen_scale
        rendered_evader_radius = evader_size * world_to_screen_scale
        
        # 画出测试场景
        center_x = debug_margin + 100
        center_y = self.height - debug_margin - debug_height + 100
        
        # 绘制追捕者
        pygame.draw.circle(self.screen, (0, 0, 255, 200), (center_x, center_y), int(rendered_pursuer_radius+0.5))
        
        # 绘制观测范围
        pygame.draw.circle(self.screen, (0, 0, 255, 50), (center_x, center_y), int(rendered_obs_radius+0.5), 2)
        
        # 绘制逃跑者在各种距离的位置
        distances = [
            observation_radius * 0.5,  # 观测范围内
            observation_radius * 0.99, # 刚好在观测范围内
            observation_radius * 1.01, # 刚好在观测范围外
            observation_radius * 1.5   # 观测范围外
        ]
        
        for i, dist in enumerate(distances):
            angle = i * np.pi / 2  # 在四个方向放置
            dx = dist * np.cos(angle)
            dy = dist * np.sin(angle)
            
            # 计算屏幕坐标
            evader_x = center_x + int(dx * world_to_screen_scale)
            evader_y = center_y + int(dy * world_to_screen_scale)
            
            # 绘制逃跑者
            color = (0, 255, 0, 200) if dist <= observation_radius else (255, 0, 0, 200)
            pygame.draw.circle(self.screen, color, (evader_x, evader_y), int(rendered_evader_radius+0.5))
            
            # 添加距离标签
            distance_text = self.game_font.render(f"{dist:.2f}", True, (0, 0, 0))
            self.screen.blit(distance_text, (evader_x + 10, evader_y - 10))
        
        # 显示关键参数
        font = self.game_font
        info_texts = [
            f"observation_radius: {observation_radius:.3f} (rendered_obs_radius: {rendered_obs_radius:.1f}px)",
            f"actual_distance: {actual_distance:.3f}",
            f"is_observed: {is_observed}",
            f"is_collision: {is_collision}",
            f"pursuer_size: {pursuer_size:.3f} (rendered_pursuer_radius: {rendered_pursuer_radius:.1f}px)", 
            f"evader_size: {evader_size:.3f} (rendered_evader_radius: {rendered_evader_radius:.1f}px)",
            f"world.comm_phase: {self.world.communication_phase}"
        ]
        
        for i, text in enumerate(info_texts):
            info_render = font.render(text, True, (0, 0, 0))
            self.screen.blit(info_render, (debug_margin + 300, self.height - debug_margin - debug_height + 30 + i * 20))
    
    def test_agent_obstacle_contact(self):
        """Test agent-obstacle collision rendering by placing an agent next to an obstacle"""
        # Find first adversary and obstacle
        agent = None
        obstacle = None
        
        for a in self.world.agents:
            if a.adversary:
                agent = a
                break
        
        for l in self.world.landmarks:
            if not l.boundary:
                obstacle = l
                break
        
        if not agent or not obstacle:
            return
        
        # Create debug panel
        debug_panel_height = 150
        debug_panel_y = 20
        debug_panel = pygame.Surface((300, debug_panel_height), pygame.SRCALPHA)
        debug_panel.fill((240, 240, 240, 200))
        self.screen.blit(debug_panel, (20, debug_panel_y))
        
        # Calculate position where agent should exactly touch the obstacle
        direction = np.array([1.0, 0.0])  # Place agent to the right of obstacle
        distance = agent.size + obstacle.size  # Sum of radii - exact contact point
        
        # Force agent to exact contact position
        agent.state.p_pos = obstacle.state.p_pos + direction * distance
        
        # Display collision information
        font = pygame.font.SysFont('Arial', 14)
        
        # Calculate actual distance
        actual_dist = np.linalg.norm(agent.state.p_pos - obstacle.state.p_pos)
        contact_dist = agent.size + obstacle.size
        
        # Render information
        texts = [
            f"Agent radius: {agent.size:.3f}",
            f"Obstacle radius: {obstacle.size:.3f}",
            f"Contact distance: {contact_dist:.3f}",
            f"Actual distance: {actual_dist:.3f}",
            f"Is collision: {self.scenario.is_collision(agent, obstacle)}"
        ]
        
        for i, text in enumerate(texts):
            text_surface = font.render(text, True, (0, 0, 0))
            self.screen.blit(text_surface, (30, debug_panel_y + 20 + i * 20))
        
        # Draw a line between agent and obstacle centers
        world_to_screen = lambda p: (
            int((p[0] + self.world_size) / (2 * self.world_size) * self.width),
            int((self.world_size - p[1]) / (2 * self.world_size) * self.height)
        )
        
        agent_center = world_to_screen(agent.state.p_pos)
        obstacle_center = world_to_screen(obstacle.state.p_pos)
        
        # Draw connection line
        pygame.draw.line(self.screen, (255, 0, 0), agent_center, obstacle_center, 1)
                
    def check_capture_condition(self,threshold = None): # agent.size = 0.075 if agent.adversary else 0.05
        """
        检查逃跑者是否被围捕。
        V2判断条件：
        1. 逃跑者被追捕者围成的区域包围
        2. 追捕者与逃跑者的平均距离小于阈值
        
        V3环境判定条件： 捕获条件设置 -> 2个及以上追击者的进入阈值。

        Args:
            threshold (float): 围捕者和逃跑者之间的最大允许距离。
        """
        if threshold is None:
            threshold = self.world_size * 0.2 # 使用世界大小的20%作为默认捕获范围
        agents = self.scenario.good_agents(self.world)  # 逃跑者
        adversaries = self.scenario.adversaries(self.world)  # 围捕者
        # 统一返回结构
        capture_info = {
            "captured": False,
            "pursuers_in_range": 0,
            "threshold": threshold
        }
        # 重置world状态
        self.world.is_surrounded = False
        self.world.is_captured = False
        # 假设只有一个逃跑者，计算所有围捕者与该逃跑者的距离
        for agent in agents:  
            pursuers_in_range = 0 # 进入阈值范围内的追捕者数量
            # 计算所有追捕者与（单个）目标的距离
            for adv in adversaries:
                distances = np.linalg.norm(agent.state.p_pos - adv.state.p_pos)
                if distances <= threshold:
                    pursuers_in_range += 1
            capture_info["pursuers_in_range"] = pursuers_in_range
            # 捕获条件设置 -> 2个及以上追捕者的进入阈值。
            if pursuers_in_range >= 2:
                self.world.is_captured = True
                capture_info["captured"] = True
                return True, capture_info
            
        return False, capture_info


env = make_env(Custom_raw_env)
parallel_env = parallel_wrapper_fn(env)


class Scenario(BaseScenario):
    def make_world(self, num_good=1, num_adversaries=3, num_obstacles=2, _world_size=2.5):
        # world = World() # core.py 中的World类
        world = CustomWorld() # Ronchy: 使用自定义的World类,重载了动力学逻辑
        # set any world properties first
        world.world_size =  _world_size # Ronchy添加世界大小
        world.capture_threshold = _world_size * 0.27  # Ronchy添加围捕半径
        # V4环境特性
        world.observation_radius = _world_size * 0.4  # # 添加观测半径，半径是世界大小的30%

        world.is_captured = False  # Ronchy添加围捕成功标志
        world.is_surrounded = False  # Ronchy添加包围成功标志
        world.triangle_areas  = {"S1": 0, "S2": 0, "S3": 0, "S4": 0, "Sum_S": 0} # Ronchy添加 pre-eva形成的面积
        world.episode_steps = 0
        world.dim_c = 0  # Ronchy set 0, communication channel dimensionality,default 2
        world.dim_p = 2  # position dimensionality, default 2
        # V5_2phase
        world.blackboard = {
            'm_target': 
            {
                'target_pos': None,
                'target_vel': None
            } # 目标位置，速度 信息
        }
        world.communication_phase = 0 # 0: 探索阶段, 1: 追捕阶段
        """
        time_step = 0.1  这个是在core.py中的World类中定义的,名称为 dt = 0.1
        agent的运动都在core.py中的World类中的step()方法中进行
        """
        world.dt = 0.1 # time_step, default 0.1
        world.damping = 0.01  # 阻尼系数 0.25是默认值
        num_good_agents = num_good
        num_adversaries = num_adversaries
        num_agents = num_adversaries + num_good_agents
        num_landmarks = num_obstacles
        # add agents
        # world.agents = [Agent() for i in range(num_agents)]

        agent_adversaries = [CustomAgent(is_scripted= False) for i in range(num_adversaries)]
        agent_good_agents = [CustomAgent(is_scripted= True) for i in range(num_good)]
        world.agents = agent_adversaries + agent_good_agents
        base_size = _world_size * 0.05 # 基础大小为世界大小的  10%还是太大了！ -> 5%
        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            # agent.size = 0.25 if agent.adversary else 0.15  # 智能体的半径，判断是否碰撞的界定
            agent.size = base_size*1 if agent.adversary else base_size*0.7  # 智能体的半径，判断是否碰撞的界定
            agent.initial_mass = 2 if agent.adversary else 1.5  # 智能体的质量 kg
            agent.accel = None # 不使用该参数
            # agent.max_speed = 1.0 if agent.adversary else 1.3
            agent.max_speed = 1.0 if agent.adversary else 1.0
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = "landmark %d" % i
            landmark.collide = True
            landmark.movable = False
            landmark.size = base_size * 2.0
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
            agent.state.p_pos = np_random.uniform(-world.world_size * 0.8, +world.world_size * 0.8, world.dim_p) #  # 留出10%边界
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
        shape = False  # Ronchy 改为True
        adversaries = self.adversaries(world)
        if (
            shape
        ):  # reward can optionally be shaped (increased reward for increased distance from adversary)
            for adv in adversaries:
                distance = np.linalg.norm(agent.state.p_pos - adv.state.p_pos)
                if distance <= world.observation_radius + adv.size:
                    rew += np.sqrt(np.sum(np.square(agent.state.p_pos - adv.state.p_pos)))

        # agent.collide default value is True
        if agent.collide:
            for a in adversaries:
                if self.is_collision(a, agent):
                    # print(f"与 {a.name} 发生碰撞！")
                    rew -= 1.0  # 0:即，不学习逃跑策略。 default value = 10

        # agents are penalized for exiting the screen, so that they can be caught by the adversaries
        # 修改为动态边界（假设边界为 world.world_size 的 96% 开始衰减）
        # 边界惩罚的新定义
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

    # 围捕者reward设置
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        agents = self.good_agents(world)
        for target in agents:  # 通常只有一个逃跑者
            distance = np.linalg.norm(agent.state.p_pos - target.state.p_pos)

            if world.communication_phase == 0:
                # print("communication_phase == 0")
                pass # simhash探索奖励！

            elif world.communication_phase == 1 : # 其他agent观测到or自己观测到
                # print("communication_phase == 1")
                target_pos = world.blackboard['m_target']['target_pos']
                target_vel = world.blackboard['m_target']['target_vel']
                distance = np.linalg.norm(agent.state.p_pos - target_pos)
                dist_factor = distance / world.world_size
                # 距离奖励
                if distance > world.capture_threshold:
                    approach_reward = -dist_factor
                    rew += approach_reward
                else:
                    rew += 5.0  # 当距离小于阈值时，给予最大奖励

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

    # bug obs对reward的影响为0了，即使有一位追捕者观测到了逃跑者，其他从comm网络传去的信息没法体现在reward中，导致其他agent没有驱动力。
    # 下一步做的是，如何把comm压缩后的信息与reward加强联系。。驱动下一个agent去追逃跑者！
    # 2025.7.13 V5版本，DONE! 加入黑板系统，观测到的agent可以将target的信息写在黑板上！
    def global_reward(self, world):
        """
            计算全局奖励，鼓励追捕者合作追捕逃避者
            奖励公式: r = sum([collision(0,i) - dist(0,i)]) - v0, i=1,2,3
            其中:
            - collision(0,i): 逃避者与追捕者i的碰撞奖励
            - dist(0,i): 逃避者与追捕者i的距离
            - v0: 逃避者的速度（作为惩罚项）
        """
        global_rew = 0.0
        agents = self.good_agents(world)      # 逃跑者
        adversaries = self.adversaries(world)  # 围捕者
        # 如果没有逃跑者或者追捕者，直接返回0奖励
        if not agents or not adversaries:
            return 0.0
        
        # 获取全局设置的围捕阈值
        # capture_threshold = world.capture_threshold if hasattr(world, 'env') else world.world_size * 0.2
        
        # if world.communication_phase == 1: # 追捕阶段 - 使用黑板上的目标信息
        #     target_pos = world.blackboard['m_target']['target_pos']
        #     # 计算所有追捕者到黑板目标位置的距离
        #     distances = []
        #     for adv in adversaries:
        #         distance = np.linalg.norm(adv.state.p_pos - target_pos)
        #         distances.append(distance)
        #     # 计算平均距离
        #     mean_distance = np.mean(distances)
        # else:
        #     mean_distance = 0

        # 奖励分配 - 简化为三个基本阶段
        # 1. 已成功围捕 - 给予最高奖励
        if world.is_captured:
            global_rew += 100.0
            return global_rew
        
        # 2. 接近奖励
        # approach_reward = -(mean_distance / world.world_size)
        # global_rew += approach_reward

        # 时间压力删除
        # time_pressure = -0.01 * world.episode_steps
        # global_rew += time_pressure
        return global_rew

    # V4版本：限制观测范围
    def observation(self, agent, world):  # 返回值，自动适配智能体的观测空间维数
        """
            智能体及地标的观测空间
            TODO:需要按需重载。
        """
        # get positions of all entities in this agent's reference frame
        # 1. 自身状态归一化
        norm_self_vel = agent.state.p_vel / agent.max_speed
        norm_self_pos = agent.state.p_pos / world.world_size
        # 2. 障碍物信息
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:  # default: False ,要执行的代码块
                relative_entity_pos = (entity.state.p_pos - agent.state.p_pos) / (world.world_size)
                entity_pos.append(relative_entity_pos)  #表示地标相对于智能体的位置。这样返回的每个地标位置将是一个 2D 向量，表示该地标与智能体之间的相对位置。
        # 3. 逃跑者的信息 - 这部分是通信网络能够学习做出贡献的关键
        other_pos, other_vel = [], []  # 保留相对位置信息
        abs_target_pos = [] # 绝对位置信息 - 初始化为空列表
        target_abs_pos = np.zeros(world.dim_p)
        target_abs_vel = np.zeros(world.dim_p)
        # 筛选出追捕者，看逃跑者相关位置
        if agent.adversary:
            for other in world.agents: # 追捕者观察逃跑者信息
                if not other.adversary:  # 对于逃跑者
                    # 计算当前追捕者与逃跑者之间的距离
                    distance = np.linalg.norm(other.state.p_pos - agent.state.p_pos)
                    # 只有当逃跑者在观测范围内时，才能观测到它的实际位置和速度
                    if distance <= world.observation_radius + other.size:
                        # print("distance <= world.observation_radius")
                        world.communication_phase = 1 # 更新全局phase为追捕阶段
                        # 发现目标的智能体负责更新黑板
                        world.blackboard['m_target']['target_pos'] = other.state.p_pos.copy()
                        world.blackboard['m_target']['target_vel'] = other.state.p_vel.copy()
                        relative_pos = (other.state.p_pos - agent.state.p_pos) / world.world_size
                        norm_vel = other.state.p_vel / other.max_speed
                        # 更新绝对位置和速度
                        target_abs_pos = other.state.p_pos.copy() / world.world_size # 归一化
                        target_abs_vel = other.state.p_vel.copy() / other.max_speed  # 归一化

                    elif world.communication_phase == 1: #从其他agent获得的目标位置
                        target_pos = world.blackboard['m_target']['target_pos']# 从黑板获取目标位置
                        target_vel = world.blackboard['m_target']['target_vel']
                        relative_pos = (target_pos - agent.state.p_pos) / world.world_size # 计算相对位置
                        norm_vel = target_vel / other.max_speed
                        # 更新绝对位置和速度
                        target_abs_pos = target_pos.copy() / world.world_size # 归一化
                        target_abs_vel = target_vel.copy() / other.max_speed  # 归一化
                    else:
                        # 超出观测范围时，提供无效信息（零向量）   或特定值
                        relative_pos = np.zeros(world.dim_p)
                        norm_vel = np.zeros(world.dim_p)
                    other_pos.append(relative_pos)
                    other_vel.append(norm_vel)
                    # 将目标的绝对位置和速度添加到abs_target_pos
                    abs_target_pos = np.concatenate([target_abs_pos, target_abs_vel])
        return np.concatenate(
            [norm_self_vel]
            + [norm_self_pos]
            + entity_pos
            + other_pos
            + other_vel
            + [np.array([world.communication_phase])]
            + [abs_target_pos]  # 作为单个数组添加 实现star所添加 & 归一化 # TODO:根据效果看这个信息是否需要剔除critic的输入; 
        )
    
    def update_communication_phase(self, world): # 每一步都检查并更新通信阶段
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)  # 围捕者
        # 检查是否有任何追捕者能观测到目标
        any_adversary_can_observe = False
        for target in agents: # 目前只有一个
            for adv in adversaries:
                distance = np.linalg.norm(adv.state.p_pos - target.state.p_pos)
                if distance <= world.observation_radius + target.size:
                    any_adversary_can_observe = True
                    world.blackboard['m_target']['target_pos'] = target.state.p_pos.copy()
                    world.blackboard['m_target']['target_vel'] = target.state.p_vel.copy()
                    break
        
        if any_adversary_can_observe:
            # 有追捕者能观测到目标 -> 追捕阶段
            world.communication_phase = 1
        else:
            world.communication_phase = 0
            world.blackboard['m_target']['target_pos'] = None
            world.blackboard['m_target']['target_vel'] = None

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