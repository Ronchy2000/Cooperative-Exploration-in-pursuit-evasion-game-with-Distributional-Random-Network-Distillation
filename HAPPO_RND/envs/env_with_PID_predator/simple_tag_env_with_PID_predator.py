# from pettingzoo.mpe import simple_adversary_v3, simple_spread_v3, simple_tag_v3

# from pettingzoo.mpe.simple_tag_v3 import raw_env

import numpy as np
import gymnasium
from gymnasium.utils import EzPickle

from pettingzoo.mpe._mpe_utils.core import Agent, Landmark, World
from pettingzoo.mpe._mpe_utils.scenario import BaseScenario
from pettingzoo.mpe._mpe_utils.simple_env import SimpleEnv, make_env
from pettingzoo.utils.conversions import parallel_wrapper_fn

from .custom_agent_dynamics_with_PID_predator import CustomAgent, CustomWorld # Ronchy自定义的World类，用于测试自定义的智能体动力学模型

import pygame  #Ronchy: 用于渲染动画环境

# 用于检测包围条件
from scipy.spatial import ConvexHull, Delaunay

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
        self.local_ratio = 0.4  # 全局奖励和局部奖励的比例
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

    def reset_world(self, world, np_random):
        # 清除历史轨迹
        self.history_positions = {agent.name: [] for agent in self.world.agents}  
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
            if is_captured:
                for a in self.agents:
                    self.terminations[a] = True
            # 如果达到最大步数，标记 truncation 为 True
            if self.steps >= self.max_cycles:
                for a in self.agents:
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
            pygame.draw.circle(circle_surface, (0, 200, 0, 50), (int(x), int(y)), threshold_pixels, 2)  # 最后一个参数2是线宽
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

            radius = entity.size * 140 * scaling_factor

            if isinstance(entity, Agent):
             # 设置透明度：例如，transparent_alpha=128 (半透明)
                transparent_alpha = 200  # 透明度，范围从0（完全透明）到255（完全不透明）
                color = (0, 0, 255, transparent_alpha) if entity.adversary else (255, 0, 0, transparent_alpha)
                # 创建透明度支持的Surface
                agent_surface = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

                # pygame.draw.circle(self.screen, color, (int(x), int(y)), int(radius))
                pygame.draw.circle(agent_surface, color, (int(x), int(y)), int(radius))

                agent_surface.set_alpha(transparent_alpha)  # 设置透明度
                self.screen.blit(agent_surface, (0, 0))

                # pygame.draw.circle(self.screen, (0, 0, 0), (int(x), int(y)), int(radius), 1) # 绘制边框
                pygame.draw.circle(self.screen, (255, 255, 255), (int(x), int(y)), int(radius), 1) # 绘制边框
            else:  # Landmark
                pygame.draw.circle(self.screen, (128, 128, 128), (int(x), int(y)), int(radius))
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

    def check_capture_condition(self,threshold = None): # agent.size = 0.075 if agent.adversary else 0.05
        """
        检查所有围捕者是否都进入逃跑者的指定范围内。

        Args:
            threshold (float): 围捕者和逃跑者之间的最大允许距离。

        Returns:
            tuple: (is_captured, capture_info) - 是否围捕成功，以及围捕相关信息
                - is_captured: 布尔值，表示是否围捕成功
                - capture_info: 字典，包含：
                    - distances: 所有追捕者到逃跑者的距离数组
                    - pursuers_pos: 追捕者位置数组
                    - evader_pos: 逃跑者位置
                    - is_surrounded: 逃跑者是否被围住
        """
        if threshold is None:
            threshold = self.world_size * 0.2 # 使用世界大小的20%作为默认捕获范围
            # 初始化返回值
        distances, pursuers_pos, evader_pos, is_surrounded, is_captured, capture_info = 0, [], None, False, False, None
        agents = self.scenario.good_agents(self.world)  # 逃跑者
        adversaries = self.scenario.adversaries(self.world)  # 围捕者
        # 确保有足够的追捕者（至少3个追捕者才能形成围捕）
        if len(adversaries) < 3:
            return is_captured, capture_info  # 明确返回初始值

        # 逃跑者，计算所有围捕者与该逃跑者的距离
        for agent in agents:
            # 1. 收集所有追捕者的位置
            pursuers_pos = np.array([adv.state.p_pos for adv in adversaries])
            evader_pos = agent.state.p_pos
            
            # 2. 检查是否所有追捕者都在合理的围捕距离内
            # 如果有追捕者距离太远，则不视为有效围捕
            distances = np.linalg.norm(pursuers_pos - evader_pos, axis=1)
            max_allowed_distance = self.world_size * 0.4  # 围捕的最大距离
            if np.max(distances) > max_allowed_distance:
                continue  # 至少有一个追捕者太远，不构成围捕

            # 检查是否靠近边界，并添加虚拟边界追捕者
            virtual_pursuers = []
            boundary_margin = self.world_size * 0.1
            world_size = self.world_size

            # 检查每个坐标轴
            for i in range(len(evader_pos)):
                # 左/下边界
                if evader_pos[i] < -world_size + boundary_margin:
                    virtual_pos = evader_pos.copy()
                    virtual_pos[i] = -world_size  # 边界位置
                    virtual_pursuers.append(virtual_pos)

                # 右/上边界
                elif evader_pos[i] > world_size - boundary_margin:
                    virtual_pos = evader_pos.copy()
                    virtual_pos[i] = world_size  # 边界位置
                    virtual_pursuers.append(virtual_pos)
            
            is_surrounded = False
            # 如果有虚拟追捕者，合并实际和虚拟追捕者
            if virtual_pursuers:
                all_pursuers = np.vstack([pursuers_pos, np.array(virtual_pursuers)])
                # 至少需要两个点(实际或虚拟)形成包围
                if len(all_pursuers) >= 2:
                    # 创建追捕者位置的凸包
                    hull = ConvexHull(all_pursuers)
                    # 检查逃跑者是否在凸包内
                    delaunay = Delaunay(all_pursuers[hull.vertices])
                    is_surrounded = delaunay.find_simplex(evader_pos) >= 0
            # 在开阔地带，使用原有的凸包检测. 使用凸包算法判断逃跑者是否在追捕者形成的多边形内部
            else:
                # 创建追捕者位置的凸包
                hull = ConvexHull(pursuers_pos)
                # 检查逃跑者是否在凸包内
                delaunay = Delaunay(pursuers_pos[hull.vertices])
                is_surrounded = delaunay.find_simplex(evader_pos) >= 0

             # 围捕条件：被围住且至少有一个追捕者足够近
            is_captured = is_surrounded and np.min(distances) < threshold
            capture_info = {# 返回围捕状态和相关信息
                'distances': distances,
                'pursuers_pos': pursuers_pos,
                'evader_pos': evader_pos,
                'is_surrounded': is_surrounded
            }
            # 如果捕获成功，立即返回
            if is_captured:
                break
        return is_captured, capture_info

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
        
        # 创建PID控制的追捕者和学习型逃跑者
        agent_adversaries = [CustomAgent(is_pid_controlled = True) for i in range(num_adversaries)]
        # 逃跑者不使用脚本控制，会由DDPG算法控制
        agent_good_agents = [CustomAgent(is_pid_controlled = False) for i in range(num_good)]
        world.agents = agent_adversaries + agent_good_agents

        for i, agent in enumerate(world.agents):
            agent.adversary = True if i < num_adversaries else False
            base_name = "adversary" if agent.adversary else "agent"
            base_index = i if i < num_adversaries else i - num_adversaries
            agent.name = f"{base_name}_{base_index}"
            agent.collide = True
            agent.silent = True
            base_size = _world_size * 0.1 # 基础大小为世界大小的10%
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
        # 重置PID控制器状态
        for agent in world.agents:
            if hasattr(agent, 'is_pid_controlled') and agent.is_pid_controlled:
                if hasattr(agent, 'pid_controller'):
                    agent.pid_controller.reset()
                agent.target_pos = None

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
        """为逃跑者提供奖励，以便DDPG学习有效的逃跑策略"""
        rew = 0
        # 获取所有追捕者
        pursuers = self.adversaries(world)
        # 1. 存活奖励 - 鼓励逃跑者存活
        rew += 0.1
        # 2. 距离奖励 - 鼓励与追捕者保持距离
        distances = []
        for pursuer in pursuers:
            distance = np.linalg.norm(agent.state.p_pos - pursuer.state.p_pos)
            distances.append(distance)
        # 使用最短距离计算奖励
        min_dist = min(distances) if distances else 0
        rew += 0.5 * min(min_dist / world.world_size, 1.0)
        # 3. 避免被围捕惩罚
        if hasattr(world, 'is_captured') and world.is_captured:
            rew -= 10.0  # 被捕获的强烈惩罚
        # 4. 边界惩罚 - 避免靠近边界
        def boundary_penalty(x):
            if abs(x) > world.world_size * 0.8:
                return 2.0 * (abs(x) - world.world_size * 0.8) / (world.world_size * 0.2)
            return 0
        # 对x和y坐标分别计算边界惩罚
        for p in range(world.dim_p):
            rew -= boundary_penalty(agent.state.p_pos[p])
        return rew

    # 围捕者reward设置
    def adversary_reward(self, agent, world):
        # Adversaries are rewarded for collisions with agents
        rew = 0
        agents = self.good_agents(world)
        adversaries = self.adversaries(world)
        # 1. 接近目标奖励 - 鼓励追捕者接近逃跑者
        for target in agents:  # 通常只有一个逃跑者
            # 计算与逃跑者的距离
            distance = np.linalg.norm(agent.state.p_pos - target.state.p_pos)
            # 归一化距离系数
            dist_factor = distance / world.world_size

            # 计算速度方向与目标方向的一致性
            direction_to_target = target.state.p_pos - agent.state.p_pos
            direction_to_target = direction_to_target / (np.linalg.norm(direction_to_target) + 1e-6)
            velocity_direction = agent.state.p_vel / (np.linalg.norm(agent.state.p_vel) + 1e-6)

            # 速度方向与目标方向的点积（1表示完全一致，-1表示完全相反）
            alignment = np.dot(direction_to_target, velocity_direction)

            # 接近奖励：距离越近、方向越一致，奖励越高
            approach_reward = 0.5 * (1 - dist_factor) + 0.5 * alignment
            rew += approach_reward
        # 2. 碰撞奖励 - 如果碰撞到逃跑者则给予高额奖励
            if self.is_collision(agent, target):
                rew += 1.0
        
        def bound(x):
            boundary_start = world.world_size * 0.96 
            full_boundary = world.world_size
            if x < boundary_start:
                return 0
            if x < full_boundary:
                return (x - boundary_start) * 10
            return min(np.exp(2 * x - 2 * full_boundary), 10)
        # ==== 必须添加实际边界计算 ====
        for p in range(world.dim_p):  # 遍历每个坐标轴 (x, y)
            x = abs(agent.state.p_pos[p])  # 获取坐标绝对值
            rew -= bound(x)  # 应用边界惩罚函数
        
        return rew
    
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
        capture_threshold = world.capture_threshold if hasattr(world, 'env') else world.world_size * 0.2
        # 安全地获取捕获信息
        distances, pursuers_pos, target_pos = None, None, None
        # 已有围捕信息，直接使用
        target = agents[0] # 获取第一个逃跑者
        target_pos = target.state.p_pos
        target_speed = np.linalg.norm(target.state.p_vel)
        pursuers_pos = np.array([adv.state.p_pos for adv in adversaries])
        distances = np.array([np.linalg.norm(p - target_pos) for p in pursuers_pos])
        # 计算平均距离和归一化
        mean_distance = np.mean(distances)
        # 尝试检查目标是否在追捕者形成的凸包内
        is_target_surrounded = False
        if len(adversaries) >= 3:  # 至少需要3个追捕者才能形成一个包围圈
            # 创建追捕者位置的凸包
            hull = ConvexHull(pursuers_pos)
            # 创建Delaunay三角剖分
            delaunay = Delaunay(pursuers_pos[hull.vertices])
            # 检查逃跑者是否在凸包内
            is_target_surrounded = delaunay.find_simplex(target_pos) >= 0 
            # delaunay.find_simplex(target_pos)返回值表示的是三角形的索引，而不是坐标值。即检查目标点在第几个被划分的三角形中。可能是0..n,如果都找不到，则返回 -1
            # delaunay.find_simplex(target_pos) >= 0 返回bool值 


        # 第1阶段：追踪阶段 - 当追捕者距离逃跑者较远时
        if np.mean(distances) > world.world_size * 0.6:
            mean_dist = np.mean(distances) / world.world_size
            global_rew += 1.0 * (1 - mean_dist)
        # 第2阶段：包围阶段 - 追捕者开始形成包围圈
        elif not is_target_surrounded:  # 如果目标还没被包围
            # 接近奖励 - 鼓励更接近目标
            approach_reward = 0.5 * (1.0 - mean_distance / world.world_size)
            global_rew += approach_reward
            # 准备包围奖励 - 追捕者之间的距离不要太靠近
            if len(adversaries) >= 3:
                # 计算追捕者之间的最小距离
                min_distance_between_pursuers = float('inf')
                for i in range(len(pursuers_pos)):
                    for j in range(i+1, len(pursuers_pos)):
                        dist = np.linalg.norm(pursuers_pos[i] - pursuers_pos[j])
                        min_distance_between_pursuers = min(min_distance_between_pursuers, dist)
                # 如果追捕者之间距离过近，给予惩罚，鼓励分散
                optimal_separation = world.world_size * 0.3  # 理想的追捕者间距
                if min_distance_between_pursuers < optimal_separation:
                    separation_penalty = 0.3 * (1.0 - min_distance_between_pursuers / optimal_separation)
                    global_rew -= separation_penalty
        # 第3阶段：捕获阶段 - 目标已在凸包内，现在需要靠近捕获
        else:
            # 给予包围成功的基础奖励
            encirclement_reward = 2.0
            global_rew += encirclement_reward

            # 如果已经正式捕获（至少一个追捕者足够近）
            if hasattr(world, 'is_captured') and world.is_captured:
                capture_reward = 10.0
                global_rew += capture_reward # 围捕成功奖励
            # 处于捕获阶段但尚未成功时的奖励
            else:
                # 缩小包围圈的奖励 - 鼓励追捕者平均距离减小
                proximity_reward = 1.0 * (1.0 - mean_distance / (world.world_size * 0.5))
                global_rew += proximity_reward

                # 防止目标逃脱的奖励 - 至少有一个追捕者足够近
                closest_distance = np.min(distances)
                if closest_distance < capture_threshold * 1.5:
                    close_reward = 1.0 * (1.0 - closest_distance / capture_threshold)
                    global_rew += close_reward
        # ========================
        # 全局惩罚项（适用于所有阶段）
        # ========================
        # 逃跑者速度惩罚：逃跑者速度越快，惩罚越大
        # speed_penalty = 0.5 * (target_speed / target.max_speed)
        # global_rew -= speed_penalty
        return global_rew
    
    def observation(self, agent, world):  # 返回值，自动适配智能体的观测空间维数
        """
            智能体及地标的观测空间
            TODO:需要按需重载。
        """
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:
            if not entity.boundary:  # default: False ,要执行的代码块
                # 归一化相对位置
                relative_entity_pos = (entity.state.p_pos - agent.state.p_pos) / world.world_size
                entity_pos.append(relative_entity_pos)  #表示地标相对于智能体的位置。这样返回的每个地标位置将是一个 2D 向量，表示该地标与智能体之间的相对位置。
        # communication of all other agents
        comm = [] # default: self.c = None
        other_pos = [] # 相对位置：其他智能体（包扩逃跑者）相对于智能体的位置
        other_vel = [] # 逃跑者的速度
        for other in world.agents:
            if other is agent:
                continue
            comm.append(other.state.c / world.world_size)  # default: self.c = None
            
            # 归一化相对位置
            relative_pos = (other.state.p_pos - agent.state.p_pos) / world.world_size
            other_pos.append(relative_pos)
            if not other.adversary:  # adversary，追逐者的值是True，逃跑者的值是False
                norm_vel = other.state.p_vel / other.max_speed
                other_vel.append(norm_vel)
        # 自身状态归一化
        norm_self_vel = agent.state.p_vel / agent.max_speed
        norm_self_pos = agent.state.p_pos / world.world_size
        return np.concatenate(
            [norm_self_vel]
            + [norm_self_pos]
            + entity_pos
            + other_pos
            + other_vel
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