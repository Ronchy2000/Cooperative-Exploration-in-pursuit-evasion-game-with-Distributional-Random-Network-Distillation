"""
该文件定义了自定义的环境，用于测试自定义的智能体动力学模型

继承自core.py

"""
import numpy as np
from pettingzoo.mpe._mpe_utils.core import EntityState, AgentState, Action, Entity, Landmark, Agent
from pettingzoo.mpe._mpe_utils.core import World



class PIDController:
    """PID控制器实现，用于控制围捕者的行为"""
    def __init__(self, kp=1.0, ki=0.0, kd=0.5, dt=0.1, max_output=1.0):
        self.kp = kp  # 比例增益
        self.ki = ki  # 积分增益
        self.kd = kd  # 微分增益
        self.dt = dt  # 时间步长
        self.max_output = max_output  # 输出限幅
        
        self.prev_error = np.zeros(2)  # 前一步误差
        self.integral = np.zeros(2)    # 积分项
        
    def reset(self):
        """重置PID控制器状态"""
        self.prev_error = np.zeros(2)
        self.integral = np.zeros(2)
    
    def compute(self, target_pos, current_pos, current_vel=None):
        """计算控制输出"""
        # 计算当前误差（目标位置-当前位置）
        error = target_pos - current_pos
        # 积分项
        self.integral += error * self.dt
        # 微分项（如果有当前速度，直接使用负速度作为微分项）
        if current_vel is not None:
            derivative = -current_vel  # 使用负速度作为微分项
        else:
            derivative = (error - self.prev_error) / self.dt
        # 计算PID输出
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        # 限制输出范围
        output = np.clip(output, -self.max_output, self.max_output)
        # 更新前一步误差
        self.prev_error = error.copy()
        return output
    

class CustomAgent(Agent):
    def __init__(self,is_pid_controlled=False):
        super().__init__()
        # 脚本行为执行
        self.action_callback = None
        self.is_pid_controlled = is_pid_controlled        
        # 初始化PID控制器
        if self.is_pid_controlled:
            # 每个维度一个PID控制器
            self.pid_controller = PIDController(kp=1.2, ki=0.0, kd=0.8)
            self.action_callback = self.pid_behavior

            
        # 目标位置缓存（用于PID控制器）
        self.target_pos = None

    def set_target(self, target_pos):
        """设置PID控制器的目标位置"""
        self.target_pos = target_pos
    
    def pid_behavior(self, agent, world):
        """基于PID控制器的行为"""
        # 如果没有目标，寻找逃跑者作为目标
        if self.target_pos is None:
            # 找到逃跑者
            for other in world.agents:
                if not other.adversary:  # 逃跑者
                    self.target_pos = other.state.p_pos
                    break
        
        # 如果有目标，计算控制信号
        if self.target_pos is not None:
            # 计算PID控制输出
            control_signal = self.pid_controller.compute(
                self.target_pos,
                agent.state.p_pos,
                agent.state.p_vel
            )
            
            # 设置agent动作
            agent.action.u = control_signal
        else:
            # 如果没有目标，则随机移动
            agent.action.u = np.random.uniform(-0.5, 0.5, 2)
        
        return agent.action

class CustomWorld(World):
    def __init__(self, world_size = 2.5 ): #
        super().__init__() # 调用父类的构造函数
        self.world_size = world_size # Ronchy 添加世界大小
        self.capture_threshold = world_size * 0.2
        self.is_captured = False
        self.dt = 0.1 # 时间步长
        self.damping = 0.2 # 阻尼系数
        # contact response parameters
        self.contact_force = 1e2 # 控制碰撞强度（默认1e2，值越大反弹越强）
        self.contact_margin = 1e-3 # 控制碰撞"柔软度"（默认1e-3，值越小越接近刚体）
        """
        常见问题示例
        实体重叠穿透	contact_force太小	增大contact_force至1e3或更高
        碰撞后震荡	damping太低	增大阻尼系数（如0.5）
        微小距离抖动	contact_margin不合理	调整到1e-2~1e-4之间
        """
    """ 
        重载底层动力学逻辑
        主要是integrate_state()函数
    """
    def step(self):
        """重载step方法，实现PID控制器的更新"""
        # 更新PID控制器的目标
        self.update_pid_targets()
        pid_agents = [agent for agent in self.agents if hasattr(agent, 'is_pid_controlled') and agent.is_pid_controlled]
        for agent in pid_agents:
            if agent.action_callback is not None:
                agent.action = agent.action_callback(agent, self)
            else:
                print("警告: PID智能体没有设置action_callback")

        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force) # 加入噪声
        # apply environment forces
        p_force = self.apply_environment_force(p_force) # 碰撞力计算 collide为True时
        # integrate physical state
        self.integrate_state(p_force) # 动力学逻辑
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent) # 更新 communication action 后的状态
    
    def update_pid_targets(self):
        """更新PID控制器的目标位置"""
        # 获取所有逃跑者
        evaders = [agent for agent in self.agents if not agent.adversary]
        # 获取所有PID控制的围捕者
        pid_pursuers = [agent for agent in self.agents if agent.adversary and getattr(agent, 'is_pid_controlled', False)]
        
        if not evaders or not pid_pursuers:
            return
        
        # 默认以第一个逃跑者为目标
        evader = evaders[0]
        evader_pos = evader.state.p_pos
        
        # 如果只有一个追捕者，直接追踪目标
        if len(pid_pursuers) == 1:
            pid_pursuers[0].set_target(evader_pos)
            return
        
        # 多个追捕者时，实现围捕行为
        num_pursuers = len(pid_pursuers)
        for i, pursuer in enumerate(pid_pursuers):
            # 计算围绕目标的位置
            angle = 2 * np.pi * i / num_pursuers
            # 围捕半径（随距离动态调整）
            dist_to_evader = np.linalg.norm(pursuer.state.p_pos - evader_pos)
            encirclement_radius = min(dist_to_evader * 0.8, self.capture_threshold * 1.5)
            # 计算目标位置
            offset = encirclement_radius * np.array([np.cos(angle), np.sin(angle)])
            target_pos = evader_pos + offset
            # 限制在世界范围内
            target_pos = np.clip(target_pos, -self.world_size, self.world_size)
            # 设置追捕者的目标
            pursuer.set_target(target_pos)
    
    # integrate physical state
    #函数功能：动力学逻辑。更新实体的位置和速度
    def integrate_state(self, p_force):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            # 速度阻尼衰减
            entity.state.p_vel *= (1 - self.damping)  # 正确应用阻尼
             # 动力学 -> 运动学
            if p_force[i] is not None:
                acceleration = p_force[i] / entity.mass # F = ma
                entity.state.p_vel += acceleration * self.dt # v = v_0 + a * t

            # 速度限幅
            if entity.max_speed is not None:
                speed = np.linalg.norm(entity.state.p_vel) # 计算向量模长
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel * (entity.max_speed / speed) # 向量缩放

            # 更新位置
            '''
            从经典物理学的角度来看，位移计算公式应该是 s = v₀t + ½at²，但在这里我们使用了简化的欧拉积分方法：
            这实际上是一种简化的欧拉积分方法，而不是精确的物理公式。在游戏和模拟环境中，这种简化是很常见的，因为：
            1. 计算效率高 ：简化的欧拉积分计算量小
            2. 时间步长小 ：当dt很小时，误差可以接受
            3. 简化实现 ：代码更简洁易懂
            '''
            entity.state.p_pos += entity.state.p_vel * self.dt  # 更新位置

            # 限制位置在世界大小范围内
            # entity.state.p_pos = np.clip(entity.state.p_pos, -self.world_size, self.world_size) # Ronchy 添加世界大小限制            
            # 限制位置在世界大小范围内 (取消注释并改用更平滑的方式)
            '''
                这两种边界限制的实现方式各有优缺点: 底层动力学方式, 奖励函数方式（给予超出边界惩罚）
            最好的方案是两种方法结合使用：
                底层动力学设硬性边界：确保所有智能体（包括脚本控制的）都在地图内活动
                奖励函数设软性惩罚：对接近边界的行为给予轻微惩罚，鼓励智能体远离边界 : 在奖励函数中，你可以添加一个"靠近边界"的小惩罚：
            '''
            old_pos = entity.state.p_pos.copy()
            entity.state.p_pos = np.clip(entity.state.p_pos, -self.world_size, self.world_size)

            # 如果位置被裁剪，则反向速度以模拟反弹
            if not np.array_equal(old_pos, entity.state.p_pos):
                # 确定哪个维度被裁剪了
                clip_mask = (old_pos != entity.state.p_pos)
                # 只在被裁剪的维度上反向速度
                entity.state.p_vel[clip_mask] = -entity.state.p_vel[clip_mask] * 0.5  # 添加一些能量损失

    # get collision forces for any contact between two entities
    # TODO: 碰撞逻辑待细化
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos))) #用norm更简洁
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size  # 两个实体的半径之和
        # softmax penetration
        k = self.contact_margin 
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k  #渗透深度， 当 dist < dist_min 时产生虚拟渗透量
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
