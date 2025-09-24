"""
该文件定义了自定义的环境，用于测试自定义的智能体动力学模型

继承自core.py

"""
import numpy as np
from pettingzoo.mpe._mpe_utils.core import EntityState, AgentState, Action, Entity, Landmark, Agent
from pettingzoo.mpe._mpe_utils.core import World

class CustomAgent(Agent):
    def __init__(self, is_scripted = False):
        super().__init__()
        # script behavior to execute
        self.action_callback = None
        self.is_scripted = is_scripted
        if self.is_scripted == True:
            self.action_callback = self.scripted_behavior

    def scripted_behavior(self, agent, world):
        """返回随机动作，从N(0,1)采样并限制在[-1,1]范围内"""
        # 从标准正态分布N(0,1)中采样
        random_action = np.random.normal(0, 1, 2)
        # 将动作限制在[-1,1]范围内
        random_action = np.clip(random_action, -1, 1)
        # 将随机动作赋值给agent的动作
        agent.action.u = random_action * agent.u_range  # self.u_range = 1.0
        return agent.action # 维度为2的零向量、nt.action

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
        # set actions for scripted agents
        # print("Using world -> step()") # 重载成功！
        for agent in self.scripted_agents:
            if agent.action_callback is not None:
                agent.action = agent.action_callback(agent, self)  # 返回值为None，问题在这。
            else:
                print("警告: scripted agent没有设置action_callback")

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
            entity.state.p_pos = np.clip(entity.state.p_pos, -self.world_size, self.world_size) # Ronchy 添加世界大小限制            
            # 限制位置在世界大小范围内 (取消注释并改用更平滑的方式)
            '''
                这两种边界限制的实现方式各有优缺点: 底层动力学方式, 奖励函数方式（给予超出边界惩罚）
            最好的方案是两种方法结合使用：
                底层动力学设硬性边界：确保所有智能体（包括脚本控制的）都在地图内活动
                奖励函数设软性惩罚：对接近边界的行为给予轻微惩罚，鼓励智能体远离边界 : 在奖励函数中，你可以添加一个"靠近边界"的小惩罚：
            '''

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
        force = self.contact_force * delta_pos / (dist+1e-8) * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
