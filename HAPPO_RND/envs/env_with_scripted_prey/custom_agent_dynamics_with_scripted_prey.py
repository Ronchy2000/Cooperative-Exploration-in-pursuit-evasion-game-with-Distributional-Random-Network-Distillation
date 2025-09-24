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
            self.target_pos = np.zeros(2)  # 目的地位置
            self.step_counter = 0          # 步数计数器
            self.reach_threshold = 0.1     # 到达判定阈值
            self.action_callback = self.scripted_behavior

    def scripted_behavior(self, agent, world):
        """简化且改进的逃跑者行为，保证逻辑合理且易于围捕"""
        # 获取追捕者信息
        pursuers = [a for a in world.agents if a.adversary]
        
        # 如果没有追捕者，随机游荡
        if not pursuers:
            if agent.step_counter % 30 == 0 or not hasattr(agent, 'target_pos'):
                agent.target_pos = np.random.uniform(-world.world_size * 0.8, world.world_size * 0.8, 2)
            
            direction = agent.target_pos - agent.state.p_pos
            distance = np.linalg.norm(direction)
            
            if distance > self.reach_threshold:
                agent.action.u = direction / (distance + 1e-8) * agent.u_range * 0.5
            else:
                agent.action.u = np.zeros(2)
                
            agent.step_counter += 1
            return agent.action
        
        # 计算是否被围困
        is_surrounded = self.check_if_surrounded(agent, pursuers, world)
        # 每20步或被围困时更新目标
        update_target = (agent.step_counter % 20 == 0) or is_surrounded
        # 计算到场地中心的距离
        center_dist = np.linalg.norm(agent.state.p_pos)
        # 1. 更新目标位置
        if update_target:
            # 获取追捕者平均位置
            mean_pos = np.mean([p.state.p_pos for p in pursuers], axis=0)
            # 选择远离平均位置的方向
            away_direction = agent.state.p_pos - mean_pos
            away_direction = away_direction / (np.linalg.norm(away_direction) + 1e-8)
            # 计算位置偏好：避免总是朝角落走
            # 如果已经靠近边界，增加向中心移动的概率
            center_pull = np.zeros(2)
            if center_dist > world.world_size * 0.7:
                center_pull = -agent.state.p_pos * 0.5  # 向中心拉力
            # 混合方向，使逃跑动作更多样化
            final_direction = away_direction + center_pull
            if np.linalg.norm(final_direction) > 0:
                final_direction = final_direction / np.linalg.norm(final_direction)
            # 根据是否被围困，调整目标距离
            if is_surrounded:
                # 被围困时，逃跑距离较近但更急迫
                distance = world.world_size * 0.5 * np.random.uniform(0.3, 0.7)
            else:
                # 未被围困时，逃跑距离更远
                distance = world.world_size * 0.7 * np.random.uniform(0.6, 1.0)
                # 20%概率向场地中心移动，防止总是贴边
                if np.random.random() < 0.2 and center_dist > world.world_size * 0.5:
                    final_direction = -agent.state.p_pos
                    final_direction = final_direction / (np.linalg.norm(final_direction) + 1e-8)
                    distance = center_dist * 0.7
            # 设置新的目标位置
            agent.target_pos = agent.state.p_pos + final_direction * distance
            # 确保目标在世界范围内
            agent.target_pos = np.clip(agent.target_pos, 
                                      -world.world_size * 0.9, 
                                      world.world_size * 0.9)
        # 2. 移动逻辑
        direction = agent.target_pos - agent.state.p_pos
        distance = np.linalg.norm(direction)
        
        if distance > self.reach_threshold:
            # 归一化方向向量
            normalized_direction = direction / (distance + 1e-8)
            # 速度因子调整
            if is_surrounded:
                # 被围困时速度降低，使围捕更容易成功
                speed_factor = 0.4 if np.random.random() < 0.7 else 0.6
            else:
                # 正常情况下的速度
                speed_factor = np.random.uniform(0.6, 0.8)
                # 如果在场地中心区域，速度可以更快
                if center_dist < world.world_size * 0.5:
                    speed_factor *= 1.0
            
            # 设置动作
            agent.action.u = normalized_direction * agent.u_range * speed_factor  # core.py 中 -> control range
            # 小幅随机扰动，使运动更自然
            if np.random.random() < 0.1:
                noise = np.random.normal(0, 0.05, 2)
                agent.action.u += noise * agent.u_range
                # 确保不超过最大速度
                speed = np.linalg.norm(agent.action.u)
                if speed > agent.u_range:
                    agent.action.u = agent.action.u * (agent.u_range / speed)
        else:
            agent.action.u = np.zeros(2, dtype=np.float32) # 维度为2的零向量、
        
        agent.step_counter += 1
        # [fixed bug]明确返回action对象, 否则返回None
        return agent.action # 维度为2的零向量、nt.action
    
    def check_if_surrounded(self, agent, pursuers, world):
        """检查逃跑者是否被围困的简化版本"""
        if len(pursuers) < 3:
            return False

        # 计算到所有追捕者的距离
        distances = [np.linalg.norm(p.state.p_pos - agent.state.p_pos) for p in pursuers]

        # 计算追捕者的角度分布
        if all(d < world.world_size * 0.4 for d in distances):
            angles = []
            for p in pursuers:
                rel_pos = p.state.p_pos - agent.state.p_pos
                angles.append(np.arctan2(rel_pos[1], rel_pos[0]))

            # 排序角度
            angles = sorted(angles)
            angles.append(angles[0] + 2 * np.pi)

            # 计算最大角度差
            max_angle_diff = max([angles[i+1] - angles[i] for i in range(len(angles)-1)])

            # 如果最大角度差小于150度(剩余210度已被覆盖)，认为被围困
            if max_angle_diff < np.pi * 5/6:
                return True
        return False


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
            agent.action = agent.action_callback(agent, self)  # 返回值为None，问题在这。
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
            # entity.state.p_pos = np.clip(entity.state.p_pos, -self.world_size, self.world_size) # Ronchy 添加世界大小限制            


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
