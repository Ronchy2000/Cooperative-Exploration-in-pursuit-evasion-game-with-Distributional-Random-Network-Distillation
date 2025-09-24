# 说明
## env_with_scripted_prey

> 追捕者由policy network控制，逃跑者自动脚本化执行，每过20个time step更换目的地，并执行动作。

## env_with_random_prey

> 追捕者由policy network控制，逃跑者自动脚本化执行，action来源于高斯分布采样，并执行动作。

## env_with_stationary_prey

> 追捕者由policy network控制，逃跑者静止，action = 0。

## env_with_PID_prey

> 追捕者由pid控制，逃跑者由policy network控制。

根目录下的：
`./custom_agents_dynamic.py` 是自定义的智能体 <br>
`./simple_tag_env.py` 是自定义的环境 <br>
搭配`legacy_mpe_official`下的算法使用 <br>

逃跑者和追捕者都是 通过Actor网络 进行控制。——all learning

### 2025.8.3
iddpg，maddpg 的baseline环境: 
- expert : envs/simple_tag_env_v4_partially.py文件
- stationary : envs/env_with_stationary_prey/simple_tag_env_with_stationary_prey_V4_partially_Perception.py
- random : 

### 2025.7.26 记录目录结构：
根据逃跑者的策略分：
1. stationary; 逃跑者是静止.<br>
envs: [env_with_stationary_prey](./envs/env_with_stationary_prey) <br>
algorithm: [maddpg_countbased_communication](../agents/maddpg_countbased_communication) 
    - agents/maddpg_countbased_communication/MADDPG_stationary_agent_star.py 
    - agents/maddpg_countbased_communication/runners/stationary_prey_runner_star.py
    - (不带star的是: 分布式计数 + 环形通信)
2. random; 逃跑者是随机游动.
envs: [env_with_randomStep_prey](.envs/env_with_randomStep_prey) <br>
algorithm: [maddpg_countbased_communication](../agents/maddpg_countbased_communication) 
    - agents/maddpg_countbased_communication/MADDPG_random_agent_star.py
    - agents/maddpg_countbased_communication/runners/random_prey_runner_star.py
    - ~~(不带star的是: 分布式计数 + 环形通信)~~这个貌似我没改 直接到count + comm;
3. DDPG - all learning; 逃跑者的策略是pretrained DDPG网络，TODO:把他的奖励调整一下.
envs: simple_tag_env_v5_partially_2Phase.py + custom_agents_dynamics.py
algorithm: []
# 2025.7.13
> v5_partially_2Phase-Env特性：
- [x] 2025.7.13修改：observation，添加world.communication_phase（但没置0转换）；添加“黑板”blackboard，写逃跑者信息； 修改adversary_reward
- [x] 待修改global_reward; communication_phase置0转换问题 （2025.7.14）
- [x] 添加分布式simhash计数功能 (TODO:待测试) [fixBug]: intrinsic reward:bonus维度与extrinsic reward维度不匹配，相加维度报错。
    - [2025.7.19fixBug]: 加入计数表reset; intrinsic_reward更新时避免除以0; 
- [x] info processor功能：  
    - 输入：发送智能体观测 $$o_\text{star center}$$,即输入为 $$m_{target}$$ 也即 $$m_\text{star center}$$ （这里是trick是，把m_target设置成一个公共变量，每个agent可以赋值or读取，发现逃跑者的agent赋值，其他agent可以读取，所以是star型拓扑。（ $$m_{target}$$ 已用黑板系统实现～
    - 输出：消息 $$\tilde{m_i}$$ (通过tanh激活限制在[-1,1]范围)
- [x] ~~修改MADDPG-count-comm算法：~~ 不需要；obs中已经包含了phase信息！
    1. replay buffer需要加入phase 0 or 1；
- [x] 下一步，梳理DDPG中的维度问题，现在需要把维度梳理清楚(应该是在stationary中修改，我就是怎么一直不起作用，这个教训要记住：根目录的环境是all-learing的即逃跑者也用DDPG策略)
- [x] stationaty_comm 方法开发结束

# 2025.7.9
> V4-Env特性:
- [x] - [x] 局部可观测 - 观测空间修改！
- [x] fix 渲染地图边框及rgb渲染
- [x] add debug_collision_rendering函数，查看碰撞的边界接触效果。


```python
    if self.render_mode == "human": # 在渲染模式为human时才调用flip
        pygame.display.flip()
```
```python
    # 绘制边框 - 为了保证边框渲染正确，手动调整像素值
            
    #这个问题与pygame中如何渲染线条有关。当绘制宽度为1像素的线条或矩形边框时，pygame会将线条放置在像素网格上，而不是像素之间。
    #由于计算机图形中的坐标通常从像素的左上角开始，这可#能导致某些边在某些环境下看起来"虚化"或不明显。
            
    pygame.draw.rect(self.screen, (0, 0, 0), 
                        (margin-1, margin-1, plot_width+1, plot_height+1), 1)
```



# 2025.4.26
- [x] 局部可观测。


# 2025.4.25
> V3环境修改适配

需要完成功能：

- [x] 捕获条件设置 -> 2个及以上的进入阈值。
- [ ] 加入逃跑者碰撞减速。
- [ ] 弹性碰撞力，得取消掉 or 减小作用 ，这个会影响平稳性。
- [ ] ~~在某一个小区域中reset~~




# 2025.4.19:
`./simple_tag_env.py` 存在问题。现在更新为`./simple_tag_env_v2.py`

## V2环境特性：
- 连续状态空间，连续动作空间
- 地图大小：2.5*2 x 2.5*2
- 动作限制:[-1, 1]
- 全局可观测环境
- 所有agent由policy network控制
- individual reward *0.5 + global reward * 0.5
- 捕获成功条件：三角形围捕。
- 更改奖励：1. 添加时间压力  2. 添加面积判断，鼓励encircle。 3.成功捕获奖励->100
- 更改围捕阈值：~~0.3*worldsize~~（暂未更改。——智能体size变小，相当于0.2的围捕阈值可以了  -> `变到0.25倍`


## 具体修复内容：

0. fix bug： `clip_grad_norm` -> `clip_grad_norm_`

    在较新版本的PyTorch中， `clip_grad_norm` 已被弃用，推荐使用 `clip_grad_norm_` 。
    在你的代码中，使用的是 `clip_grad_norm_`，这是正确的做法，它会直接修改模型参数的梯度，限制其范数不超过0.5，有助于防止梯度爆炸问题。

1. 渲染问题修复

    原版的渲染有问题，主要集中在：
围捕圈内显示`is_collision`判断！！！！。。
在evaluate时，在`is_collision`中加入打印，确实会发现问题！
- 在物理世界中：当两个智能体相距 `agent1.size + agent2.size `时发生碰撞
- 在渲染中：智能体看起来半径是 `agent.size * 140 * scaling_factor` 像素
问题在于这两种计算方式不成比例，导致即使物理上已经碰撞了，视觉上还有很大距离

    修改方式：
    ```
    # radius = entity.size * 140 * scaling_factor # 原始
    # # 修改为：根据世界到屏幕的实际转换比例计算
    world_to_screen_scale = (self.width / (2 * self.world_size)) * 0.9
    radius = entity.size * world_to_screen_scale
    ```



2. 捕获判定条件修复。
    使用三角形面积判定方式。 （局限性，只能适配3个追捕者
    ```
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

    ```
    ```
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
    ```

3. reward 距离奖励修复。
    ```
    # 修改部分：距离奖励：使用非线性奖励，让接近时奖励增长更快——错误的！
    # distance_reward = 0.5 * (dist_factor ** 2)  # 使用二次函数放大近距离奖励  #如果超出了边界，离逃跑者越远，高
    if distance > world.capture_threshold:
        approach_reward = -dist_factor
        rew += approach_reward
    else:
        rew += 1.0  # 当距离小于阈值时，给予最大
    ```


# # 2025.4.xx:

继续更新环境设定`./simple_tag_env_v3.py`
待更新：

- 弹性碰撞力，得取消掉，这个会影响平稳性。
- 加入逃跑者碰撞减速。
