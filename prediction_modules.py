import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from common_utils import *


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        # 位置索引（0到max_len-1），形状 [max_len, 1]
        position = torch.arange(max_len).unsqueeze(1)
        
        # 计算位置编码的除数项（用于控制频率衰减速度）
        # 公式：exp( ( -ln(10000) / d_model ) * i ) 其中i是偶数索引
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        # 初始化位置编码矩阵 [max_len, 1, d_model]
        pe = torch.zeros(max_len, 1, d_model)
        
        # 生成正弦波位置编码（偶数位置），position * div_term 是位置索引乘以除数项
        # position 的维度为 [100, 1]，div_term 的维度为 [128]，相乘之后的维度为 [100, 128]
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 0::2 表示从0开始每隔2个取一个
        
        # 生成余弦波位置编码（奇数位置）
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 1::2 表示从1开始每隔2个取一个
        
        # 调整维度顺序为 [1, max_len, d_model] 
        # 便于后续与输入张量相加时的广播机制
        pe = pe.permute(1, 0, 2)
        
        # 注册为不可训练参数（模型保存/加载时会自动处理）
        self.register_buffer('pe', pe)
        
        # 初始化dropout层（训练时随机丢弃部分位置编码信息）
        # 在训练时，通过随机丢弃部分位置编码的信息，模型可以学习到更鲁棒的特征表示，
        # 避免对某些特定的位置编码过于敏感。而在测试或推理时，dropout层通常是不启用的，
        # 因此不影响模型的最终表现。
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        """ 将位置编码信息与输入特征相加
        输入x形状: [batch_size, seq_len, d_model]
        位置编码self.pe形状: [1, max_len, d_model]，通过广播机制自动扩展
        self.pe的形状是[1, max_len, d_model]，而输入x的形状可能是[batch_size, seq_len, d_model]。
        当它们相加时，PyTorch会自动将self.pe扩展为[batch_size, seq_len, d_model]，
        这样每个批次和序列位置都能加上相同的位置编码。
        
        广播机制的好处：
        可能还需要对比不使用广播机制的情况，比如手动扩展self.pe的形状，
        这样会增加代码复杂性和内存使用。广播机制在这里简化了操作，使代码更简洁高效。
        
        """
        x = x + self.pe # 广播机制
        
        # 应用dropout正则化（训练模式时随机丢弃部分位置信息，防止过拟合）
        # 在评估模式(eval)下，dropout层会自动关闭
        return self.dropout(x)


class AgentEncoder(nn.Module):
    """
    智能体编码器
    输入：智能体的运动数据（位置、速度等状态信息）
    输出：智能体的运动特征表示
    该编码器使用LSTM网络提取智能体的运动特征，
    并将整个运动序列的时序信息进行编码，
    捕获智能体的运动模式和行为特征。
    """
    def __init__(self, agent_dim):
        # 调用父类AgentEncoder的构造函数
        # 调用父类AgentEncoder的构造函数
        super(AgentEncoder, self).__init__()
        # 初始化LSTM网络用于提取智能体运动特征
        # 参数说明：
        # agent_dim: 输入特征维度（包含位置、速度等状态信息）
        # 256: 隐藏层维度
        # 2: LSTM层数
        # batch_first=True: 输入数据格式为 (batch_size, seq_len, feature_size)
        self.motion = nn.LSTM(agent_dim, 256, 2, batch_first=True)

    def forward(self, inputs):
        # 调用motion方法处理输入，得到轨迹traj和其他信息（此处未使用）
        # inputs形状: [batch_size, seq_len, agent_dim] 输入序列数据
        # traj形状: [batch_size, seq_len, 256] LSTM输出的完整轨迹特征
        traj, _ = self.motion(inputs)

        # 提取轨迹traj的最后一行作为输出
        # 输出形状: [batch_size, 256] 取最后一个时间步的特征表示
        # 该特征捕获了智能体整个运动序列的时序信息
        output = traj[:, -1]

        return output
    

class VectorMapEncoder(nn.Module):
    def __init__(self, map_dim, map_len):
        super(VectorMapEncoder, self).__init__()
        # 地图元素编码网络（车道线/斑马线等）
        # 结构：map_dim -> 64 -> 128 -> 256
        # 作用：提取单个地图点的局部特征
        self.point_net = nn.Sequential(
            nn.Linear(map_dim, 64),  # 将原始特征映射到64维
            nn.ReLU(),               # 引入非线性
            nn.Linear(64, 128),      # 特征升维
            nn.ReLU(),
            nn.Linear(128, 256)      # 输出256维点特征
        )
        
        # 位置编码器（为点序列添加位置信息）
        # max_len: 单个地图元素的最大点数（如车道线50个点）
        # 使用正弦位置编码保留点的顺序信息
        self.position_encode = PositionalEncoding(max_len=map_len)

    def segment_map(self, map, map_encoding):
        # 输入参数说明：
        # map: 原始地图数据 [B, N_e, N_p, map_dim]
        # map_encoding: 经过编码的地图特征 [B, N_e, N_p, D]
        
        B, N_e, N_p, D = map_encoding.shape  # 获取维度信息
        # 特征池化操作（沿点序列维度进行下采样）
        # 1. 维度置换为 [B, D, N_e, N_p] 适配二维池化操作
        # 2. 使用1x10最大池化窗口，每10个相邻点取最大值（相当于步长10的下采样）
        # 3. 池化后恢复为 [B, N_e, new_N_p, D] 的维度排列
        map_encoding = F.max_pool2d(map_encoding.permute(0, 3, 1, 2), kernel_size=(1, 10))
        map_encoding = map_encoding.permute(0, 2, 3, 1).reshape(B, -1, D)  # 重塑为 [B, total_segments, D]

        # 生成地图掩码（标识无效地图点）
        # 1. 检测原始地图数据中的零值位置（无效点）[B, N_e, N_p]
        # 2. 将掩码重塑为 [B, N_e, N_p//10, 10] 的分组形式
        # 3. 沿最后一个维度取最大值，得到每个分组的掩码状态 [B, N_e, N_p//10]
        # 4. 最终重塑为 [B, total_segments] 的二维掩码
        map_mask = torch.eq(map, 0)[:, :, :, 0].reshape(B, N_e, N_p//10, N_p//(N_p//10))
        map_mask = torch.max(map_mask, dim=-1)[0].reshape(B, -1)

        return map_encoding, map_mask  # 返回下采样后的地图特征和对应掩码

    def forward(self, input):
        # 地图编码处理流程：
        # 1. 通过点特征网络提取局部特征 [B, N_e, N_p, map_dim] => [B, N_e, N_p, 256]
        # 2. 添加位置编码信息（保留点序列的顺序特征）
        output = self.position_encode(self.point_net(input))
        
        # 3. 进行地图分段和下采样操作
        # 返回：
        # encoding - 下采样后的地图特征 [B, total_segments, 256]
        # mask - 无效段标识 [B, total_segments]（True表示该段地图数据无效）
        encoding, mask = self.segment_map(input, output)

        return encoding, mask


class CrossAttention(nn.Module):
    def __init__(self, heads=8, dim=256, dropout=0.1):
        # 跨模态注意力机制初始化
        super(CrossAttention, self).__init__()
        
        # 核心注意力模块配置
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=dim,        # 输入特征维度256
            num_heads=heads,      # 8头注意力机制（256/8=32）
            dropout=dropout,      # 注意力权重0.1的随机丢弃率
            batch_first=True      # 输入格式为(batch, seq, feature)
        )
        
        # 归一化层配置（Transformer标准结构），促进训练的稳定性
        self.norm_1 = nn.LayerNorm(dim)  # 首层归一化（稳定注意力输出）
        self.norm_2 = nn.LayerNorm(dim)  # 第二层归一化（稳定前馈输出）
        
        # 前馈神经网络（Position-wise FFN）
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim*4),  # 特征升维（256->1024）
            nn.GELU(),             # 高斯误差线性单元（比ReLU更平滑）
            nn.Dropout(dropout),   # 前馈层0.1随机丢弃
            nn.Linear(dim*4, dim)  # 特征降维（1024->256）
        )
        
        # 残差连接后的随机失活
        self.dropout = nn.Dropout(dropout)  # 残差路径0.1丢弃率（正则化）

    def forward(self, query, key, value, mask=None):
        # 跨模态注意力计算流程
        # 步骤1：执行多头注意力计算（query与key-value对交互）
        attention_output, _ = self.cross_attention(
            query=query,         # 查询向量 [B, target_len, 256]
            key=key,             # 键向量 [B, source_len, 256] 
            value=value,         # 值向量 [B, source_len, 256]
            attn_mask=mask       # 注意力掩码（屏蔽无效位置）
        )
        
        # 步骤2：层归一化 + 残差连接（稳定注意力输出）
        attention_output = self.norm_1(attention_output)  # [B, target_len, 256]
        
        # 步骤3：前馈神经网络（增强特征表达能力）
        linear_output = self.ffn(attention_output)  # [B, target_len, 256]
        
        # 步骤4：残差连接（保留原始注意力特征）
        output = attention_output + self.dropout(linear_output)  # 随机丢弃部分前馈输出
        
        # 步骤5：最终层归一化（规范化输出分布）
        output = self.norm_2(output)  # [B, target_len, 256]

        return output


class AgentDecoder(nn.Module):
    def __init__(self, max_time, max_branch, dim):
        super(AgentDecoder, self).__init__()
        # 核心参数配置
        self._max_time = max_time    # 预测时间步（默认8秒）
        self._max_branch = max_branch  # 树状分支数（默认30分支）
        
        # 轨迹解码网络（将高维特征映射为轨迹坐标）
        self.traj_decoder = nn.Sequential(
            nn.Linear(dim, 128),     # 特征降维（512->128 减少计算量）
            nn.ELU(),               # 指数线性单元（平衡梯度特性）
            nn.Linear(128, 3*10)    # 输出轨迹点（3维x10个点：x,y,yaw）
        )

    def forward(self, encoding, current_state):
        """
        智能体轨迹解码流程（树状多模态预测）
        
        输入：
        encoding: [B, M*T, 512] 融合后的场景特征 
            - B: batch_size（批大小）
            - M: 30分支（树状预测的候选分支数）
            - T: 8时间步（预测总时长，如8秒）
            - 512维特征 = 环境特征256 + 自车状态特征256
        current_state: [B, 3] 当前状态（x坐标，y坐标，航向角yaw）

        处理流程：
        1. 特征重塑 → 构建树状预测结构
            - 将扁平化的特征序列 [B, 240, 512] 重构为三维结构 [B,30,8,512]
            - 30分支对应不同驾驶策略（如左转、直行、右变道等）
            - 8时间步对应预测时间维度（每个分支的时序演化）

        2. 轨迹解码 → 生成候选轨迹
            - 通过全连接网络将512维特征映射为30维输出
            - 30维输出分解为：10个轨迹点 × 3维（x,y,yaw）
            - 最终形状 [B,30,80,3]（80=8秒×10点/秒）

        3. 坐标转换 → 相对转绝对坐标系
            - 解码得到的是相对于当前状态的位移量
            - 通过广播机制将current_state加到所有预测轨迹点
            - 最终输出绝对坐标系下的多模态预测结果
        """
        # 
        # 输入参数说明：
        # encoding: [B, M*T, 512] 融合后的场景特征（B: batch_size, M: 30分支, T: 8时间步）
        # 之所以是512，是因为在Encoder中，我们将环境解码和自车条件轨迹解码的特征在最后一
        # 个维度拼接, 进行条件预测。
        # current_state: [B, 3] 智能体当前状态（x,y,yaw）
        
        # 步骤1：特征维度重塑（适配树状预测结构）
        '''
        原始维度：batch_size × (30×8) × 512  转换后：batch_size × 30 × 8 × 512
        '''
        encoding = torch.reshape(encoding, 
            (encoding.shape[0], self._max_branch, self._max_time, 512))  # [B,30,8,512]
        
        # 步骤2：轨迹解码（生成相对坐标轨迹），输入为环境编码和自车轨迹编码
        agent_traj = self.traj_decoder(encoding)  # [B,30,8,30]
        agent_traj = agent_traj.reshape(
            encoding.shape[0], self._max_branch, self._max_time*10, 3)  # [B,30,80,3]
        
        # 步骤3：坐标转换（相对坐标转绝对坐标）
        # 将当前状态作为初始位置，广播到所有分支和时间步
        agent_traj += current_state[:, None, None, :3]  # [B,30,80,3]

        return agent_traj
    

class ScoreDecoder(nn.Module):
    def __init__(self, variable_cost=False):
        """
        轨迹评分解码器初始化（多目标代价计算）
        
        参数说明：
        variable_cost - 是否启用可变代价权重（默认False使用固定权重）
        
        网络结构：
        1. 交互特征编码器：将10维交互特征映射为256维
           输入：10维交互特征（相对位置/速度/属性等）
           结构：Linear(10->64) → ReLU → Linear(64->256)
           
        2. 交互特征解码器：提取4维潜在交互特征
           结构：Linear(256->64) → ELU → Linear(64->4) → Sigmoid
           输出：归一化的潜在特征（范围[0,1]）
           Sigmoid作用：1. 归一化到[0,1]区间，2. 概率化输出
           
        3. 权重解码器：生成特征权重系数
           结构：Linear(256->64) → ELU → Linear(64->8) → Softplus
           输出：8维正权重（4个潜在特征权重 + 4个硬编码特征权重）
           Softplus作用：1. 确保权重非负，2. 平滑梯度特性，3. 动态范围可控
                a.相比ReLU，Softplus更平滑，有助于梯度传播，防止权重值突然归零（避免特征被完全忽略）;
                b.与Sigmoid相比，不限制权重的上限值（适合需要大权重的场景,允许权重在(0, +∞)范围内连续变化;
                c.设计必要性：当计算轨迹评分时，需要通过加权求和融合不同特征（如碰撞风险、行驶舒适性等）。负权重
                    会导致反向贡献，这与实际物理意义相悖。Softplus确保所有特征权重都是正向的，符合"代价越大分数
                    越低"的设计逻辑。
        """
        super(ScoreDecoder, self).__init__()
        # 核心参数配置
        self._n_latent_features = 4    # 潜在交互特征维度
        self._variable_cost = variable_cost  # 是否启用可变代价计算
        
        # 特征编码解码网络
        self.interaction_feature_encoder = nn.Sequential(
            nn.Linear(10, 64),    # 输入10维交互特征（相对位置/速度/属性等）
            nn.ReLU(),            # 保证特征非负性
            nn.Linear(64, 256)    # 编码为256维高级特征
        )
        self.interaction_feature_decoder = nn.Sequential(
            nn.Linear(256, 64),   # 特征降维
            nn.ELU(),             # 平衡梯度特性
            nn.Linear(64, self._n_latent_features),  # 输出4维潜在特征
            nn.Sigmoid()          # 归一化潜在特征值到[0,1]
        )
        self.weights_decoder = nn.Sequential(
            nn.Linear(256, 64),   # 接收场景编码特征
            nn.ELU(),
            nn.Linear(64, self._n_latent_features+4),  # 输出8维权重（4+4）
            nn.Softplus()         # 保证权重值为正数
        )

    def get_hardcoded_features(self, ego_traj, max_time):
        # ego_traj: B, M, T, 6
        # x, y, yaw, v, a, r

        speed = ego_traj[:, :, :max_time, 3]
        acceleration = ego_traj[:, :, :max_time, 4]
        jerk = torch.diff(acceleration, dim=-1) / 0.1
        jerk = torch.cat((jerk[:, :, :1], jerk), dim=-1)
        curvature = ego_traj[:, :, :max_time, 5]
        lateral_acceleration = speed ** 2 * curvature

        speed = -speed.mean(-1).clip(0, 15) / 15
        acceleration = acceleration.abs().mean(-1).clip(0, 4) / 4
        jerk = jerk.abs().mean(-1).clip(0, 6) / 6
        lateral_acceleration = lateral_acceleration.abs().mean(-1).clip(0, 5) / 5

        features = torch.stack((speed, acceleration, jerk, lateral_acceleration), dim=-1)

        return features
    
    def calculate_collision(self, ego_traj, agent_traj, agents_states, max_time):
        # ego_traj: B, T, 3
        # agent_traj: B, N, T, 3
        # agents_states: B, N, 11

        agent_mask = torch.ne(agents_states.sum(-1), 0) # B, N

        # Compute the distance between the two agents
        dist = torch.norm(ego_traj[:, None, :max_time, :2] - agent_traj[:, :, :max_time, :2], dim=-1)
    
        # Compute the collision cost
        cost = torch.exp(-0.2 * dist ** 2) * agent_mask[:, :, None]
        cost = cost.sum(-1).sum(-1)

        return cost
    
    def get_latent_interaction_features(self, ego_traj, agent_traj, agents_states, max_time):
        # ego_traj: B, T, 6
        # agent_traj: B, N, T, 3
        # agents_states: B, N, 11

        # Get agent mask
        agent_mask = torch.ne(agents_states.sum(-1), 0) # B, N

        # Get relative attributes of agents
        relative_yaw = agent_traj[:, :, :max_time, 2] - ego_traj[:, None, :max_time, 2]
        relative_yaw = torch.atan2(torch.sin(relative_yaw), torch.cos(relative_yaw))
        relative_pos = agent_traj[:, :, :max_time, :2] - ego_traj[:, None, :max_time, :2]
        relative_pos = torch.stack([relative_pos[..., 0] * torch.cos(relative_yaw), 
                                    relative_pos[..., 1] * torch.sin(relative_yaw)], dim=-1)
        agent_velocity = torch.diff(agent_traj[:, :, :max_time, :2], dim=-2) / 0.1
        agent_velocity = torch.cat((agent_velocity[:, :, :1, :], agent_velocity), dim=-2)
        ego_velocity_x = ego_traj[:, :max_time, 3] * torch.cos(ego_traj[:, :max_time, 2])
        ego_velocity_y = ego_traj[:, :max_time, 3] * torch.sin(ego_traj[:, :max_time, 2])
        relative_velocity = torch.stack([(agent_velocity[..., 0] - ego_velocity_x[:, None]) * torch.cos(relative_yaw),
                                         (agent_velocity[..., 1] - ego_velocity_y[:, None]) * torch.sin(relative_yaw)], dim=-1) 
        relative_attributes = torch.cat((relative_pos, relative_yaw.unsqueeze(-1), relative_velocity), dim=-1)

        # Get agent attributes
        agent_attributes = agents_states[:, :, None, 6:].expand(-1, -1, relative_attributes.shape[2], -1)
        attributes = torch.cat((relative_attributes, agent_attributes), dim=-1)
        attributes = attributes * agent_mask[:, :, None, None]

        # Encode relative attributes and decode to latent interaction features
        features = self.interaction_feature_encoder(attributes)
        features = features.max(1).values.mean(1)
        features = self.interaction_feature_decoder(features)
  
        return features

    def forward(self, ego_traj, ego_encoding, agents_traj, agents_states, timesteps):
        """ 轨迹评分前向计算流程（多分支代价评估）
        
        处理流程：
        1. 提取硬编码特征（速度/加速度等动力学参数）
        2. 生成动态权重系数（场景自适应特征权重）
        3. 遍历候选分支计算综合代价
        4. 整合评分并过滤无效轨迹
        """
        # 1. 提取自车轨迹的硬编码特征（batch_size × 分支数 × 特征数）
        ego_traj_features = self.get_hardcoded_features(ego_traj, timesteps)
        
        # 固定代价权重模式：使用全1向量代替编码特征（禁用场景自适应）
        if not self._variable_cost:
            ego_encoding = torch.ones_like(ego_encoding)
        
        # 2. 解码特征权重（batch_size × 8维权重）
        weights = self.weights_decoder(ego_encoding)
        
        # 生成有效轨迹掩码（过滤全零填充的无效候选轨迹）
        ego_mask = torch.ne(ego_traj.sum(-1).sum(-1), 0)  # [B, M]

        # 3. 遍历每个候选分支计算综合评分
        scores = []
        for i in range(agents_traj.shape[1]):  # agents_traj.shape[1] = 分支数
            # 获取当前分支的硬编码特征（速度/加速度等）
            hardcoded_features = ego_traj_features[:, i]  # [B, 4]
            
            # 计算潜在交互特征（与其他交通参与者的时空关系）
            interaction_features = self.get_latent_interaction_features(
                ego_traj[:, i],       # 当前分支的自车轨迹
                agents_traj[:, i],    # 当前分支的周围车辆预测
                agents_states,        # 周围车辆历史状态
                timesteps             # 预测时间步
            )  # [B, 4]
            
            # 特征拼接与加权求和（4硬编码特征 + 4交互特征）
            features = torch.cat((hardcoded_features, interaction_features), dim=-1)  # [B, 8]
            score = -torch.sum(features * weights, dim=-1)  # 线性加权计算基础评分
            
            # 计算碰撞代价（指数级放大碰撞风险）
            collision_feature = self.calculate_collision(
                ego_traj[:, i],      # 当前分支自车轨迹
                agents_traj[:, i],   # 当前分支周围车辆轨迹
                agents_states,       # 周围车辆状态
                timesteps            # 时间步
            )  # [B]
            score += -10 * collision_feature  # 碰撞代价系数强化（提高安全性权重）
            
            scores.append(score)

        # 4. 整合与后处理
        scores = torch.stack(scores, dim=1)  # 堆叠分支维度 → [B, M]
        scores = torch.where(ego_mask, scores, float('-inf'))  # 掩码过滤无效轨迹

        return scores, weights
