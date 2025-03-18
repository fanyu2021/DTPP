import torch
from prediction_modules import *


class Encoder(nn.Module):
    # 场景编码器：整合多源信息（自车、邻居、地图元素）的特征编码
    def __init__(self, dim=256, layers=3, heads=8, dropout=0.1):
        # 场景编码器初始化：整合多模态交通要素的特征编码
        super(Encoder, self).__init__()

        # 地图元素参数配置
        self._lane_len = 50        # 单条车道线采样点数（每车道50个离散点）
        self._lane_feature = 7     # 车道线特征维度
        self._crosswalk_len = 30   # 单条斑马线采样点数（每斑马线30个离散点）
        self._crosswalk_feature = 3 # 斑马线特征维度

        # 动态交通参与者编码器
        # 邻居车辆编码：处理11维时序特征（x,y,heading,速度,加速度,转向角,类型等）
        self.agent_encoder = AgentEncoder(agent_dim=11)  # 编码20个邻居车辆
        # 自车编码：处理7维时序特征（x,y,heading,速度,加速度,转向角,车辆状态）
        self.ego_encoder = AgentEncoder(agent_dim=7)     # 编码ego车辆历史轨迹

        # 静态地图编码器
        self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len)       # 编码车道线（50点/条）
        self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len)  # 编码斑马线（30点/条）

        # 多模态特征融合模块（Transformer架构）
        attention_layer = nn.TransformerEncoderLayer(
            d_model=dim,            # 特征维度256（保持各模态特征维度一致）
            nhead=heads,            # 8头注意力机制（256/8=32）
            dim_feedforward=dim*4,  # 前馈网络维度1024（256*4）
            activation=F.gelu,      # 使用GELU激活函数（比ReLU更平滑）
            dropout=dropout,        # 0.1的dropout比例（防止过拟合）
            batch_first=True        # 输入格式为(batch, sequence, feature)
        )

        # 堆叠3层Transformer编码层（深层特征融合）
        """
        该参数控制两种计算模式：
            enable_nested_tensor=True (默认)
            使用嵌套张量(Nested Tensor)优化，自动跳过padding部分的计算
            优势：对包含大量padding的序列（如NLP中不同长度的句子）可提升20-30%的计算速度
            限制：要求必须提供有效的src_key_padding_mask，且当所有序列都是满长度时会自动回退到普通张量模式
            enable_nested_tensor=False (您当前代码的设置)

            使用标准张量计算模式
            优势：计算过程更稳定，适用于以下场景：
            序列长度相对统一（如您代码中的地图元素点序列长度固定）
            需要与旧版PyTorch (<1.9) 兼容
            遇到嵌套张量相关的边缘情况报错时
            在您的情景中设为False的可能原因：

            地图元素的点序列长度固定（车道线50点/斑马线30点）
            交通参与者的历史轨迹长度统一（21个时间步）
            更关注计算稳定性而非极限性能优化
        """
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)

    def forward(self, inputs):
        # 场景特征编码主流程
        # 处理动态交通参与者特征 --------------------------------------------------
        # 获取输入数据
        ego = inputs['ego_agent_past']       # 自车历史轨迹 [B, 21, 7]
        neighbors = inputs['neighbor_agents_past']  # 邻居车辆历史轨迹 [B, 20, 21, 11]
        
        # 合并所有交通参与者（自车+邻居）并提取关键特征
        # 维度说明：在dim=1插入自车维度，并截取前5个运动学特征（x,y,heading,speed,accel）
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)  # [B, 21, 21, 5]

        # 编码动态参与者特征
        encoded_ego = self.ego_encoder(ego)  # 自车特征编码 [B, 256]
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])]  # 20个邻居编码
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1)  # 合并所有参与者特征 [B, 21, 256]
        
        # 生成参与者掩码（无效轨迹标记）
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0)  # 检查最后一个特征维度的求和结果是否为0 [B, 21]

        # 处理静态地图特征 ------------------------------------------------------
        map_lanes = inputs['map_lanes']        # 车道线数据 [B, N_lane, 50, 7]
        map_crosswalks = inputs['map_crosswalks']  # 斑马线数据 [B, N_crosswalk, 30, 3]
        
        # 编码地图特征
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)         # 车道线编码 [B, S_lane, 256]
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)  # 斑马线编码 [B, S_cross, 256]

        # 多模态特征融合 --------------------------------------------------------
        # 合并所有特征序列（参与者+车道线+斑马线）
        input = torch.cat([encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1)  # [B, total_seq, 256]
        
        # 合并对应掩码（标识无效数据位置）
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1)  # [B, total_seq]
        
        # 通过Transformer编码器进行全局特征融合
        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)  # 输出 [B, total_seq, 256]

        # 封装输出结果
        encoder_outputs = {
            'encoding': encoding,  # 融合后的场景特征编码
            'mask': mask           # 全局掩码（用于后续解码时忽略无效位置）
        }
        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, neighbors=10, max_time=8, max_branch=30, n_heads=8, dim=256, variable_cost=False):
        # 场景解码器初始化：负责多模态轨迹预测与评分
        super(Decoder, self).__init__()
        # 关键参数配置
        self._neighbors = neighbors   # 最大邻居车辆数（默认10辆）
        self._nheads = n_heads        # 注意力头数（8头）
        self._time = max_time        # 预测时间步（8个时间步）
        self._branch = max_branch    # 树状预测分支数（30分支）

        # 注意力解码模块
        self.environment_decoder = CrossAttention(n_heads, dim)  # 环境感知解码器
        self.ego_condition_decoder = CrossAttention(n_heads, dim)  # 自车条件解码器

        # 时间序列嵌入
        self.time_embed = nn.Embedding(max_time, dim)  # 时间步嵌入（8时间步 x 256维）

        # 轨迹编解码网络
        self.ego_traj_encoder = nn.Sequential(  # 自车轨迹编码器
            nn.Linear(6, 64),    # 输入6维（x,y,yaw,speed,accel,steering）
            nn.ReLU(),           # 激活函数
            nn.Linear(64, 256)   # 输出256维特征
        )
        self.agent_traj_decoder = AgentDecoder(max_time, max_branch, dim*2)  # 智能体轨迹解码器（输入512维）
        self.ego_traj_decoder = nn.Sequential(  # 自车轨迹解码器
            nn.Linear(256, 256),  # 特征对齐
            nn.ELU(),            # 指数线性单元
            nn.Linear(256, max_time*10*3)  # 输出8秒轨迹（80点x3维：x,y,yaw）
        )

        # 评分与正则化模块
        self.scorer = ScoreDecoder(variable_cost)  # 轨迹评分解码器（含可变代价计算）
        
        # 注册缓冲区（非学习参数）
        self.register_buffer('casual_mask', self.generate_casual_mask())  # 因果掩码（防止未来信息泄漏）
        self.register_buffer('time_index', torch.arange(max_time).repeat(max_branch, 1))  # 时间索引矩阵[30分支 x 8时间步]

    def pooling_trajectory(self, trajectory_tree):
        B, M, T, D = trajectory_tree.shape
        trajectory_tree = torch.reshape(trajectory_tree, (B, M, T//10, 10, D))
        trajectory_tree = torch.max(trajectory_tree, dim=-2)[0]

        return trajectory_tree

    def generate_casual_mask(self):
        time_mask = torch.tril(torch.ones(self._time, self._time))
        casual_mask = torch.zeros(self._branch * self._time, self._branch * self._time)
        for i in range(self._branch):
            casual_mask[i*self._time:(i+1)*self._time, i*self._time:(i+1)*self._time] = time_mask

        return casual_mask

    def forward(self, encoder_outputs, ego_traj_inputs, agents_states, timesteps):
        # get inputs
        current_states = agents_states[:, :self._neighbors, -1]
        encoding, encoding_mask = encoder_outputs['encoding'], encoder_outputs['mask']
        ego_traj_ori_encoding = self.ego_traj_encoder(ego_traj_inputs)
        branch_embedding = ego_traj_ori_encoding[:, :, timesteps-1]
        ego_traj_ori_encoding = self.pooling_trajectory(ego_traj_ori_encoding)
        time_embedding = self.time_embed(self.time_index)
        tree_embedding = time_embedding[None, :, :, :] + branch_embedding[:, :, None, :]

        # get mask
        ego_traj_mask = torch.ne(ego_traj_inputs.sum(-1), 0)
        ego_traj_mask = ego_traj_mask[:, :, ::(ego_traj_mask.shape[-1]//self._time)]
        ego_traj_mask = torch.reshape(ego_traj_mask, (ego_traj_mask.shape[0], -1))
        env_mask = torch.einsum('ij,ik->ijk', ego_traj_mask, encoding_mask.logical_not())
        env_mask = torch.where(env_mask == 1, 0, -1e9)
        env_mask = env_mask.repeat(self._nheads, 1, 1)
        ego_condition_mask = self.casual_mask[None, :, :] * ego_traj_mask[:, :, None]
        ego_condition_mask = torch.where(ego_condition_mask == 1, 0, -1e9)
        ego_condition_mask = ego_condition_mask.repeat(self._nheads, 1, 1)

        # decode
        agents_trajecotries = []
        for i in range(self._neighbors):
            # learnable query
            query = encoding[:, i+1, None, None] + tree_embedding
            query = torch.reshape(query, (query.shape[0], -1, query.shape[-1]))
      
            # decode from environment inputs
            env_decoding = self.environment_decoder(query, encoding, encoding, env_mask)

            # decode from ego trajectory inputs
            ego_traj_encoding = torch.reshape(ego_traj_ori_encoding, (ego_traj_ori_encoding.shape[0], -1, ego_traj_ori_encoding.shape[-1]))
            ego_condition_decoding = self.ego_condition_decoder(query, ego_traj_encoding, ego_traj_encoding, ego_condition_mask)

            # trajectory outputs
            decoding = torch.cat([env_decoding, ego_condition_decoding], dim=-1)
            trajectory = self.agent_traj_decoder(decoding, current_states[:, i])
            agents_trajecotries.append(trajectory)

        # score outputs
        agents_trajecotries = torch.stack(agents_trajecotries, dim=2)
        scores, weights = self.scorer(ego_traj_inputs, encoding[:, 0], agents_trajecotries, current_states, timesteps)

        # ego regularization
        ego_traj_regularization = self.ego_traj_decoder(encoding[:, 0])
        ego_traj_regularization = torch.reshape(ego_traj_regularization, (ego_traj_regularization.shape[0], 80, 3))

        return agents_trajecotries, scores, ego_traj_regularization, weights