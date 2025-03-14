import torch
from prediction_modules import *


class Encoder(nn.Module):
    def __init__(self, dim=256, layers=3, heads=8, dropout=0.1):
        # 调用父类构造函数
        super(Encoder, self).__init__()

        # 定义车道长度
        self._lane_len = 50
        # 定义车道特征维度
        self._lane_feature = 7
        # 定义斑马线长度
        self._crosswalk_len = 30
        # 定义斑马线特征维度
        self._crosswalk_feature = 3

        # 初始化agent编码器
        # TODO(fanyu): 这里的agent_dim 为什么是11？
        self.agent_encoder = AgentEncoder(agent_dim=11)
        # 初始化自车编码器
        # TODO(fanyu): 这里的agent_dim 为什么是7？
        self.ego_encoder = AgentEncoder(agent_dim=7)
        # 初始化车道编码器
        self.lane_encoder = VectorMapEncoder(self._lane_feature, self._lane_len)
        # 初始化斑马线编码器
        self.crosswalk_encoder = VectorMapEncoder(self._crosswalk_feature, self._crosswalk_len)

        # 初始化注意力层
        attention_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4,
                                                     activation=F.gelu, dropout=dropout, batch_first=True)
        # 初始化融合编码器
        self.fusion_encoder = nn.TransformerEncoder(attention_layer, layers, enable_nested_tensor=False)

    def forward(self, inputs):
        # agents
        ego = inputs['ego_agent_past'] # ego shape: torch.Size([1, 21, 7])
        neighbors = inputs['neighbor_agents_past'] # neighbors shape: torch.Size([1, 20, 21, 11])
        # actors shape: torch.Size([1, 21, 21, 5)
        print(f'---41---ego shape: {ego.shape}, neighbors shape: {neighbors.shape}')
        actors = torch.cat([ego[:, None, :, :5], neighbors[..., :5]], dim=1)

        # agent encoding
        encoded_ego = self.ego_encoder(ego) # encoded_ego shape: torch.Size([1, 256])
        # encoded_neighbors List(len=20) 20 X shape: torch.Size([1, 256])
        encoded_neighbors = [self.agent_encoder(neighbors[:, i]) for i in range(neighbors.shape[1])] 
        # encoded_actors shape: torch.Size([1, 21, 256])
        encoded_actors = torch.stack([encoded_ego] + encoded_neighbors, dim=1) 
        actors_mask = torch.eq(actors[:, :, -1].sum(-1), 0) # actors[:, :, -1] 等价于 actors[:, :, -1, :], 去掉无效值？

        # vector maps
        map_lanes = inputs['map_lanes']
        # TODO(fanyu): 这里的map_crosswalks 先屏蔽掉
        # map_crosswalks = inputs['map_crosswalks']
        map_crosswalks = torch.zeros(size=(1, 5, 30, 3)).to(device=ego.device)

        # map encoding
        encoded_map_lanes, lanes_mask = self.lane_encoder(map_lanes)
        encoded_map_crosswalks, crosswalks_mask = self.crosswalk_encoder(map_crosswalks)

        # attention fusion encoding
        input = torch.cat([encoded_actors, encoded_map_lanes, encoded_map_crosswalks], dim=1)
        """
        TODO(fanyu): 这里的mask 为什么要拼接？
        问题：输入的注意力掩码（mask）未使用布尔类型（torch.bool），而是其他类型（如 float32 或 int64），导致性能下降。
        """
        mask = torch.cat([actors_mask, lanes_mask, crosswalks_mask], dim=1) # .to(type=torch.bool) # 显示定义为bool类型
        # fanyu: 增加断言，确保mask不为None
        assert mask is not None, "Mask cannot be None"
        encoding = self.fusion_encoder(input, src_key_padding_mask=mask)

        # outputs
        encoder_outputs = {'encoding': encoding, 'mask': mask}

        return encoder_outputs


class Decoder(nn.Module):
    def __init__(self, neighbors=10, max_time=8, max_branch=30, n_heads=8, dim=256, variable_cost=False):
        super(Decoder, self).__init__()
        self._neighbors = neighbors
        self._nheads = n_heads
        self._time = max_time
        self._branch = max_branch

        self.environment_decoder = CrossAttention(n_heads, dim)
        self.ego_condition_decoder = CrossAttention(n_heads, dim)
        self.time_embed = nn.Embedding(max_time, dim)
        self.ego_traj_encoder = nn.Sequential(nn.Linear(6, 64), nn.ReLU(), nn.Linear(64, 256))
        self.agent_traj_decoder = AgentDecoder(max_time, max_branch, dim*2)
        self.ego_traj_decoder = nn.Sequential(nn.Linear(256, 256), nn.ELU(), nn.Linear(256, max_time*10*3))
        self.scorer = ScoreDecoder(variable_cost)
        self.register_buffer('casual_mask', self.generate_casual_mask())
        self.register_buffer('time_index', torch.arange(max_time).repeat(max_branch, 1))

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