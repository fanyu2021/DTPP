import torch
import scipy
import random
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely import Point, LineString
from shapely.geometry.base import CAP_STYLE
from path_planner import calc_spline_course
from bezier_path import calc_4points_bezier_path
from collections import defaultdict
from spline_planner import SplinePlanner
from torch.nn.utils.rnn import pad_sequence
from scenario_tree_prediction import *
from planner_utils import *
from nuplan.planning.simulation.observation.idm.utils import path_to_linestring


class TrajTree:
    def __init__(self, traj, parent, depth):
        self.traj = traj
        self.state = traj[-1, :5]
        self.children = list()
        self.parent = parent
        self.depth = depth
        self.attribute = dict()
        if parent is not None:
            self.total_traj = torch.cat((parent.total_traj, traj), 0)
        else:
            self.total_traj = traj

    def expand(self, child):
        self.children.append(child)

    def expand_set(self, children):
        self.children += children

    def expand_children(self, paths, horizon, speed_limit, planner):
        """ 扩展当前节点的子节点（轨迹分支）
        
        参数:
        paths -- 候选路径列表（来自路径规划模块）
        horizon -- 预测时间范围（秒）
        speed_limit -- 当前道路限速（m/s）
        planner -- 轨迹生成器实例
        
        流程:
        1. 调用轨迹生成器生成多条候选轨迹
        2. 将每条轨迹封装为轨迹树节点
        3. 建立父子节点关系
        """
        # 生成多条候选轨迹（基于当前状态和路径约束）
        trajs = planner.gen_trajectories(self.state, horizon, paths, speed_limit, self.isroot())
        
        # 创建子节点（每个第一阶段轨迹作为一个独立分支）
        children = [TrajTree(traj, self, self.depth + 1) for traj in trajs]
        
        # 将子节点加入当前节点
        self.expand_set(children)

    def isroot(self):
        return self.parent is None

    def isleaf(self):
        return len(self.children) == 0

    def get_subseq_trajs(self):
        return [child.traj for child in self.children]
    
    def get_all_leaves(self, leaf_set=[]):
        if self.isleaf():
            print(self.state)
            leaf_set.append(self)
        else:
            for child in self.children:
                leaf_set = child.get_all_leaves(leaf_set)

        return leaf_set

    @staticmethod
    def get_children(obj):
        if isinstance(obj, TrajTree):
            return obj.children
        
        elif isinstance(obj, list):
            children = [node.children for node in obj]
            children = list(itertools.chain.from_iterable(children))
            return children
        
        else:
            raise TypeError("obj must be a TrajTree or a list")

    def plot_tree(self, ax=None, msize=12):
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 10))
        state = self.state.cpu().detach().numpy()
        
        ax.plot(state[0], state[1], marker="o", color="b", markersize=msize)

        if self.traj.shape[0] > 1:
            if self.parent is not None:
                traj_l = torch.cat((self.parent.traj[-1:],self.traj),0)
                traj = traj_l.cpu().detach().numpy()
            else:
                traj = self.traj.cpu().detach().numpy()

            ax.plot(traj[:, 0], traj[:, 1], color="k")

        for child in self.children:
            child.plot_tree(ax)

        return ax

    @staticmethod
    def get_children_index_torch(nodes):
        indices = dict()
        for depth, nodes_d in nodes.items():
            if depth+1 in nodes:
                childs_d = nodes[depth+1]
                indices_d = list()
                for node in nodes_d:
                    indices_d.append(torch.tensor([childs_d.index(child) for child in node.children]))
                indices[depth] = pad_sequence(indices_d, batch_first=True, padding_value=-1)

        return indices
    
    @staticmethod
    def get_nodes_by_level(obj, depth, nodes=None, trim_short_branch=True):
        assert obj.depth <= depth
        if nodes is None:
            nodes = defaultdict(lambda: list())

        if obj.depth == depth:
            nodes[depth].append(obj)

            return nodes, True
        else:
            if obj.isleaf():
                return nodes, False
            else:
                flag = False
                children_flags = dict()
                for child in obj.children:
                    nodes, child_flag = TrajTree.get_nodes_by_level(child, depth, nodes)
                    children_flags[child] = child_flag
                    flag = flag or child_flag

                if trim_short_branch:
                    obj.children = [child for child in obj.children if children_flags[child]]
                if flag:
                    nodes[obj.depth].append(obj)

                return nodes, flag


class TreePlanner:
    def __init__(self, device, encoder, decoder, n_candidates_expand=5, n_candidates_max=30):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_path_len = 120 # [m]
        self.target_depth = MAX_LEN # [m]
        self.target_speed = 13 # [m/s]
        self.horizon = 8 # [s]
        self.first_stage_horizon = 3 # [s]
        self.n_candidates_expand = n_candidates_expand # second stage
        self.n_candidates_max = n_candidates_max # max number of candidates
        self.planner = SplinePlanner(self.first_stage_horizon, self.horizon)  

    def get_candidate_paths(self, edges):
        """ 生成候选路径集合
        
        参数:
        edges: 候选车道边缘列表（来自get_candidate_edges）
        
        返回:
        经过筛选的候选路径列表，每个元素为元组 (路径长度, 距自车最短距离, 路径点阵)
        """
        # 步骤1：通过深度优先搜索生成原始路径
        paths = []
        for edge in edges:
            paths.extend(self.depth_first_search(edge))  # 递归探索所有可能路径分支

        # 步骤2：路径后处理与特征提取
        candidate_paths = []
        for i, path in enumerate(paths):
            # 合并车道的离散路径点
            path_polyline = []
            for edge in path:
                path_polyline.extend(edge.baseline_path.discrete_path)
            
            # 路径有效性检查与坐标转换
            path_polyline = check_path(np.array(path_to_linestring(path_polyline).coords))  # 确保路径连续
            dist_to_ego = scipy.spatial.distance.cdist([self.ego_point], path_polyline)     # 计算自车到路径的最短距离
            
            # 路径修剪：从最近点开始截取后续路径
            path_polyline = path_polyline[dist_to_ego.argmin():]
            if len(path_polyline) < 3:  # 过滤过短路径
                continue

            # 计算路径特征：长度（假设每点间隔0.25米）、航向角
            path_len = len(path_polyline) * 0.25
            polyline_heading = calculate_path_heading(path_polyline)  # 计算每个路径点的航向
            path_polyline = np.stack([path_polyline[:, 0], path_polyline[:, 1], polyline_heading], axis=1)
            candidate_paths.append((path_len, dist_to_ego.min(), path_polyline))

        # 步骤3：路径长度筛选（保留长度超过阈值路径）
        max_path_len = max([v[0] for v in candidate_paths])
        acceptable_path_len = MAX_LEN/2 if max_path_len > MAX_LEN/2 else max_path_len
        paths = [v for v in candidate_paths if v[0] >= acceptable_path_len]

        return paths

    def get_candidate_edges(self, starting_block):
        """ 获取自车所在道路块内的候选车道边缘
        
        参数:
        starting_block: 自车当前所在道路块对象
        
        返回:
        候选车道边缘列表，包含距离自车较近的车道边缘对象
        """
        edges = []           # 候选边缘容器
        edges_distance = []  # 边缘距离记录
        # 获取自车后轴中心坐标 (x,y)
        self.ego_point = (self.ego_state.rear_axle.x, self.ego_state.rear_axle.y)

        # 遍历道路块内部所有车道边缘
        for edge in starting_block.interior_edges:
            # 计算边缘多边形到自车的欧式距离
            dist = edge.polygon.distance(Point(self.ego_point))
            edges_distance.append(dist)
            
            # 筛选距离自车4米范围内的车道边缘
            if dist < 4:
                edges.append(edge)
        
        # 回退逻辑：若无符合条件边缘，选择最近边缘
        if len(edges) == 0:
            closest_idx = np.argmin(edges_distance)
            edges.append(starting_block.interior_edges[closest_idx])

        return edges

    def generate_paths(self, routes):
        # 获取自车状态（后轴中心坐标和航向角）
        ego_state = self.ego_state.rear_axle.x, self.ego_state.rear_axle.y, self.ego_state.rear_axle.heading
        
        # 初始化路径容器
        new_paths = []
        path_distance = []  # 存储路径到自车的初始距离

        # 遍历所有候选路径（第一阶段）
        for (path_len, dist, path_polyline) in routes:
            # 根据路径长度动态调整采样点密度（平衡计算效率与路径连续性）
            if len(path_polyline) > 81:
                sampled_index = np.array([5, 10, 15, 20]) * 4  # 长路径：多阶段采样
            elif len(path_polyline) > 61:
                sampled_index = np.array([5, 10, 15]) * 4     # 中长路径：三阶段采样
            elif len(path_polyline) > 41:
                sampled_index = np.array([5, 10]) * 4          # 中等路径：双阶段采样
            elif len(path_polyline) > 21:
                sampled_index = [20]                            # 短路径：末端采样
            else:
                sampled_index = [1]                             # 极短路径：起点采样
     
            # 基于采样点生成目标状态
            target_states = path_polyline[sampled_index].tolist()
            
            # 两阶段路径生成（贝塞尔曲线 + 原路径延伸）
            for j, state in enumerate(target_states):
                # 第一阶段：生成连接当前状态和目标点的贝塞尔曲线（3秒轨迹）
                first_stage_path = calc_4points_bezier_path(ego_state[0], ego_state[1], ego_state[2], 
                                                            state[0], state[1], state[2], 3, sampled_index[j])[0]
                # 第二阶段：接续原始路径后续点
                second_stage_path = path_polyline[sampled_index[j]+1:, :2]
                # 合并两阶段路径
                path_polyline = np.concatenate([first_stage_path, second_stage_path], axis=0)
                new_paths.append(path_polyline)  
                path_distance.append(dist)   # 保留原始距离参数用于后续评估

        # 路径评估与筛选
        candiate_paths = {}
        # 计算每条路径的综合代价值
        for path, dist in zip(new_paths, path_distance):
            cost = self.calculate_cost(path, dist)
            candiate_paths[cost] = path

        # 按代价值排序并保留最优三条路径
        candidate_paths = []
        for cost in sorted(candiate_paths.keys())[:3]:
            path = candiate_paths[cost]
            path = self.post_process(path)  # 进行路径后处理（坐标系转换+样条插值）
            candidate_paths.append(path)

        return candidate_paths

    def calculate_cost(self, path, dist):
        # path curvature
        curvature = self.calculate_path_curvature(path[0:100])
        curvature = np.max(curvature)

        # lane change
        lane_change = dist

        # check obstacles
        obstacles = self.check_obstacles(path[0:100:10], self.obstacles)
        
        # final cost
        cost = 10 * obstacles + 1 * lane_change  + 0.1 * curvature

        return cost

    def post_process(self, path):
        path = self.transform_to_ego_frame(path)
        index = np.arange(0, len(path), 10)
        x = path[:, 0][index]
        y = path[:, 1][index]

        # spline interpolation
        rx, ry, ryaw, rk = calc_spline_course(x, y)
        spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
        ref_path = spline_path[:self.max_path_len*10]

        return ref_path

    def depth_first_search(self, starting_edge, depth=0):
        if depth >= self.target_depth:
            return [[starting_edge]]
        else:
            traversed_edges = []
            child_edges = [edge for edge in starting_edge.outgoing_edges if edge.id in self.candidate_lane_edge_ids]

            if child_edges:
                for child in child_edges:
                    edge_len = len(child.baseline_path.discrete_path) * 0.25
                    traversed_edges.extend(self.depth_first_search(child, depth+edge_len))

            if len(traversed_edges) == 0:
                return [[starting_edge]]

            edges_to_return = []

            for edge_seq in traversed_edges:
                edges_to_return.append([starting_edge] + edge_seq)
                    
            return edges_to_return

    @staticmethod
    def calculate_path_curvature(path):
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(3/2)

        return curvature

    @staticmethod
    def check_obstacles(path, obstacles):
        expanded_path = LineString(path).buffer((WIDTH/2), cap_style=CAP_STYLE.square)

        for obstacle in obstacles:
            obstacle_polygon = obstacle.geometry
            if expanded_path.intersects(obstacle_polygon):
                return 1

        return 0

    def predict(self, encoder_outputs, traj_inputs, agent_states, timesteps):
        """ 轨迹预测方法：使用解码器生成周边交通参与者轨迹预测
        
        参数:
        encoder_outputs -- 编码器输出的环境特征
        traj_inputs     -- 候选轨迹集合（多个候选轨迹的时序状态）
        agent_states    -- 周边交通参与者历史状态
        timesteps       -- 预测时间步长

        返回:
        agent_trajs -- 预测的交通参与者轨迹 [batch, n_agents, timesteps, state_dim]
        scores      -- 候选轨迹的匹配分数（用于轨迹选择）
        """
        # 初始化候选轨迹张量容器（最大候选数 × 时间步数 × 状态维度）
        ego_trajs = torch.zeros((self.n_candidates_max, self.horizon*10, 6)).to(self.device)
        
        # 填充候选轨迹数据（处理变长输入）
        for i, traj in enumerate(traj_inputs):
            # 截取前6个状态参数（x,y,heading,speed,accel,curvature）
            ego_trajs[i, :len(traj)] = traj[..., :6].float()

        # 增加批次维度（适配解码器输入格式）
        ego_trajs = ego_trajs.unsqueeze(0)  # shape: [1, n_candidates, timesteps, 6]
        
        # 调用解码器进行轨迹预测
        agent_trajs, scores, _, _ = self.decoder(
            encoder_outputs,    # 环境编码特征
            ego_trajs,          # 候选自车轨迹
            agent_states,       # 其他交通参与者历史状态
            timesteps           # 预测时间步长
        )

        return agent_trajs, scores

    def transform_to_ego_frame(self, path):
        # 将全局坐标系下的路径点转换到自车坐标系
        # 平移变换：以自车后轴中心为原点
        x = path[:, 0] - self.ego_state.rear_axle.x
        y = path[:, 1] - self.ego_state.rear_axle.y
        
        # 旋转变换：消除自车航向角影响
        # 使用二维旋转矩阵：[cosθ  -sinθ]
        #                [sinθ   cosθ]
        x_e = x * np.cos(-self.ego_state.rear_axle.heading) - y * np.sin(-self.ego_state.rear_axle.heading)
        y_e = x * np.sin(-self.ego_state.rear_axle.heading) + y * np.cos(-self.ego_state.rear_axle.heading)
        # # 将上述变换合并为一个矩阵运算
        # rotation_matrix = np.array([[np.cos(-self.ego_state.rear_axle.heading), -np.sin(-self.ego_state.rear_axle.heading)],
        #                             [np.sin(-self.ego_state.rear_axle.heading), np.cos(-self.ego_state.rear_axle.heading)]])
        # path_transformed = np.dot(rotation_matrix, np.vstack([x, y]))


        # x_e = path_transformed[0, :]
        # y_e = path_transformed[1, :]
        
    
        # 合并坐标并返回新路径
        path = np.column_stack([x_e, y_e])

        return path
    
    def transform_to_global_frame(self, path):
        # 将自车坐标系下的路径点转换到全局坐标系
        # 旋转变换：消除自车航向角影响
        # 使用二维旋转矩阵：[cosθ  -sinθ]
        #                [sinθ   cosθ]
        # 采用矩阵运算实现
        rotation_matrix = np.array([[np.cos(self.ego_state.rear_axle.heading), -np.sin(self.ego_state.rear_axle.heading)], 
                                    [np.sin(self.ego_state.rear_axle.heading), np.cos(self.ego_state.rear_axle.heading)]])
        path_transformed = np.dot(rotation_matrix, path.T).T

        # 平移变换：将路径点转换到全局坐标系
        x = path_transformed[:, 0] + self.ego_state.rear_axle.x
        y = path_transformed[:, 1] + self.ego_state.rear_axle.y
        # 合并坐标并返回新路径
        path = np.column_stack([x, y])

        return path

    def plan(self, iteration, ego_state, env_inputs, starting_block, route_roadblocks, candidate_lane_edge_ids, traffic_light, observation, debug=False):
        """ 两阶段轨迹规划核心方法
        
        参数:
        iteration: 规划迭代次数（用于调试）
        ego_state: 自车状态（位置/速度/加速度等）
        env_inputs: 环境输入（包含地图/障碍物/其他交通参与者等信息）
        starting_block: 起始道路块对象
        route_roadblocks: 导航路径道路块序列
        candidate_lane_edge_ids: 候选车道边缘ID列表
        traffic_light: 当前交通灯状态
        observation: 环境感知数据
        debug: 调试模式开关

        返回:
        最佳候选轨迹张量 [时间步数, 3] (x,y,航向角)
        """
        # ------------------ 环境信息初始化 ------------------
        self.ego_state = ego_state
        self.candidate_lane_edge_ids = candidate_lane_edge_ids
        self.route_roadblocks = route_roadblocks
        self.traffic_light = traffic_light

        # 障碍物处理（筛选静止车辆和其他障碍物）
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                        TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                        TrackedObjectType.GENERIC_OBJECT]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
        self.obstacles = []
        for obj in objects:
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                # 仅保留速度低于0.1m/s的静止车辆
                if obj.velocity.magnitude() < 0.1:
                    self.obstacles.append(obj.box)
            else:
                self.obstacles.append(obj.box)

        # ------------------ 轨迹树初始化 ------------------
        # 构建根节点状态 [x, y, 航向角, 速度, 加速度, 曲率, 时间]
        state = torch.tensor([[0, 0, 0, 
                               ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                               ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, 
                               0, 0]], dtype=torch.float32)
        tree = TrajTree(state, None, 0)  # 创建轨迹树根节点

        # ------------------ 环境特征编码 ------------------
        encoder_outputs = self.encoder(env_inputs)  # 编码地图/障碍物等信息
        agent_states = env_inputs['neighbor_agents_past']  # 其他交通参与者历史轨迹

        # ------------------ 路径生成 ------------------
        edges = self.get_candidate_edges(starting_block)  # 获取候选车道边缘
        candidate_paths = self.get_candidate_paths(edges) # 生成候选路径集合
        paths = self.generate_paths(candidate_paths)      # 路径后处理与筛选
        self.speed_limit = edges[0].speed_limit_mps or self.target_speed  # 获取道路限速

        # ------------------ 第一阶段轨迹扩展（3秒）------------------
        tree.expand_children(paths, self.first_stage_horizon, self.speed_limit, self.planner)
        leaves = TrajTree.get_children(tree)  # 获取第一阶段叶子节点

        # ------------------ 模型预测与轨迹筛选 ------------------
        # 第一阶段轨迹评分
        trajs = [leaf.total_traj[1:] for leaf in leaves]
        agent_trajectories, scores = self.predict(encoder_outputs, trajs, agent_states, self.first_stage_horizon*10)
        indices = torch.topk(scores, self.n_candidates_expand)[1][0]  # 选择Top-K高分轨迹, 默认5条
        
        # 保留有效高分叶子节点
        pruned_leaves = []
        for i in indices:
            if i.item() < len(leaves): # 因为leaves数量有可能比较少，所以需要判断索引是否超出范围
                pruned_leaves.append(leaves[i])
                parent_scores[leaves[i]] = scores[0, i].item() # 记录第一层节点分数

        # ------------------ 第二阶段轨迹扩展（5秒）------------------
        for leaf in pruned_leaves:
            leaf.expand_children(paths, self.horizon-self.first_stage_horizon, self.speed_limit, self.planner)
        
        # ------------------ 最终轨迹选择 ------------------
        leaves = TrajTree.get_children(leaves)   # 获取第二层所有叶子节点
        if len(leaves) > self.n_candidates_max:  # 随机采样控制计算量
           leaves = random.sample(leaves, self.n_candidates_max) # 随机选择30个叶子节点

        # 最终轨迹评分
        trajs = [leaf.total_traj[1:] for leaf in leaves]
        agent_trajectories, scores = self.predict(encoder_outputs, trajs, agent_states, self.horizon*10)
        
        # 综合两阶段评分选择最佳轨迹
        children_scores = {}
        for i, leaf in enumerate(leaves):
            if leaf.parent in children_scores:
                children_scores[leaf.parent].append(scores[0, i].item())
            else:
                children_scores[leaf.parent] = [scores[0, i].item()]

        # 寻找综合评分最高的轨迹分支
        best_parent = None
        best_child_index = None
        best_score = -np.inf
        for parent in parent_scores.keys():
            score = parent_scores[parent] + np.max(children_scores[parent])  # 两阶段评分加权
            if score > best_score:
                best_parent = parent
                best_score = score
                best_child_index = np.argmax(children_scores[parent])

        best_traj = best_parent.children[best_child_index].total_traj[1:, :3]  # 提取最佳轨迹坐标

        # 调试模式可视化
        if debug:
            for i, traj in enumerate(trajs):
                self.plot(iteration, env_inputs, traj, agent_trajectories[0, i])

        return best_traj
