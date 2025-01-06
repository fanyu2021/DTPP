import torch
import logging
import random
import numpy as np
from torch.utils.data import Dataset
from torch.nn import functional as F


def initLogging(log_file: str, level: str = "INFO"):
    # 设置日志的基本配置
    logging.basicConfig(
        # 日志文件路径
        filename=log_file,
        # 文件写入模式，'w'表示覆盖写入
        filemode='w',
        # 日志级别，默认为INFO
        level=getattr(logging, level, None),
        # 日志格式，包括级别、时间和消息内容
        format='[%(levelname)s %(asctime)s] %(message)s',
        # 时间格式
        datefmt='%m-%d %H:%M:%S'
    )
    # 添加日志输出到控制台的处理器
    logging.getLogger().addHandler(logging.StreamHandler())


def set_seed(CUR_SEED):
    # 设置Python内置的随机数生成器的种子
    random.seed(CUR_SEED)
    # 设置NumPy的随机数生成器的种子
    np.random.seed(CUR_SEED)
    # 设置PyTorch的随机数生成器的种子
    torch.manual_seed(CUR_SEED)
    # 设置PyTorch的cuDNN后端为确定性模式，确保每次运行的结果一致
    torch.backends.cudnn.deterministic = True
    # 禁用PyTorch的cuDNN后端的基准测试模式，以确保结果的一致性
    torch.backends.cudnn.benchmark = False


class DrivingData(Dataset):
    def __init__(self, data_list, n_neighbors, n_candidates):
        self.data_list = data_list
        self._n_neighbors = n_neighbors
        self._n_candidates = n_candidates
        self._time_length = 80

    def __len__(self):
        return len(self.data_list)
    
    def process_ego_trajectory(self, ego_trajectory):
        trajectory = np.zeros((self._n_candidates, self._time_length, 6), dtype=np.float32)
        if ego_trajectory.shape[0] > self._n_candidates:
            ego_trajectory = ego_trajectory[:self._n_candidates]
        
        if ego_trajectory.shape[1] < self._time_length:
            trajectory[:ego_trajectory.shape[0], :ego_trajectory.shape[1]] = ego_trajectory
        else:
            trajectory[:ego_trajectory.shape[0]] = ego_trajectory

        return trajectory

    def __getitem__(self, idx):
        data = np.load(self.data_list[idx])
        ego = data['ego_agent_past']
        neighbors = data['neighbor_agents_past']
        route_lanes = data['route_lanes'] 
        map_lanes = data['map_lanes']
        map_crosswalks = data['map_crosswalks']
        ego_future_gt = data['ego_agent_future']
        neighbors_future_gt = data['neighbor_agents_future'][:self._n_neighbors]
        first_stage = self.process_ego_trajectory(data['first_stage_ego_trajectory'][..., :6])
        second_stage = self.process_ego_trajectory(data['second_stage_ego_trajectory'][..., :6])

        return ego, neighbors, map_lanes, map_crosswalks, route_lanes, ego_future_gt, neighbors_future_gt, first_stage, second_stage


def calc_loss(neighbors, ego, ego_regularization, scores, weights, ego_gt, neighbors_gt, neighbors_valid):
    mask = torch.ne(ego.sum(-1), 0)
    neighbors = neighbors[:, 0] * neighbors_valid 
    cmp_loss = F.smooth_l1_loss(neighbors, neighbors_gt, reduction='none')
    cmp_loss = cmp_loss * mask[:, 0, None, :, None]
    cmp_loss = cmp_loss.sum() / mask[:, 0].sum()

    regularization_loss = F.smooth_l1_loss(ego_regularization, ego_gt, reduction='none')
    regularization_loss = regularization_loss * mask[:, 0, :, None]
    regularization_loss = regularization_loss.sum() / mask[:, 0].sum()

    label = torch.zeros(scores.shape[0], dtype=torch.long).to(scores.device)    
    irl_loss = F.cross_entropy(scores, label)

    weights_regularization = torch.square(weights).mean()

    loss = cmp_loss + irl_loss + 0.1 * regularization_loss + 0.01 * weights_regularization

    return loss


def calc_metrics(plan_trajectory, prediction_trajectories, scores, ego_future, neighbors_future, neighbors_future_valid):
    best_idx = torch.argmax(scores, dim=-1)
    plan_trajectory = plan_trajectory[torch.arange(plan_trajectory.shape[0]), best_idx]
    prediction_trajectories = prediction_trajectories[torch.arange(prediction_trajectories.shape[0]), best_idx]
    prediction_trajectories = prediction_trajectories * neighbors_future_valid
    plan_distance = torch.norm(plan_trajectory[:, :, :2] - ego_future[:, :, :2], dim=-1)
    prediction_distance = torch.norm(prediction_trajectories[:, :, :, :2] - neighbors_future[:, :, :, :2], dim=-1)

    # planning
    plannerADE = torch.mean(plan_distance)
    plannerFDE = torch.mean(plan_distance[:, -1])

    # prediction
    predictorADE = torch.mean(prediction_distance, dim=-1)
    predictorADE = torch.masked_select(predictorADE, neighbors_future_valid[:, :, 0, 0])
    predictorADE = torch.mean(predictorADE)
    predictorFDE = prediction_distance[:, :, -1]
    predictorFDE = torch.masked_select(predictorFDE, neighbors_future_valid[:, :, 0, 0])
    predictorFDE = torch.mean(predictorFDE)

    return plannerADE.item(), plannerFDE.item(), predictorADE.item(), predictorFDE.item()
