import os
import csv
import glob
import torch
import argparse
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from scenario_tree_prediction import Encoder, Decoder
from torch.utils.data import DataLoader
from train_utils import *

import dtpp_data_path as ddp


def train_epoch(data_loader, encoder, decoder, optimizer):
    epoch_loss = []  # 存储每个batch的损失值
    epoch_metrics = []  # 存储每个batch的评估指标
    encoder.train()  # 设置编码器为训练模式
    decoder.train()  # 设置解码器为训练模式

    # 初始化 tqdm 进度条，描述为"Training"，单位为"batch"，进度条将显示每个批次的损失值和进度
    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        # 遍历 data_loader中的每个批次（batch）
        for batch in data_epoch:
            # 准备输入数据（移动到指定设备）
            inputs = {
                'ego_agent_past': batch[0].to(args.device),  # 自车历史轨迹 [batch_size, time_steps, features]
                'neighbor_agents_past': batch[1].to(args.device),  # 邻居车辆历史轨迹 [batch_size, num_neighbors, time_steps, features]
                'map_lanes': batch[2].to(args.device),  # 车道线特征
                'map_crosswalks': batch[3].to(args.device),  # 人行横道特征
                'route_lanes': batch[4].to(args.device)  # 规划路线特征
            }

            # 获取真实未来轨迹
            exist_ok=True
            ego_gt_future = batch[5].to(args.device)  # 自车未来轨迹 [batch_size, 80, 3]
            neighbors_gt_future = batch[6].to(args.device)  # 邻居车辆未来轨迹 [batch_size, 10, 80, 3]
            
            # 生成有效掩码（判断x,y,heading是否非零）
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)

            # 编码阶段（提取环境特征）
            optimizer.zero_grad()
            encoder_outputs = encoder(inputs)  # 环境特征编码输出

            # 第一阶段预测（前3秒轨迹生成）
            first_stage_trajectory = batch[7].to(args.device)  # 候选轨迹 [batch_size, 30, 5]
            # 解码器输出：邻居轨迹预测、分数、自车预测、轨迹权重
            neighbors_trajectories, scores, ego, weights = \
                decoder(encoder_outputs, first_stage_trajectory, inputs['neighbor_agents_past'], 30)
            # 计算多任务损失（规划损失 + 预测损失）
            loss = calc_loss(neighbors_trajectories, first_stage_trajectory, ego, scores, weights,
                             ego_gt_future, neighbors_gt_future, neighbors_future_valid)

            # 第二阶段预测（后5秒轨迹生成）
            second_stage_trajectory = batch[8].to(args.device)  # 候选轨迹 [batch_size, 80, 5]
            neighbors_trajectories, scores, ego, weights = \
                decoder(encoder_outputs, second_stage_trajectory, inputs['neighbor_agents_past'], 80)
            # 累计总损失（第二阶段损失权重为0.2）
            loss += 0.2 * calc_loss(neighbors_trajectories, second_stage_trajectory, ego, scores, weights,
                              ego_gt_future, neighbors_gt_future, neighbors_future_valid)

            # 反向传播与梯度裁剪
            loss.backward()
            # 限制编码器和解码器的梯度范围，防止梯度爆炸或消失
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5.0) # 编码器梯度裁剪
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), 5.0) # 解码器梯度裁剪
            optimizer.step() # 更新模型参数，使损失函数最小化

            # 计算评估指标
            metrics = calc_metrics(second_stage_trajectory, neighbors_trajectories, scores,
                                   ego_gt_future, neighbors_gt_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            # 更新进度条的描述信息，显示当前批的平均损失值
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    # 汇总epoch指标
    epoch_metrics = np.array(epoch_metrics)
    # 计算平均指标（规划器ADE/FDE，预测器ADE/FDE）
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [planningADE, planningFDE, predictionADE, predictionFDE]
    logging.info(f"plannerADE: {planningADE:.4f}, plannerFDE: {planningFDE:.4f}, " +
                 f"predictorADE: {predictionADE:.4f}, predictorFDE: {predictionFDE:.4f}\n")
        
    return np.mean(epoch_loss), epoch_metrics


def valid_epoch(data_loader, encoder, decoder):
    epoch_loss = []
    epoch_metrics = []
    encoder.eval()
    decoder.eval()

    with tqdm(data_loader, desc="Validation", unit="batch") as data_epoch:
        for batch in data_epoch:
            # prepare data for predictor
            inputs = {
                'ego_agent_past': batch[0].to(args.device),
                'neighbor_agents_past': batch[1].to(args.device),
                'map_lanes': batch[2].to(args.device),
                'map_crosswalks': batch[3].to(args.device),
                'route_lanes': batch[4].to(args.device)
            }

            ego_gt_future = batch[5].to(args.device)
            neighbors_gt_future = batch[6].to(args.device)
            neighbors_future_valid = torch.ne(neighbors_gt_future[..., :3], 0)

            # predict
            with torch.no_grad():
                encoder_outputs = encoder(inputs)

                # first stage prediction
                first_stage_trajectory = batch[7].to(args.device)
                neighbors_trajectories, scores, ego, weights = \
                    decoder(encoder_outputs, first_stage_trajectory, inputs['neighbor_agents_past'], 30)

                loss = calc_loss(neighbors_trajectories, first_stage_trajectory, ego, scores, weights,
                                 ego_gt_future, neighbors_gt_future, neighbors_future_valid)

                # second stage prediction
                second_stage_trajectory = batch[8].to(args.device)
                neighbors_trajectories, scores, ego, weights = \
                    decoder(encoder_outputs, second_stage_trajectory, inputs['neighbor_agents_past'], 80)

                loss += 0.2 * calc_loss(neighbors_trajectories, second_stage_trajectory, ego, scores, weights,
                                  ego_gt_future, neighbors_gt_future, neighbors_future_valid)
 
            # compute metrics

            metrics = calc_metrics(second_stage_trajectory, neighbors_trajectories, scores,
                                   ego_gt_future, neighbors_gt_future, neighbors_future_valid)
            epoch_metrics.append(metrics)
            epoch_loss.append(loss.item())
            data_epoch.set_postfix(loss='{:.4f}'.format(np.mean(epoch_loss)))

    epoch_metrics = np.array(epoch_metrics)
    planningADE, planningFDE = np.mean(epoch_metrics[:, 0]), np.mean(epoch_metrics[:, 1])
    predictionADE, predictionFDE = np.mean(epoch_metrics[:, 2]), np.mean(epoch_metrics[:, 3])
    epoch_metrics = [planningADE, planningFDE, predictionADE, predictionFDE]
    logging.info(f"val-plannerADE: {planningADE:.4f}, val-plannerFDE: {planningFDE:.4f}, " +
                 f"val-predictorADE: {predictionADE:.4f}, val-predictorFDE: {predictionFDE:.4f}\n")

    return np.mean(epoch_loss), epoch_metrics


def model_training(args):
    # 初始化训练日志目录和日志文件
    log_path = f"./training_log/{args.name}/"
    os.makedirs(log_path, exist_ok=True)  # 自动创建日志目录, exist_ok=True 如果目录已存在则不报错
    initLogging(log_file=log_path+'train.log')  # 初始化日志系统

    # 记录训练参数信息
    logging.info("------------- {} -------------".format(args.name))
    logging.info("Batch size: {}".format(args.batch_size))
    logging.info("Learning rate: {}".format(args.learning_rate))
    logging.info("Use device: {}".format(args.device))

    # 设置随机种子保证实验可重复性
    set_seed(args.seed)

    # 初始化模型结构
    encoder = Encoder().to(args.device)  # 场景编码器（提取环境特征）
    logging.info("Encoder Params: {}".format(sum(p.numel() for p in encoder.parameters())))
    decoder = Decoder(neighbors=args.num_neighbors, max_branch=args.num_candidates, \
                      variable_cost=args.variable_weights).to(args.device)  # 轨迹解码器（生成预测轨迹）
    logging.info("Decoder Params: {}".format(sum(p.numel() for p in decoder.parameters())))

    # 配置优化器和学习率调度器
    optimizer = optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=args.learning_rate)  # 联合优化器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # 每5个epoch学习率减半

    # 准备训练数据加载器
    train_set = DrivingData(glob.glob(os.path.join(args.train_set, '*.npz')), args.num_neighbors, args.num_candidates)
    valid_set = DrivingData(glob.glob(os.path.join(args.valid_set, '*.npz')), args.num_neighbors, args.num_candidates)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=os.cpu_count())  # 多线程加载训练数据
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=os.cpu_count())  # 多线程加载验证数据
    logging.info("Dataset Prepared: {} train data, {} validation data\n".format(len(train_set), len(valid_set)))

    # 开始训练循环
    for epoch in range(args.train_epochs):
        # 执行单个epoch的训练和验证
        train_loss, train_metrics = train_epoch(train_loader, encoder, decoder, optimizer)
        val_loss, val_metrics = valid_epoch(valid_loader, encoder, decoder)

        # 保存训练日志到CSV文件
        log = {
            'epoch': epoch+1,  # 当前训练轮次
            'loss': train_loss,  # 训练损失
            'lr': optimizer.param_groups[0]['lr'],  # 当前学习率
            'val-loss': val_loss,  # 验证损失
            # 以下为规划器和预测器的评估指标
            'train-planningADE': train_metrics[0], 'train-planningFDE': train_metrics[1],
            'train-predictionADE': train_metrics[2], 'train-predictionFDE': train_metrics[3],
            'val-planningADE': val_metrics[0], 'val-planningFDE': val_metrics[1],
            'val-predictionADE': val_metrics[2], 'val-predictionFDE': val_metrics[3]
        }

        # 创建/追加日志文件
        if epoch == 0:
            with open(f'./training_log/{args.name}/train_log.csv', 'w') as csv_file: 
                writer = csv.writer(csv_file) 
                writer.writerow(log.keys())  # 写入表头
                writer.writerow(log.values())
        else:
            with open(f'./training_log/{args.name}/train_log.csv', 'a') as csv_file: 
                writer = csv.writer(csv_file)
                writer.writerow(log.values())  # 追加数据行

        # 更新学习率
        scheduler.step()  # 按预定策略调整学习率

        # 保存模型检查点
        model = {
            'encoder': encoder.state_dict(),  # 编码器参数
            'decoder': decoder.state_dict()  # 解码器参数
        }
        # 按验证指标命名模型文件（包含epoch数和验证ADE值）
        torch.save(model, f'training_log/{args.name}/model_epoch_{epoch+1}_valADE_{val_metrics[0]:.4f}.pth')
        logging.info(f"Model saved in training_log/{args.name}\n")


if __name__ == "__main__":
    dpath = ddp.dtpp_data_path()
    # Arguments
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--name', type=str, help='log name', default="DTPP_training")
    parser.add_argument('--seed', type=int, help='fix random seed', default=3407)
    # parser.add_argument('--train_set', type=str, help='path to training data')
    parser.add_argument('--train_set', type=str, help='path to training data', default=dpath+'processed_data/train')
    # parser.add_argument('--valid_set', type=str, help='path to validation data')
    parser.add_argument('--valid_set', type=str, help='path to validation data', default=dpath+'processed_data/val')

    parser.add_argument('--num_neighbors', type=int, help='number of neighbor agents to predict', default=10)
    parser.add_argument('--num_candidates', type=int, help='number of max candidate trajectories', default=30)
    parser.add_argument('--variable_weights', type=bool, help='use variable cost weights', default=False)
    parser.add_argument('--train_epochs', type=int, help='epochs of training', default=30)
    parser.add_argument('--batch_size', type=int, help='batch size', default=16)
    parser.add_argument('--learning_rate', type=float, help='learning rate', default=2e-4)
    parser.add_argument('--device', type=str, help='run on which device', default='cuda')
    args = parser.parse_args()

    # Run model training
    model_training(args)
