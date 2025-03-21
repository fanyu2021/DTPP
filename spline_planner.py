import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d


def cubic_spline_coefficients(x0, dx0, xf, dxf, tf):
    return (x0, dx0, -2 * dx0 / tf - dxf / tf - 3 * x0 / tf ** 2 + 3 * xf / tf ** 2,
            dx0 / tf ** 2 + dxf / tf ** 2 + 2 * x0 / tf ** 3 - 2 * xf / tf ** 3)


def compute_spline_xyvaqrt(v0, dv0, vf, tf, path, N, offset):
    """ 三次样条轨迹计算核心函数
    
    参数:
    v0 -- 初始速度（m/s）
    dv0 -- 初始加速度（m/s²）
    vf -- 终点速度（m/s）
    tf -- 轨迹总时间（秒）
    path -- 参考路径 [n,4]（x,y,heading,curvature）
    N -- 轨迹分段数量（总点数=N+1）
    offset -- 路径起始偏移量（米）
    
    返回:
    轨迹张量 [N+1,7] 包含以下维度：
    x, y, 航向角, 速度, 加速度, 曲率, 时间戳
    """
    # 时间序列生成（均匀采样）
    t = torch.arange(N+1).to(v0.device) * tf / N  # 生成时间序列 [0, tf] 共N+1个点
    tp = t[..., None] ** torch.arange(4).to(v0.device)  # 时间多项式 [t^0, t^1, t^2, t^3]
    dtp = t[..., None] ** torch.tensor([0, 0, 1, 2]).to(v0.device) * torch.arange(4).to(v0.device)  # 导数多项式
    
    # 计算三次样条系数（满足起始和终止条件）
    coefficients = cubic_spline_coefficients(v0, dv0, vf, 0, tf)  # 获取三次样条系数
    coefficients = torch.stack(coefficients).unsqueeze(-1)  # 重塑为矩阵运算格式 [4,1]
    
    # 速度/加速度计算（向量化运算）
    v = tp @ coefficients  # 速度曲线 v(t) = a + b*t + c*t² + d*t³
    a = dtp @ coefficients  # 加速度曲线 a(t) = b + 2c*t + 3d*t²
    
    # 位移积分计算（用于路径采样）
    s = torch.cumsum(v * tf / N, dim=0)  # 数值积分计算纵向位移（v*Δt 的累加），相当于st图的终点s长度
    s = torch.cat((torch.zeros(1, 1).to(v0.device), s[:-1]), dim=0)  # 位移对齐（当前速度对应下一时段的位移）
    s += offset  # 应用路径偏移量
    
    # 路径采样（基于位移量）
    i = (s / 0.1).long()  # 0.1米为路径点间隔的采样分辨率
    if i[-1] > path.shape[0] - 1:  # 终点超出路径长度时返回空
        return
    
    # 提取路径参数
    x = path[i, 0]  # x坐标采样
    y = path[i, 1]  # y坐标采样
    yaw = path[i, 2]  # 航向角采样
    r = path[i, 3]  # 路径曲率采样
    
    # 合并轨迹数据
    # 速度/加速度的维度说明
    v = tp @ coefficients  # 形状: (N+1, 1)  例如：31个时间步 x 1列
    a = dtp @ coefficients  # 形状: (N+1, 1) 同上
    
    # 最终合并后的轨迹维度
    return torch.cat((x, y, yaw, v, a, r, t.unsqueeze(-1)), -1).squeeze(0)  
    # 输出形状: (N+1, 7)  例如：31个时间步 x 7个状态量


class SplinePlanner:
    def __init__(self, first_stage_horizion, horizon):
        """ 样条轨迹规划器初始化
        
        参数说明:
        first_stage_horizion -- 第一阶段规划时长（秒）
        horizon -- 总规划时长（秒）
        
        运动学约束参数:
        - spline_order: 3次样条曲线用于轨迹插值
        - max_curve: 0.3弧度/米（最大路径曲率，限制转弯半径）
        - max_lat_acc: 3.0 m/s²（最大横向加速度，保证舒适性）
        - acce_bound: [-5,3] m/s²（纵向加速度范围，急刹/正常加速）
        - vbound: [0,15.0] m/s（速度限制范围，约0-54km/h）
        
        时间规划参数:
        - first_stage_horizion: 初始规划阶段时长（通常3秒）
        - horizon: 总规划时长（通常8秒）
        """
        self.spline_order = 3         # 三次样条插值阶数
        self.max_curve = 0.3          # 最大路径曲率（1/m）
        self.max_lat_acc = 3.0        # 最大横向加速度（m/s²）
        self.acce_bound = [-5, 3]     # 纵向加速度边界 [min, max]（m/s²）
        self.vbound = [0, 15.0]       # 速度边界 [min, max]（m/s）
        self.first_stage_horizion = first_stage_horizion  # 初始阶段时长
        self.horizon = horizon        # 总规划时长

    def calc_trajectory(self, v0, a0, vf, tf, path, N_seg, offset=0):
        """ 轨迹计算包装方法
        
        参数:
        v0 -- 初始速度（m/s）
        a0 -- 初始加速度（m/s²）
        vf -- 目标终点速度（m/s）
        tf -- 轨迹总时间（秒）
        path -- 参考路径 [n,4]（包含x,y,heading,curvature）
        N_seg -- 轨迹分段数量（总点数=N_seg+1）
        offset -- 路径起始点偏移量（米）
        
        返回:
        轨迹张量 [N_seg+1, 7] 包含：
        x, y, 航向角, 速度, 加速度, 曲率, 时间戳
        """
        traj = compute_spline_xyvaqrt(v0, a0, vf, tf, path, N_seg, offset)
        return traj

    def gen_short_term_trajs(self, x0, tf, paths, dyn_filter):
        """ 短期候选轨迹生成器（3秒级详细轨迹）
        
        参数:
        x0 -- 初始状态张量 [x, y, 航向角, 速度, 加速度]
        tf -- 轨迹持续时间（秒）
        paths -- 候选路径列表（来自路径规划模块）
        dyn_filter -- 是否启用动力学约束过滤
        
        返回:
        候选轨迹张量 [轨迹数, 时间步数, 7个状态维度]
        """
        xf_set = []  # 轨迹终点去重集合
        trajs = []    # 候选轨迹容器
        
        # 遍历所有候选路径
        for path in paths:
            # 转换路径为GPU张量（加速计算）
            path = torch.from_numpy(path).to(x0.device).type(torch.float)
            
            # 遍历速度采样网格生成轨迹
            for v in self.v_grid:
                # 生成单条轨迹（31个时间步，对应3秒轨迹）
                traj = self.calc_trajectory(x0[3], x0[4], v, tf, path, self.first_stage_horizion*10)
                if traj is None:  # 无效轨迹过滤
                    continue

                # 轨迹终点去重检测（0.5米阈值）
                xf = traj[-1, :2] # # 取轨迹最后一个点的x,y坐标
                if xf_set and torch.cdist(xf.unsqueeze(0), torch.stack(xf_set)).min() < 0.5:
                    continue
                else:
                    xf_set.append(xf)
                    trajs.append(traj)

        # 合并所有轨迹为三维张量 [n_traj, n_steps, 7]
        trajs = torch.stack(trajs)
        
        # 动力学可行性过滤（加速度/曲率等约束）
        if dyn_filter:
            feas_flag = self.feasible_flag(trajs)
            trajs = trajs[feas_flag]

        return trajs

    def gen_long_term_trajs(self, x0, tf, paths, dyn_filter):
        """ 长期候选轨迹生成器（5秒级扩展轨迹）
        
        参数:
        x0 -- 初始状态张量 [x, y, 航向角, 速度, 加速度]
        tf -- 轨迹持续时间（秒）
        paths -- 候选路径列表（来自路径规划模块）
        dyn_filter -- 是否启用动力学约束过滤
        
        返回:
        候选轨迹张量 [轨迹数, 时间步数, 7个状态维度]
        """
        xf_set = []  # 轨迹终点坐标集合（用于去重）
        trajs = []    # 候选轨迹容器
        
        # 遍历所有候选路径
        for path in paths:
            # 转换路径为PyTorch张量并移至对应设备
            path = torch.from_numpy(path).to(x0.device).type(torch.float)
            
            # 计算路径点与自车的欧式距离
            dist = torch.norm(path[:, :2] - x0[:2], dim=1)
            if dist.min() > 0.1:  # 过滤距离最近的路径点超过0.1米的路径
                continue
            
            # 计算路径起点偏移量（找到最近路径点索引，乘以0.1米精度）
            offset = torch.argmin(dist) * 0.1

            # 遍历速度采样网格生成轨迹
            for v in self.v_grid:
                # 调用轨迹计算核心函数（生成50个时间步的轨迹）
                traj = self.calc_trajectory(x0[3], x0[4], v, tf, path, 
                                           (self.horizon-self.first_stage_horizion)*10, offset)
                if traj is None:  # 无效轨迹过滤
                    continue

                # 提取轨迹终点坐标（用于去重检测）
                xf = traj[-1, :2]
  
                # 轨迹终点去重检测（0.5米阈值）
                if xf_set and torch.cdist(xf.unsqueeze(0), torch.stack(xf_set)).min() < 0.5:
                    continue  # 跳过终点过于接近的轨迹
                else:
                    xf_set.append(xf)
                    trajs.append(traj)

        # 处理空轨迹情况
        if len(trajs) == 0:
            return
        else:
            trajs = torch.stack(trajs)  # 合并为三维张量 [n_traj, n_steps, 7]
        
        # 动力学可行性过滤（加速度/曲率等约束）
        if dyn_filter:
            feas_flag = self.feasible_flag(trajs)
            trajs = trajs[feas_flag]

        return trajs

    def feasible_flag(self, trajs):
        """ 轨迹可行过滤器
        
        参数:
        trajs -- 候选轨迹张量 [n_traj, n_steps, 7]
        
        返回:
        布尔掩码张量 [n_traj] 表示各轨迹是否满足动力学约束
        
        约束条件（需同时满足）:
        1. 速度范围约束: v ∈ [0, 15.0] m/s
        2. 纵向加速度约束: a ∈ [-5, 3] m/s² 
        3. 横向加速度约束: |curvature * v²| ≤ 3.0 m/s²
        4. 路径曲率约束: |curvature| ≤ 0.3 rad/m
        """
        # 多条件联合判断（从第2个时间步开始检验）
        feas_flag = ((trajs[:, 1:, 3] >= self.vbound[0]) &  # 速度下限
                     (trajs[:, 1:, 3] <= self.vbound[1]) &   # 速度上限
                     (trajs[:, 1:, 4] >= self.acce_bound[0]) &  # 加速度下限
                     (trajs[:, 1:, 4] <= self.acce_bound[1]) &  # 加速度上限
                     (trajs[:, 1:, 5].abs() * trajs[:, 1:, 3] ** 2 <= self.max_lat_acc) &  # 横向加速度公式
                     (trajs[:, 1:, 5].abs() <= self.max_curve)  # 路径曲率约束
                    ).all(1)  # 要求所有时间步满足约束

        # 保底逻辑（当没有可行轨迹时返回全True避免程序中断）
        if feas_flag.sum() == 0:
            print("No feasible trajectory")
            feas_flag = torch.ones(trajs.shape[0], dtype=torch.bool).to(trajs.device)
        
        return feas_flag

    def gen_trajectories(self, x0, tf, paths, speed_limit, is_root):
        """ 多模式轨迹生成入口函数
        
        参数:
        x0 -- 初始状态 [x, y, heading, speed, accel]
        tf -- 轨迹持续时间（秒）
        paths -- 候选路径列表（来自上层路径规划）
        speed_limit -- 道路限速（m/s）
        is_root -- 是否根节点（首层轨迹需要更广的速度采样）
        
        返回:
        候选轨迹集合（张量）[num_traj, time_steps, 7个维度]
        7个维度包含：x, y, 航向角, 速度, 加速度, 曲率, 时间戳
        """
        # 提取当前速度用于速度规划
        v0 = x0[3]  # x0[3]对应初始速度值

        # 根节点轨迹生成（初始决策层）
        if is_root:
            # 速度采样范围：考虑紧急制动和加速场景
            v_min = max(v0 - 4.0 * tf, 0.0)  # 最大减速度4m/s²
            v_max = min(v0 + 2.4 * tf, speed_limit)  # 最大加速度2.4m/s²
            # 生成10个均匀速度采样点
            self.v_grid = torch.linspace(v_min, v_max, 10).to(x0.device)
            # 调用短期轨迹生成器（3秒轨迹）
            trajs = self.gen_short_term_trajs(x0, tf, paths, dyn_filter=False)
        # 子节点轨迹生成（后续层）
        else:
            # 速度采样范围：保守策略
            v_min = max(v0 - tf, 0.0)       # 1m/s²减速度边界
            v_max = min(v0 + tf, speed_limit)  # 1m/s²加速度边界
            # 生成5个速度采样点
            self.v_grid = torch.linspace(v_min, v_max, 5).to(x0.device)
            # 调用长期轨迹生成器（5秒轨迹）
            trajs = self.gen_long_term_trajs(x0, tf, paths, dyn_filter=False)

        # 时间戳对齐（保证多阶段轨迹时间连续性）
        if not is_root:
            # 时间戳向后偏移（首阶段3秒 + 当前阶段），每个点位都加5s
            trajs[:, :, -1] += self.horizon - self.first_stage_horizion

        # 移除初始时刻状态（避免与父节点轨迹重复）
        trajs = trajs[:, 1:]  # 维度变为 [num_traj, time_steps-1, 7]

        return trajs
