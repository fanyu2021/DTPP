import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from custom_format import *
logger = create_colored_logger(__name__)


def cubic_spline_coefficients(x0, dx0, xf, dxf, tf):
    return (x0, dx0, -2 * dx0 / tf - dxf / tf - 3 * x0 / tf ** 2 + 3 * xf / tf ** 2,
            dx0 / tf ** 2 + dxf / tf ** 2 + 2 * x0 / tf ** 3 - 2 * xf / tf ** 3)


def compute_spline_xyvaqrt(v0, dv0, vf, tf, path, N, offset):
    t = torch.arange(N+1).to(v0.device) * tf / N
#    logger.debug(f't = {t}')
    tp = t[..., None] ** torch.arange(4).to(v0.device)
    dtp = t[..., None] ** torch.tensor([0, 0, 1, 2]).to(v0.device) * torch.arange(4).to(v0.device)
    
    coefficients = cubic_spline_coefficients(v0, dv0, vf, 0, tf)
    coefficients = torch.stack(coefficients).unsqueeze(-1) # [4, 1], 增加一个维度

    v = tp @ coefficients
    a = dtp @ coefficients
    s = torch.cumsum(v * tf / N, dim=0) # 沿着行方向累加，得到累积位移
    # logger.warning(f's={s}')
    s = torch.cat((torch.zeros(1, 1).to(v0.device), s[:-1]), dim=0)
    s += offset
    # logger.warning(f's={s}')
    # logger.warning(f's.shape={s.shape}')
    i = (s / 0.1).long()
    # logger.debug(f'i.shape = {i.shape}')
    # logger.debug(f'i={i}')
    end_ind = path.shape[0] - 1
    if i[-1] > end_ind:
        # logger.warning(f'--- i[-1]: {i[-1]}, path.shape[0] - 1: {path.shape[0] - 1}')
    #     # i[end_ind+1:] = i[end_ind]
    #     return
        i_ind = len(i)-1
        for j in range(i_ind, -1, -1):
            if i[j] < end_ind:
                break
        # logger.debug(f'j = {j}')
        i[j:] = i[j]
        v[j:,0] = v[j,0]
        a[j:,0] = a[j,0]
    # ind = min(i[-1], path.shape[0]-1)
    # i = i[:ind]
    # logger.debug(f'i = {i}')

    x = path[i, 0]
    y = path[i, 1]
    yaw = path[i, 2]
    r = path[i, 3]
    # v = v[i]
    # a = a[i]
    # t = t[i]

    return torch.cat((x, y, yaw, v, a, r, t.unsqueeze(-1)), -1).squeeze(0)


class SplinePlanner:
    def __init__(self, first_stage_horizion, horizon):
        self.spline_order = 3
        self.max_curve = 0.3
        self.max_lat_acc = 3.0
        self.acce_bound = [-5, 3]
        self.vbound = [0, 15.0]
        self.first_stage_horizion = first_stage_horizion
        self.horizon = horizon

    def calc_trajectory(self, v0, a0, vf, tf, path, N_seg, offset=0):
        traj = compute_spline_xyvaqrt(v0, a0, vf, tf, path, N_seg, offset)

        return traj

    def gen_short_term_trajs(self, x0, tf, paths, dyn_filter):
        xf_set = []
        trajs = []
        
        # generate speed profile and trajectories
        for path in paths:
            path = torch.from_numpy(path).to(x0.device).type(torch.float)
            for v in self.v_grid:
                # 根据不同的终点采样速度，计算轨迹，path提供横向路径
                traj = self.calc_trajectory(x0[3], x0[4], v, tf, path, self.first_stage_horizion*10) # [x, y, yaw, v, a, r, t] 31 x 7
                if traj is None:
                    continue

                xf = traj[-1, :2]
                if xf_set and torch.cdist(xf.unsqueeze(0), torch.stack(xf_set)).min() < 0.5: # 判断终点是否过近，如果位置过近，则跳过
                    continue
                else:
                    xf_set.append(xf)
                    trajs.append(traj)
                    # logger.warning(f'short final:{traj[-1, :2]}')

        trajs = torch.stack(trajs)
        
        # remove trajectories that are not feasible
        if dyn_filter:
            feas_flag = self.feasible_flag(trajs)
            trajs = trajs[feas_flag]

        return trajs
    
    def plot_paths(self, paths):
        for path in paths:
            x = [pt[0] for pt in path]
            y = [pt[1] for pt in path]
            plt.plot(x, y)
            plt.show()

    def write_paths_json(self, paths):
        import json

        traj = {}
        trajs = []
        for path in paths:
            traj['x'] = [pt[0] for pt in path]
            traj['y'] = [pt[1] for pt in path]
            trajs.append(traj)

        json_str = json.dumps(trajs)
        byte_data = json_str.encode()

        # 打开文件，使用 'w' 模式表示写入        
        # 使用 write 方法写入内容
        with open('debug/paths_json.bin', 'wb') as file:
            # 将字典转换为 JSON 字符串
            file.write(byte_data)



    
    def gen_long_term_trajs(self, x0, tf, paths, dyn_filter):
        xf_set = []
        trajs = []
        # generate speed profile and trajectories
        for path in paths:
            path = torch.from_numpy(path).to(x0.device).type(torch.float)
            dist = torch.norm(path[:, :2] - x0[:2], dim=1)
            # logger.info(f'dist: {dist}')
            if dist.min() > 0.12:
                logger.debug("--- current path is not on current pos!")
                continue
            
            offset = torch.argmin(dist) * 0.1
            # logger.warning(f'offset: {offset}')
            # logger.warning(f'--- path min_dis: {path[torch.argmin(dist)]}')

            for v in self.v_grid:
                traj = self.calc_trajectory(x0[3], x0[4], v, tf, path, (self.horizon-self.first_stage_horizion)*10, offset) # [x, y, yaw, v, a, r, t]
                if traj is None:
                    logger.debug('--- calculate trajectory failed!')
                    continue

                xf = traj[-1, :2]
  
                if xf_set and torch.cdist(xf.unsqueeze(0), torch.stack(xf_set)).min() < 0.5:
                    continue
                else:
                    xf_set.append(xf)
                    trajs.append(traj)

        if len(trajs) == 0:
            self.write_paths_json(paths)
            print(f'len(trajs) == 0')
            return
        else:
            trajs = torch.stack(trajs)
        
        # remove trajectories that are not feasible
        if dyn_filter:
            feas_flag = self.feasible_flag(trajs)
            trajs = trajs[feas_flag]

        return trajs

    def feasible_flag(self, trajs):
        feas_flag = ((trajs[:, 1:, 3] >= self.vbound[0]) & 
                     (trajs[:, 1:, 3] <= self.vbound[1]) &
                     (trajs[:, 1:, 4] >= self.acce_bound[0]) & 
                     (trajs[:, 1:, 4] <= self.acce_bound[1]) &
                     (trajs[:, 1:, 5].abs() * trajs[:, 1:, 3] ** 2 <= self.max_lat_acc) &
                     (trajs[:, 1:, 5].abs() <= self.max_curve)
                    ).all(1)

        if feas_flag.sum() == 0:
            print("No feasible trajectory")
            feas_flag = torch.ones(trajs.shape[0], dtype=torch.bool).to(trajs.device)
        
        return feas_flag

    def gen_trajectories(self, x0, tf, paths, speed_limit, is_root):
        # generate trajectories
        v0 = x0[3]

        if is_root:
            v_min = max(v0 - 4.0 * tf, 0.0)
            v_max = min(v0 + 1 * tf, speed_limit)
            self.v_grid = torch.linspace(v_min, v_max, 10).to(x0.device)
            trajs = self.gen_short_term_trajs(x0, tf, paths, dyn_filter=False)
            assert trajs is not None, "No feasible short term trajectory"
        else:
            v_min = max(v0 - tf, 0.0)
            v_max = min(v0 + 0.1*tf, speed_limit)
            logger.debug(f"v_min: {v_min}, v_max: {v_max}, v0: {v0}, tf: {tf}, speed_limit: {speed_limit}")
            self.v_grid = torch.linspace(v_min, v_max, 5).to(x0.device)
            trajs = self.gen_long_term_trajs(x0, tf, paths, dyn_filter=False)
            assert trajs is not None, "No feasible long term trajectory"

        if trajs is None:
            logger.error(f"No long feasible trajectory")
            return None
        # adjust timestep
        if (not is_root):
            trajs[:, :, -1] += self.horizon - self.first_stage_horizion

        # remove the first time step
        trajs = trajs[:, 1:]

        return trajs
