import matplotlib.pyplot as plt
import numpy as np
import scipy.special


def calc_4points_bezier_path(sx, sy, syaw, ex, ey, eyaw, offset, n_points=100):
    """ 四阶贝塞尔曲线路径生成器
    
    参数:
    sx -- 起点x坐标（自车后轴中心）
    sy -- 起点y坐标
    syaw -- 起点航向角（弧度）
    ex -- 终点x坐标（目标路径点）
    ey -- 终点y坐标
    eyaw -- 终点航向角（弧度）
    offset -- 曲线控制点缩放系数（值越大曲线越平缓）
    n_points -- 路径点数量（默认100点）
    
    返回:
    路径点数组（n_points×2），控制点数组（4×2）
    """
    # 计算控制点间距（基于起点到终点的直线距离）
    dist = np.hypot(sx - ex, sy - ey) / offset  # offset=3时，控制点间距约为总距离的1/3
    
    # 构建四阶贝塞尔控制点（起点方向控制点+终点方向控制点）
    control_points = np.array([
        [sx, sy],  # 起点
        [sx + dist * np.cos(syaw), sy + dist * np.sin(syaw)],  # 沿起点航向延伸
        [ex - dist * np.cos(eyaw), ey - dist * np.sin(eyaw)],  # 沿终点航向反向延伸
        [ex, ey]   # 终点
    ])
    
    # 生成贝塞尔曲线路径
    path = calc_bezier_path(control_points, n_points=n_points)
    
    return path, control_points


def calc_bezier_path(control_points, n_points=100):
    """
    Compute bezier path (trajectory) given control points.

    :param control_points: (numpy array)
    :param n_points: (int) number of points in the trajectory
    :return: (numpy array)
    """
    traj = []
    for t in np.linspace(0, 1, n_points):
        traj.append(bezier(t, control_points))

    return np.array(traj)


def bernstein_poly(n, i, t):
    """
    Bernstein polynom.

    :param n: (int) polynom degree
    :param i: (int)
    :param t: (float)
    :return: (float)
    """
    return scipy.special.comb(n, i) * t ** i * (1 - t) ** (n - i)


def bezier(t, control_points):
    """
    Return one point on the bezier curve.

    :param t: (float) number in [0, 1]
    :param control_points: (numpy array)
    :return: (numpy array) Coordinates of the point
    """
    n = len(control_points) - 1
    return np.sum([bernstein_poly(n, i, t) * control_points[i] for i in range(n + 1)], axis=0)


def bezier_derivatives_control_points(control_points, n_derivatives):
    """
    Compute control points of the successive derivatives of a given bezier curve.

    A derivative of a bezier curve is a bezier curve.
    See https://pomax.github.io/bezierinfo/#derivatives
    for detailed explanations

    :param control_points: (numpy array)
    :param n_derivatives: (int)
    e.g., n_derivatives=2 -> compute control points for first and second derivatives
    :return: ([numpy array])
    """
    w = {0: control_points}
    for i in range(n_derivatives):
        n = len(w[i])
        w[i + 1] = np.array([(n - 1) * (w[i][j + 1] - w[i][j])
                             for j in range(n - 1)])
    return w


def curvature(dx, dy, ddx, ddy):
    """
    Compute curvature at one point given first and second derivatives.

    :param dx: (float) First derivative along x axis
    :param dy: (float)
    :param ddx: (float) Second derivative along x axis
    :param ddy: (float)
    :return: (float)
    """
    return (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** (3 / 2)