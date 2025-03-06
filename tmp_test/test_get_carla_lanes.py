import sys
import os
# import logging
from custom_format import *
import glob
import carla
import matplotlib.pyplot as plt
import time
import numpy as np

from typing import List


# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
    logger.debug('Adding CARLA egg to PYTHONPATH: %s' % sys.path[-1])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

# 连接 Carla 服务端
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
carla_map = world.get_map()
logger.info("Carla Map: {}".format(carla_map.name))

SAMPLE_RESOLUTION = 0.5

def get_lane_center_lines(map_obj, sampling_resolution=SAMPLE_RESOLUTION, timeout=20.0):
    """
    获取所有车道的中心线坐标
    :param map_obj: carla.Map 对象
    :param sampling_resolution: 路径点采样间隔（单位：米）
    :param timeout: 超时时间（单位：秒）
    :return: 车道中心线列表，每条线为 [(x1,y1), (x2,y2), ...]
    """
    # 获取所有道路段的拓扑结构（起点和终点路径点）
    topology = map_obj.get_topology()
    logger.info("Topology: {}".format(topology))
    
    lane_lines, junction_lines = [], []
    
    for i, segment in enumerate(topology):
        start_wp, end_wp = segment[0], segment[1]
        # lane_points = get_lane_points(sampling_resolution, start_wp, end_wp, i, timeout)        
        lane_points = start_wp.next_until_lane_end(sampling_resolution) # 或者上面一行代码
        junction_points = [wp for wp in lane_points if wp.is_junction]
        
        lane_lines.append([(wp.transform.location.x, wp.transform.location.y) for wp in lane_points])
        junction_lines.append([(wp.transform.location.x, wp.transform.location.y) for wp in junction_points])
    
    return lane_lines, junction_lines

def get_lane_points(sampling_resolution, start_wp, end_wp, i = None, timeout=20.0):
    current_wp = start_wp
        
        # 沿着车道生成连续的路径点
    lane_points = []
    start_time = time.time()
    while True:
            # 检查超时
        if time.time() - start_time > timeout:
            logger.warning(f"--- {i} ---超时，停止生成路径点")
            break
            
            # 添加当前路径点的位置（忽略Z轴）
            # loc = current_wp.transform.location
        lane_points.append(current_wp)  # Y轴取反，适配常规坐标系
        logger.info(f"current_wp: {current_wp}")
            
            # 到达终点或车道ID变化时停止
        if current_wp.id == end_wp.id:
            break
            
            # 获取下一个路径点（沿车道方向）
        next_wps = current_wp.next(sampling_resolution)
        if not next_wps:
            break
        current_wp = next_wps[0]
        if current_wp.transform.location.distance(end_wp.transform.location) < sampling_resolution:
            break
    return lane_points

def deduplicate_markings(markings_list, tolerance=0.1):
    """
    由于相邻路径点的车道线可能重复，可通过坐标或ID去重：
    :param markings_list: 车道线列表
    :param tolerance: 允许的误差范围（单位：米）
    :return: 去重后的车道线列表"""
    unique_markings = []
    seen_locations = []
    
    for marking in markings_list:
        loc = marking['location']
        # 检查是否已有近似坐标的车道线
        is_duplicate = False
        for seen in seen_locations:
            if np.linalg.norm(np.array([loc.x, loc.y, loc.z]) - np.array([seen.x, seen.y, seen.z])) < tolerance:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_markings.append(marking)
            seen_locations.append(loc)
    return unique_markings

def get_lane_boudary(carla_map):
    left_lane_markings = []
    right_lane_markings = []

    for segment in carla_map.get_topology():
        start_waypoint, end_waypoint = segment[0], segment[1]
        
        # 沿路段生成连续的Waypoints（间隔1米）
        # waypoints = carla_map.generate_waypoints(start_waypoint.transform.location, end_waypoint.transform.location, 1.0)
        # waypoints = get_lane_points(SAMPLE_RESOLUTION, start_waypoint, end_waypoint)
        waypoints = start_waypoint.next_until_lane_end(SAMPLE_RESOLUTION) # 或者上面的函数
        
        for waypoint in waypoints:
            # 提取左侧车道线
            left_marking = waypoint.left_lane_marking
            if left_marking:
                left_lane_markings.append({
                    "type": str(left_marking.type),      # 类型（Solid, Broken等）
                    "color": str(left_marking.color),    # 颜色（White, Yellow等）
                    "location": left_marking.world_location,  # 车道线位置（世界坐标）
                    "width": left_marking.width          # 宽度（米）
                })
            
            # 提取右侧车道线
            right_marking = waypoint.right_lane_marking
            if right_marking:
                right_lane_markings.append({
                    "type": str(right_marking.type),
                    "color": str(right_marking.color),
                    "location": right_marking.world_location,
                    "width": right_marking.width
                })
    # 去重处理
    left_lane_markings = deduplicate_markings(left_lane_markings)
    right_lane_markings = deduplicate_markings(right_lane_markings)
    return left_lane_markings, right_lane_markings


def get_traffic_lights_info(world):
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    traffic_lights_info = []
    for tl in traffic_lights:
        transform = tl.get_transform()
        state = tl.get_state()
        stop_waypoints = tl.get_stop_waypoints()
        affected_lane_points = tl.get_affected_lane_waypoints()
        traffic_lights_info.append((transform.location.x, transform.location.y, state, stop_waypoints, affected_lane_points))
    return traffic_lights_info

def get_corsswalks_info(carla_map: carla.Map):
    corsswalks = carla_map.get_crosswalks()
    corsswalks_info: List[List[(float, float)]] = []
    crosswalk : List[(float, float)] = []
    last_loc = None

    for loc in corsswalks:
        if last_loc and loc.distance(last_loc) < 30 and len(crosswalk) < 5:
            crosswalk.append((loc.x, loc.y))
        else:
            if crosswalk:
                corsswalks_info.append(crosswalk)
            crosswalk = [(loc.x, loc.y)]
            last_loc = loc

    if crosswalk:
        corsswalks_info.append(crosswalk)

    return corsswalks_info

def plot_lane_center_lines(lane_lines, juntion_lines, traffic_lights_info, cross_walks, left_lane_markings, right_lane_markings):
    # plt.figure(figsize=(12, 12))
    fig, ax = plt.subplots()
    
    for lane in lane_lines:
        x = [point[0] for point in lane]
        y = [point[1] for point in lane]
        
        # plt.plot(x, y, 'b.', linewidth=0.5)
        plt.plot(x, y, 'k--')
        # 画出起始点和终点，用不同的颜色
        # plt.plot(x[0], y[0], 'r.')
        plt.plot(x[-1], y[-1], 'b.')
        logger.info(f"lane: {lane}")
        
    # 绘制junction
    for junction in juntion_lines:
        x = [point[0] for point in junction]
        y = [point[1] for point in junction]
        plt.plot(x, y, 'r--', linewidth=2)
        
    for info in traffic_lights_info:
        x, y, state, stop_waypoints, affected_lane_wp = info
        if state == carla.TrafficLightState.Red:
            color = 'r'
        elif state == carla.TrafficLightState.Yellow:
            color = 'y'
        elif state == carla.TrafficLightState.Green:
            color = 'g'
        else:
            color = 'k'
        ax.scatter(x, y, c=color, marker='X', label='Traffic Light', linewidth=8) # 取负值y，左手系
        stpwx = [wp.transform.location.x for wp in stop_waypoints]
        stpwy = [wp.transform.location.y for wp in stop_waypoints] # 取负值y，左手系
        plt.plot(stpwx, stpwy, 'k')
        

        
        stop_ids = [wp.id for wp in stop_waypoints]
        
        aflpwx = [wp.transform.location.x for wp in affected_lane_wp]
        aflpwy = [wp.transform.location.y for wp in affected_lane_wp]
        plt.plot(aflpwx, aflpwy, 'ys')
        
        # stop_ids = [{"is_junction": wp.is_junction, "id": wp.id, "road_id": wp.lane_id, "section_id":wp.section_id, "lane_id": wp.lane_id, "junction_id": wp.junction_id} for wp in stop_waypoints]
        # affected_ids = [{"is_junction": wp.is_junction, "id": wp.id, "road_id": wp.lane_id, "section_id":wp.section_id, "lane_id": wp.lane_id, "junction_id": wp.junction_id} for wp in affected_lane_wp]
        # for stop_id in stop_ids:
        #     print(f"stop_id: {stop_id}")
        
        # for affected_id in affected_ids:
        #     print(f"affected_id: {affected_id}")
        
        # # 判断stop line 是否属于junction
        # is_juction = False
        # for wp in stop_waypoints:
        #     if wp.is_junction:
        #         is_juction = True
        #         break
        # print(f"stop line is_juction: {is_juction}")
        # flag = True
        # for wp in affected_lane_wp:
        #     if not wp.is_junction:
        #         flag = False
        #         break
        # print(f"affected lane is_juction: {flag}")
        
        # 画出关联的红绿灯
        # 直线链接红绿灯和停止点
        for i in range(len(stpwx)):
            plt.plot([x, stpwx[i]], [y, stpwy[i]], color=color)
            plt.text(stpwx[i], stpwy[i], 's', color='k', fontsize=20)
        # 直线链接红绿灯和影响车道
        for i in range(len(aflpwx)):
            plt.plot([x, aflpwx[i]], [y, aflpwy[i]], color=color)
            plt.text(aflpwx[i], aflpwy[i], 'a', color='y', fontsize=20)
            
        # cross_walks
        for cw in cross_walks:
            x = [point[0] for point in cw]
            y = [point[1] for point in cw]
            plt.plot(x, y, 'g-', linewidth=2)
        
        
    # # 绘制车道线边界线
    # for lb in left_lane_markings:
    #     x = lb['location'].x
    #     y = -lb['location'].y # 取负值y，左手系
    #     plt.plot(x, y, 'r.', markersize=0.5)   
    
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('DTPP Carla Traffic Map')
    plt.axis('equal')  # 保持比例一致
    plt.show()

# 主程序
if __name__ == '__main__':
    logger.basicConfig(level=logger.INFO)
    lane_center_lines, juntion_lines = get_lane_center_lines(carla_map)
    traffic_light_info = get_traffic_lights_info(world)
    cross_walks = get_corsswalks_info(carla_map)
    # lb, rb = get_lane_boudary(carla_map)
    lb, rb = [], []
    plot_lane_center_lines(lane_center_lines, juntion_lines, traffic_light_info, cross_walks, lb, rb)
