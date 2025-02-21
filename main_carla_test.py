#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : test_9.py

"""
实现动态障碍物的感知，进行路径规划和速度规划
1.获取自车匹配点的索引
2.根据匹配点的索引在全局路径上采样81个点
3.对采样点进行平滑
4.根据采样得到的局部路径为参考线，将根据对动态障碍物的预测，判断自车会在什么时间，什么地方遭遇障碍物
5.形成虚拟障碍物，采用路径规划避障（S-L + DP + QP）
6.构建S-T图，为新的路径规划速度信息
7.控制器根据规划得到的新的路径和速度信息引导自车躲避动态障碍物
8.后面可能还需要对规划的轨迹进行拼接和平滑
"""

import carla

import multiprocessing
import time
import numpy as np
import matplotlib.pyplot as plt # 画图



from controller.controller import Vehicle_control
# from planner.global_planning import global_path_planner
from carla_planner.global_planning import global_path_planner
# from planner.global_planning import global_path_planner
from sensors.Sensors_detector_lib import Obstacle_detector

import pygame

# nuplan 数据
import torch
import argparse


from dtpp_planner import DTPPPlanner
from carla_planner.carla_utils import *
from carla_planner import planning_utils
from motion_planning.motion_planning import motion_planning_e2e, motion_planning_rule


# 初始化 Pygame
pygame.init()

# 设置 Pygame 窗口大小
display_width = 1920 * 0.6
display_height = 1080 * 0.6
screen = pygame.display.set_mode((display_width, display_height))
pygame.display.set_caption("CARLA DTPP Planning and Control")

def global_routing(amap: carla.Map, start_tsf: carla.Transform, end_tsf: carla.Transform, debug: carla.DebugHelper,plot: bool = False):
    global_route_plan = global_path_planner(world_map=amap, sampling_resolution=0.5)  # 实例化全局规划器
    
    # # 手动设置起始点和终点
    # start_node_id = 326
    # end_node_id = 313
    # route_start_location, route_end_location = global_route_plan.get_start_end_waypoint_by_node_id(start_node_id, end_node_id)

    
    # 1. 规划路径，输出的每个路径点是一个元组形式[(wp, road_option), ...]第一个是元素是carla中的路点，第二个是当前路点规定的一些车辆行为
    routing_pathway = global_route_plan.routing_search(origin=start_tsf.location, destination=end_tsf.location)
    # 2. 将路径点构成的路径转换为[(x, y, theta, kappa], ...]的形式
    global_frenet_routing = planning_utils.waypoint_list_2_target_path(routing_pathway)
    # 3. 获取道路 id set
    road_ids: set = set()
    for path_item in routing_pathway:
        road_ids.add(path_item[0].road_id)
        debug.draw_point(path_item[0].transform.location, size=0.1, color=carla.Color(), life_time=0,)
        # debug.draw_string(path_item[0].transform.location, 'x', draw_shadow=False, color=carla.Color(), life_time=0)
        # print(f'--- road_id:{path_item[0].road_id}, path item: {path_item[0]}, type: {path_item[1]}')
    # 绘制拓扑图和routing信息
    if plot:
        global_route_plan.draw_routing_result(plt=plt)
        global_route_plan.draw_all_spawn_points(plt=plt, All_spawn_points=All_spawn_points)
    return routing_pathway, global_frenet_routing, road_ids

def get_obstacle_list(possible_static_obs, possible_dynamic_obs):
    static_obs_info = []
    dynamic_obs_info = []
    static_actor_list = []
    dynamic_actor_list = []
    time_stamp = time.time()*1e6 # 时间戳 单位为微秒
    for obs_v, dis in possible_static_obs:
        obs_loc = obs_v.get_transform().location
        heading = obs_v.get_transform().rotation.yaw * math.pi / 180.0
        vx = obs_v.get_velocity().x
        vy = obs_v.get_velocity().y
        width = 2 * obs_v.bounding_box.extent.x
        length = 2 * obs_v.bounding_box.extent.y
        static_obs_info.append((obs_loc.x, obs_loc.y, dis))
                # print("static_obs_id:", obs_v.type_id, "dis:", dis)
        static_actor_list.append((obs_v.id, obs_loc.x, obs_loc.y, vx, vy, heading, width, length, obs_v.type_id, time_stamp))
    
    for _obs_v, _dis, _speed in possible_dynamic_obs:
        obs_loc = _obs_v.get_transform().location
        heading = _obs_v.get_transform().rotation.yaw * math.pi / 180.0
        vx = _obs_v.get_velocity().x
        vy = _obs_v.get_velocity().y
        width = 2 * _obs_v.bounding_box.extent.x
        length = 2 * _obs_v.bounding_box.extent.y
        dynamic_obs_info.append((obs_loc.x, obs_loc.y, _dis, _speed))
                # print("dynamic_obs_id:", obs_v.type_id, "dis:", dis)
        dynamic_actor_list.append((_obs_v.id, obs_loc.x, obs_loc.y, heading, vx, vy, width, length, _obs_v.type_id, time_stamp))
    return static_obs_info, static_actor_list, dynamic_obs_info, dynamic_actor_list

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_path', type=str, default='/media/xph123/DATA/f_tmp/DTPP_datasets/nuplan-v1.1_test/data/cache/test')
    parser.add_argument('--map_path', type=str, default='/media/xph123/DATA/f_tmp/DTPP_datasets/nuplan-maps-v1.0/maps')
    parser.add_argument('--model_path', type=str, default='/home/xph123/fanyu/E2E/DTPP/base_model/base_model.pth')
    parser.add_argument('--test_type', type=str, default='closed_loop_reactive_agents')
    parser.add_argument('--load_test_set', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scenarios_per_type', type=int, default=1)
    parser.add_argument('--use_e2e', type=bool, default=False)
    args = parser.parse_args()
    
    global carla_scenario_input
    
        # initialize planner
    torch.set_grad_enabled(False)
    planner = DTPPPlanner(model_path=args.model_path, device=args.device)
    
        # create an animation window to observe our ego-vehicle's behaviour.
    pygame.init()  # init pygame
    pygame.font.init()  # init the font in pygame
    display = pygame.display.set_mode(size=(display_width, display_height))  # set the window size or resolution of pygame
    pygame.display.set_caption("CARLA DTPP Planning and Control")  # set the title of the window
    display.fill((1, 1, 1))
    pygame.display.flip()
    
    """
    1.创建一个子线程，并通过一个管道（Pipe）进行进程间通信；
    2.返回两个管道的连接对象，一个用于发送数据，一个永磊接受数据；
    3.p1 即为创建新的进程，target 指定子进程中要执行的函数，args 为一个元组，包含了传递给子进程的参数，conn2 即为之前管道的另一端；
    """
    
    # conn1, conn2 = multiprocessing.Pipe()
    # p1 = None
    # if args.use_e2e:
    #     """ 构建模型的输入 """
    #     p1 = multiprocessing.Process(target=motion_planning_e2e, args=(conn2,))
    # else:
    #     p1 = multiprocessing.Process(target=motion_planning_rule, args=(conn2,))
    # p1.start() # 开始线程

    client = carla.Client("localhost", 2000)
    client.set_timeout(20)
    # 对象创建好了之后，在对象中添加需要的环境中的地图
    world = client.load_world('Town05')  # type: carla.World
    debug = world.debug  # type: carla.DebugHelper
    amap = world.get_map()  # type: carla.Map
    All_spawn_points = amap.get_spawn_points()  # 获取所有carla提供的actor产生位置
    start_wp = All_spawn_points[189] # 259
    end_wp = All_spawn_points[259] # 48
    
    # 大型环线测试，从节点20到节点15，注意有可能生成actor不成功，可以结合 All_spawn_points 打印出来选择合适的位置
    # start_wp = carla.Transform(location=carla.Location(x=-142.31, y=147.54, z=0.06)) # node_id: 20
    # end_wp = carla.Transform(location=carla.Location(x=-142.29, y=144.04, z=0.06)) # node_id: 15
    
    # start_wp = amap.get_waypoint(location=start_loc, project_to_road=True, lane_type=(carla.LaneType.Driving|carla.LaneType.Sidewalk))
    # end_wp = amap.get_waypoint(location=end_loc, project_to_road=True, lane_type=(carla.LaneType.Driving|carla.LaneType.Sidewalk))
    
    
    # 方法2
    # start_wp = All_spawn_points[209] # node_id: 15
    # end_wp = All_spawn_points[210] # node_id: 20
    
    # 描绘出所有的actor产生位置

    routing_pathway, global_frenet_routing, road_ids = global_routing(amap = amap, start_tsf=start_wp, end_tsf=end_wp, debug=debug)    
    
    """# 定义一个ego-vehicle"""
    ego_spawn_point, ego_actor = set_ego_vehicle_on_routing(world, spawn_point=end_wp, routing=routing_pathway, along_route=False)

    """为车辆配备一个障碍物的传感器"""
    # obs_detector = Obstacle_detector(ego_vehicle=ego_actor, world=world)  # 实例化传感器
    # obs_detector.create_sensor()  # 在仿真环境中生成传感器
    
    """ 添加一个摄像头传感器 """
    image_data, camera = get_camera_image(display_width, display_height, world, ego_actor)        
    # 注册摄像头回调函数
    camera.listen(lambda image: camera_callback(image, image_data))

    """设置动态和静止车辆"""
    sobs_actor1, sobs_actor2, sobs_actor3 = generate_static_vehicles(world)
    agent, dobs_actor = generate_dynamic_vehicles(world, All_spawn_points, ego_spawn_point)
    
    print('Running simulations...')    

    # 3.提取局部路径
    ego_loc, match_point_list, local_frenet_refline_opt = get_local_refline(ego_actor, global_frenet_routing, back_length=10, forward_length=60)
    # 计算参考线的s, l
    cur_path_s, cur_path_l = get_frenet_sl(ego_loc, local_frenet_refline_opt)

    """整车参数设定"""
    vehicle_para = (1.015, 2.910 - 1.015, 1412, -148970, -82204, 1537)      # 车辆特性(侧偏刚度、转动惯量...)
    controller_name = "MPC_controller"
    Controller = Vehicle_control(ego_vehicle=ego_actor, vehicle_para=vehicle_para,
                                 pathway=local_frenet_refline_opt,
                                 controller_type=controller_name)  # 实例化控制器, "MPC_controller" or "LQR_controller"
    direction = []
    speed = []
    target_speed = []
    max_speed = 50  # 假设正常行驶速度是40km/h，市区允许的最大速度是50km/h, 前方的动态障碍物行驶的速度是30 km/h.
    # 一旦选择超车，自车可以将速度提升到50km/h
    # 设定一个观察者视角
    spectator = world.get_spectator()
    count = 1  # 控制规划器和控制器相对频率
    main_process_start_time = time.time()
    plan_count = 10
    # control_count = 10
    pred_ts = 0.1
    while True:
        print(f'-------- count: {count} -----------')
        """获取交通速度标志,考虑道路速度限制"""
        # 获取车辆位置信息（包括坐标和姿态信息）， get the transformation, a combination of location and rotation
        ego_transform = ego_actor.get_transform()
        # 不断更新观测视角的位置， update the position of spectator
        spectator.set_transform(carla.Transform(ego_transform.location + carla.Location(z=40), carla.Rotation(pitch=-90)))
        ego_loc = ego_transform.location  # 获取车辆的当前位置
        ego_rot = ego_transform.rotation  # 获取车辆的当前姿态
        ego_v = ego_actor.get_velocity()
        ego_a = ego_actor.get_acceleration()

        dobs_actor.apply_control(agent.run_step())
        """获取局部路径，局部路径规划的频率是控制的1/100"""
        if count % plan_count == 0:  # 这里表示控制器执行 plan_count 次规划器执行1次
            cur_time = time.time()
            # print("main_process_cost_time", time.time() - main_process_start_time)
            main_process_start_time = cur_time
            # mark = "replan" + str((round(ego_loc.x, 2), round(ego_loc.y, 2)))
            # world.debug.draw_string(carla.Location(ego_loc.x, ego_loc.y, 2), mark, draw_shadow=False,
            #                         color=carla.Color(r=0, g=0, b=255), life_time=1000,
            #                         persistent_lines=True)
            # debug.draw_point(carla.Location(ego_loc.x, ego_loc.y, 2),
            #                  size=0.05, color=carla.Color(0, 0, 0), life_time=0)



            # 基于笛卡尔坐标系预测ts秒过后车辆的位置，以预测点作为规划起点
            pred_x, pred_y, pred_fi = planning_utils.predict_block(ego_actor, ts=pred_ts)
            # 基于frenet坐标系预测ts秒过后车辆的位置，以预测点作为规划起点
            # pred_x, pred_y = planning_utils.predict_block_based_on_frenet(ego_loc, ego_v,
            #                                                               local_frenet_path_opt,
            #                                                               cur_path_s, cur_path_l, ts=0.2)

            # mark = "predict" + str((round(pred_x, 2), round(pred_y, 2)))
            # world.debug.draw_string(carla.Location(pred_x, pred_y, 2), mark, draw_shadow=False,
            #                         color=carla.Color(r=0, g=0, b=255), life_time=1000,
            #                         persistent_lines=True)
            # debug.draw_point(carla.Location(pred_x, pred_y, 2),
            #                  size=0.05, color=carla.Color(255, 255, 255), life_time=0)
            """
            没有找到合适的传感器，暂时用车联网的方法,设定合适的感知范围，获取周围环境中的actor，这里我们人为制造actor作为障碍物
            再到后面可以考虑用多传感器数据融合来做动态和静态障碍物的融合感知
            """
            possible_static_obs, possible_dynamic_obs = get_actor_from_world(ego_actor, dis_limitation=100)
            # 提取障碍物的位置信息
            static_obs_info, static_actor_list, dynamic_obs_info, dynamic_actor_list = get_obstacle_list(possible_static_obs, possible_dynamic_obs)
            print("static_obs_info:", static_obs_info)
            print("dynamic_obs_info:", dynamic_obs_info)
            print("static_actor_list:", static_actor_list)
            print("dynamic_actor_list:", dynamic_actor_list)
            # 将当前的道路状况和车辆信息发送给规划器进行规划控制
            if args.use_e2e:
                # conn1.send((static_actor_list, dynamic_actor_list, 
                #             (ego_loc.x, ego_loc.y, ego_rot.yaw * math.pi/180.0, ego_v.x, ego_v.y, ego_a.x, ego_a.y), 
                #             (pred_x, pred_y, pred_fi),
                #             local_frenet_path_opt,
                #             global_frenet_routing, match_point_list, road_ids))        
                trajectory = motion_planning_e2e(planner, static_actor_list, dynamic_actor_list, 
                            (ego_loc.x, ego_loc.y, ego_rot.yaw * math.pi/180.0, ego_v.x, ego_v.y, ego_a.x, ego_a.y), 
                            (pred_x, pred_y, pred_fi),
                            local_frenet_refline_opt,
                            global_frenet_routing, match_point_list, road_ids)        
            else:    
                # conn1.send((static_obs_info, dynamic_obs_info, 
                #             (ego_loc.x, ego_loc.y, ego_rot.yaw * math.pi/180.0, ego_v.x, ego_v.y, ego_a.x, ego_a.y), 
                #             (pred_x, pred_y, pred_fi),
                #             local_frenet_path_opt,
                #             global_frenet_routing, match_point_list, road_ids))
                cur_local_frenet_path_opt, match_point_list, cur_path_s, cur_path_l = motion_planning_rule(static_obs_info, dynamic_obs_info, 
                            (ego_loc.x, ego_loc.y, ego_rot.yaw * math.pi/180.0, ego_v.x, ego_v.y, ego_a.x, ego_a.y), 
                            (pred_x, pred_y, pred_fi),
                            global_frenet_routing, match_point_list)
            

            # if count != plan_count:  # 第一个循环周期，因为有初始阶段规划好的局部路径，第二个周期的规划还未计算完成，一旦执行接收数据，会阻塞主进程
            if True:
                # cur_local_frenet_path_opt, match_point_list, cur_path_s, cur_path_l = conn1.recv()  # 新规划出的轨迹
                if args.use_e2e:
                    cur_local_frenet_path_opt, match_point_list, cur_path_s, cur_path_l = get_from_trajectory(trajectory)
                """轨迹拼接
                思路比较简单，由于规划是在预测点进行的，对下个周期进行规划，因此当前周期的车辆运动结束点一定在预测点之前，
                找到上个规划周期轨迹中距离预测点最近的点，与新的规划路径进行拼接，保证轨迹的连续性,拼接完之后还需要进一步平滑
                """
                # min_DIS = 10000
                # for i in range(len(local_frenet_path_opt)):
                #     if (pred_x - local_frenet_path_opt[i][0]) ** 2 + (
                #             pred_y - local_frenet_path_opt[i][1]) ** 2 < min_DIS:
                #         min_DIS = (pred_x - local_frenet_path_opt[i][0]) ** 2 + (
                #                 pred_y - local_frenet_path_opt[i][1]) ** 2
                #     else:
                #         local_frenet_path_opt = local_frenet_path_opt[0:i] + cur_local_frenet_path_opt
                #         break
                local_frenet_path_opt = cur_local_frenet_path_opt
                for point in local_frenet_path_opt:
                    # print(waypoint)
                    debug.draw_point(carla.Location(point[0], point[1], 2), size=0.1, color=carla.Color(r=255), life_time=0.1)
            # 注意重新实例化控制器的位置，不能放错了
            Controller = Vehicle_control(ego_vehicle=ego_actor, vehicle_para=vehicle_para,
                                         pathway=local_frenet_path_opt,
                                         controller_type=controller_name)  # 依据当前局部路径实例化控制器

        """控制部分"""
        control = Controller.run_step(target_speed=max_speed)  # 实例化的时候已经将必要的信息传递给规划器，这里告知目标速度即可
        direction.append(ego_actor.get_transform().rotation.yaw * (math.pi / 180))
        V = ego_actor.get_velocity()  # 利用 carla API to 获取速度矢量， use the API of carla to get the velocity vector
        V_len = 3.6 * math.sqrt(V.x * V.x + V.y * V.y + V.z * V.z)  # transfer m/s to km/h
        speed.append(V_len)
        target_speed.append(max_speed)
        ego_actor.apply_control(control)  # 执行最终控制指令, execute the final control signal

        """debug 部分"""
        # # 将预测点和投影点的位置标出来, mark the predicted point and project point in the simulation world for debug
        # debug.draw_point(carla.Location(Controller.Lat_control.x_pre, Controller.Lat_control.y_pre, 2),
        #                  size=0.05, color=carla.Color(0, 0, 255), life_time=0)  # 预测点为蓝色
        # debug.draw_point(carla.Location(Controller.Lat_control.x_pro, Controller.Lat_control.y_pro, 2),
        #                  size=0.05, color=carla.Color(0, 255, 0), life_time=0)  # 投影点为绿色

        """距离判断，程序终止条件"""
        count += 1
        # 计算当前车辆和终点的距离, calculate the distance between vehicle and destination
        # dist = ego_loc.distance(routing_pathway[-1][0].transform.location)
        dist = ego_loc.distance(end_wp.location)
        print("The distance to the destination: ", dist)

        if dist < 0.2:  # 到达终点后产生制动信号让车辆停止运动
            control = emergence_brake()
            ego_actor.apply_control(control)
            print("last waypoint reached")
            # p1.terminate()
            break
        
        
        """ 获取摄像头数据 """
        if image_data['image'] is not None:
            # image_data['image'].save('./output/%d.png' % count)
            # 将摄像头数据转化为 Pygame 可以显示的格式
            image_surface = pygame.surfarray.make_surface(image_data['image'].swapaxes(0,1))
            # 在 Pygame 窗口中显示摄像头数据
            screen.blit(image_surface, (0, 0))
            pygame.display.flip()

    """可视化速度变化和航向变化"""
    # import matplotlib.pyplot as plt

    # plt.figure(1)
    # plt.plot(direction)
    # plt.ylim(bottom=-5, top=5)    
    # plt.figure(2)
    # plt.plot(speed)
    # plt.plot(target_speed, color="r")
    # plt.ylim(bottom=0, top=max(target_speed) + 10)
    # plt.show()
    print(f'-------- ending -----------')
