import carla
import numpy as np
import math

from agents.navigation.behavior_agent import BehaviorAgent
from carla_planner import planning_utils


def carla_to_pygame(location, camera_transform, fov = 110, scale = 0.1, display_width=1920, display_height=1080):
    """
    将 CARLA 中的位置转换为 Pygame 中屏幕坐标；
    考虑将摄像头的视角和投影，使路径看起来像是画在地面上。
    """
    
    # 获取摄像头的位置和和旋转
    camera_location = camera_transform.location
    camera_roation = camera_transform.rotation
    
    # 计算相对位置
    dx = location.x - camera_location.x
    dy = location.y - camera_location.y
    dz = location.z - camera_location.z
    
    # 将相对位置转为摄像头坐标系
    yaw = np.radians(camera_roation.yaw)
    dx_cam = dx * np.cos(yaw) - dy * np.sin(yaw)
    dy_cam = dx * np.sin(yaw) + dy * np.cos(yaw)
    
    # 透视投影
    f = display_width / (2 * np.tan(np.radians(fov/2)))
    x = int(display_width / 2 + dx_cam * f / dy_cam * scale)
    y = int(display_height / 2 - dz * f / dy_cam * scale)
    
    return (x, y)

# 绘制路径
def draw_path(pygame, screen, route, camera_transform, color=(0, 255, 0), thickness=2):
    """在 Pygame 窗口中绘制路径"""
    if len(route) < 2:
        return
    points = [carla_to_pygame(waypoint[0].transform.location, camera_transform) for waypoint in route]
    pygame.draw.lines(screen, color, False, points, thickness)

# def get_traffic_light_state(current_traffic_light):  # get_state()方法只能返回红灯和黄灯状态，没有绿灯状态（默认把绿灯认为正常）
#     # 此方法把红绿蓝三种状态都标定出来
#     if current_traffic_light is None:
#         current_traffic_light_state = "Green"
#     else:
#         current_traffic_light_state = current_traffic_light.get_state()
#     return current_traffic_light_state


def emergence_brake():
    brake_control = carla.VehicleControl()
    brake_control.steer = 0  # 转向控制
    brake_control.throttle = 0  # 油门控制
    brake_control.brake = 1  # 刹车控制
    return brake_control


def get_actor_from_world(ego_vehicle: carla.Vehicle, dis_limitation=100):
    """已验证
    获取当前车辆前方潜在的车辆障碍物
    首先获取在主车辆一定范围内的其他车辆，再通过速度矢量和位置矢量将在主车辆运动方向后方的车辆过滤掉
    param:  ego_vehicle: 主车辆
            dis_limitation: 探测范围
    return: v_list:(vehicle, dist)
    """
    carla_world = ego_vehicle.get_world()  # type:carla.World
    vehicle_loc = ego_vehicle.get_location()
    static_vehicle_list = []  # 储存范围内的静态车辆
    dynamic_vehicle_list = []  # 储存范围内的动态车辆
    vehicle_list = carla_world.get_actors().filter("vehicle.*")
    for vehicle in vehicle_list:
        dis = math.sqrt((vehicle_loc.x - vehicle.get_location().x) ** 2 +
                        (vehicle_loc.y - vehicle.get_location().y) ** 2 +
                        (vehicle_loc.z - vehicle.get_location().z) ** 2)
        if dis < dis_limitation and ego_vehicle.id != vehicle.id:
            v1 = np.array([vehicle.get_location().x - vehicle_loc.x,
                           vehicle.get_location().y - vehicle_loc.y,
                           vehicle.get_location().z - vehicle_loc.z])  # 其他车辆到ego_vehicle的矢量
            ego_vehicle_velocity = np.array([ego_vehicle.get_velocity().x, ego_vehicle.get_velocity().y,
                                             ego_vehicle.get_velocity().z])  # ego_vehicle的速度矢量
            """# 如果车辆出现在ego_vehicle的运动前方，则有可能是障碍物
            # 还需要控制可能的障碍物距离参考线的横向距离, 我的想法是将障碍物在参考线上投影，计算投影点和车辆的距离，
            # 如果距离大于阈值则认为不影响ego-vehicle的运动，反之认为是障碍物会影响ego-vehicle的运动
            # 现在简化一下，将横向距离暂时设定为ego-vehicle当前航向方向的垂直距离，即ego完全按照参考线行驶"""
            ego_vehicle_theta = ego_vehicle.get_transform().rotation.yaw * (math.pi / 180)
            n_r = np.array([-math.sin(ego_vehicle_theta), math.cos(ego_vehicle_theta), 0])
            if -5 < np.dot(v1, n_r) < 5:  # v1在n_r方向上的投影
                if np.dot(v1, ego_vehicle_velocity) > -10:  # 在ego车后10m及车前的视为障碍物
                    vehicle_speed = math.sqrt(vehicle.get_velocity().x ** 2 +
                                              vehicle.get_velocity().y ** 2 + vehicle.get_velocity().z ** 2)
                    if vehicle_speed > 1:
                        dynamic_vehicle_list.append((vehicle, dis, vehicle_speed))
                    else:
                        static_vehicle_list.append((vehicle, dis))
                # elif np.dot(v1, ego_vehicle_velocity) < 0 and dis < 10:  # 自车后面十米以内的障碍物仍然考虑，超过十米就不再考虑
                #     static_vehicle_list.append((vehicle, -dis))
    static_vehicle_list.sort(key=lambda tup: tup[1])  # 按距离排序
    dynamic_vehicle_list.sort(key=lambda tup: tup[1])  # 按距离排序
    return static_vehicle_list, dynamic_vehicle_list







def set_ego_vehicle_on_routing(world: carla.World, spawn_point: carla.Transform = None, routing = None, along_route = False):
    '''
    设置ego vehicle在routing路径上, CARLA 提供了 try_spawn_actor 方法，它会尝试在指定位置生成 Actor，如果位置不安全（有碰撞），则返回 None。你可以通过循环尝试多个位置，直到成功生成。
    :param world: carla.World
    :param routing: 路径点列表, 元素为(carla.Waypoint, edge['type'])
    :return: spawn_point, model3_actor
    '''
    model3_bp = world.get_blueprint_library().find('vehicle.tesla.model3')
    model3_bp.set_attribute('color', '255,88,0')
    if along_route and len(routing) != 0:
        # print(f'---wplist: {routing}')
        for rt in routing:
            spawn_point = rt[0].transform
            model3_actor = world.try_spawn_actor(model3_bp, spawn_point)  # type: carla.Vehicle
            if model3_actor:
                print('--- 312 --- spawn_actor success, generate actor along the route!')
                break
    else:
         model3_actor = world.try_spawn_actor(model3_bp, spawn_point)  # type: carla.Vehicle
         print(f'--- 326 --- spawn_actor at {spawn_point} successfully!')
    if not model3_actor:
        print('--- 312 --- spawn_actor failed!')
        raise RuntimeError('spawn_actor failed!')
    # 定义轮胎特性
    # wheel_f = carla.WheelPhysicsControl()  # type: carla.WheelPhysicsControl
    # 定义车辆特性
    physics_control = carla.VehiclePhysicsControl()  # type: carla.VehiclePhysicsControl
    physics_control.mass = 1412  # 质量kg
    model3_actor.apply_physics_control(physics_control)
    return spawn_point, model3_actor

def generate_static_vehicles(world: carla.World):
    # 静止车辆1
    obs_vehicle_bp1 = world.get_blueprint_library().find('vehicle.tesla.model3')
    obs_vehicle_bp1.set_attribute('color', '0,0,255')
    obs_spawn_point1 = carla.Transform()
    obs_spawn_point1.location = carla.Location(x=174.01, y=147.61, z=0.3)
    obs_spawn_point1.rotation = carla.Rotation(yaw=91)
    obs_actor1 = world.spawn_actor(obs_vehicle_bp1, obs_spawn_point1)  # type: carla.Vehicle

    # 静止车辆2
    obs_vehicle_bp2 = world.get_blueprint_library().find('vehicle.audi.tt')
    obs_vehicle_bp2.set_attribute('color', '0,255,0')
    obs_spawn_point2 = carla.Transform()
    obs_spawn_point2.location = carla.Location(x=105, y=190, z=0.3)
    obs_spawn_point2.rotation = carla.Rotation(yaw=180)
    obs_actor2 = world.spawn_actor(obs_vehicle_bp2, obs_spawn_point2)  # type: carla.Vehicle

    # 静止车辆3
    obs_vehicle_bp3 = world.get_blueprint_library().find('vehicle.audi.tt')
    obs_vehicle_bp3.set_attribute('color', '0,255,0')
    obs_spawn_point3 = carla.Transform()
    obs_spawn_point3.location = carla.Location(x=105, y=193, z=0.3)
    obs_spawn_point3.rotation = carla.Rotation(yaw=180)
    obs_actor3 = world.spawn_actor(obs_vehicle_bp3, obs_spawn_point3)  # type: carla.Vehicle
    return obs_actor1, obs_actor2, obs_actor3

def generate_dynamic_vehicles(world, All_spawn_points, model3_spawn_point):
    obs_dy_vehicle_bp1 = world.get_blueprint_library().find('vehicle.tesla.model3')
    obs_dy_vehicle_bp1.set_attribute('color', '0,0,255')
    obs_dy_spawn_point1 = carla.Transform()
    obs_dy_spawn_point1.location = carla.Location(x=192.31, y=10, z=0.3)
    obs_dy_spawn_point1.rotation = model3_spawn_point.rotation
    dobs_actor = world.spawn_actor(obs_dy_vehicle_bp1, obs_dy_spawn_point1)  # type: carla.Vehicle
    agent = BehaviorAgent(dobs_actor, "normal")
    destination = All_spawn_points[48].location
    agent.set_destination(destination)
    agent.set_target_speed(30.0)
    return agent, dobs_actor

def get_local_refline(ego_actor, global_frenet_routing, back_length=10, forward_length=60):
    transform = ego_actor.get_transform()
    ego_loc = transform.location  # 获取车辆的当前位置
    match_point_list, _ = planning_utils.find_match_points(xy_list=[(ego_loc.x, ego_loc.y)],
                                                        frenet_path_node_list=global_frenet_routing,
                                                        is_first_run=True,  # 寻找车辆起点的匹配点就属于第一次运行，
                                                        pre_match_index=0)  # 没有上一次运行得到的索引，索引自然是全局路径的起点
    local_frenet_refline = planning_utils.sampling(match_point_list[0], global_frenet_routing, back_length=back_length, forward_length=forward_length)
    local_frenet_refline_opt = planning_utils.smooth_reference_line(local_frenet_refline)
    return ego_loc,match_point_list,local_frenet_refline_opt

def get_frenet_sl(vehicle_loc, local_frenet_path_opt):
    cur_s_map = planning_utils.cal_s_map_fun(local_frenet_path_opt, origin_xy=(vehicle_loc.x, vehicle_loc.y))
    cur_path_s, cur_path_l = planning_utils.cal_s_l_fun(local_frenet_path_opt, local_frenet_path_opt, cur_s_map)
    return cur_path_s, cur_path_l


def camera_callback(image, data_dict):
    '''
    定义一个回调函数来处理摄像头数据
    '''
    data_dict['image'] = np.frombuffer(image.raw_data, dtype=np.uint8)
    data_dict['image'] = np.reshape(data_dict['image'], newshape=(image.height, image.width, 4))
    data_dict['image'] = data_dict['image'][:, :, :3] # 去掉 Alpha 通道
    
def get_camera_image(display_width, display_height, world, ego_actor):
    camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', str(display_width))
    camera_bp.set_attribute('image_size_y', str(display_height))
    camera_bp.set_attribute('fov', '110') # 视野范围
    
    # 设置摄像头的位置（车辆前方）
    view_dir = {"default_tsf":carla.Transform(carla.Location(x=-8, z=5)),
                "topdown": carla.Transform(carla.Location(x=0, z=35), carla.Rotation(pitch=-90, yaw = -90))}
    camera_transform = view_dir["topdown"]
    # camera_transform = carla.Transform(carla.Location(x=-8, z=5))
    
    #生成摄像头
    camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_actor)
    print(f'Camera spawned at {camera_transform}')
    
    # 创建一个字典来存储摄像头数据
    image_data = {'image': None}

    return image_data, camera


def get_from_trajectory(trajectory):
    local_frenet_path_opt = []
    match_point_list = []
    cur_path_s = []
    cur_path_l = []
    # TODO(fanyu):
    # for i in range(len(trajectory)):
    #     local_frenet_path_opt.append((trajectory[i][0], trajectory[i][1]))
    #     match_point_list.append(trajectory[i][2])
    #     cur_path_s.append(trajectory[i][3])
    #     cur_path_l.append(trajectory[i][4])
    return local_frenet_path_opt, match_point_list, cur_path_s, cur_path_l