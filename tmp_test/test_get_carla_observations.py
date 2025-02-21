'''
Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
Author: 范雨
Date: 2025-02-26 19:45:38
LastEditTime: 2025-02-26 20:44:35
LastEditors: 范雨
Description: 
'''

import carla
import random
import time

# 连接到 Carla 服务端
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()
map = world.get_map()

# ================== 配置参数 ==================
VEHICLE_NUM = 20       # 车辆数量
PEDESTRIAN_NUM = 15    # 行人数量
BICYCLE_NUM = 3        # 自行车数量
MIN_SPAWN_DISTANCE = 3.0  # 生成点最小间距（米）

# ================== 清理现有对象 ==================
def destroy_all_actors():
    actors = world.get_actors()
    for actor in actors:
        if actor.type_id.startswith(('vehicle.', 'walker.', 'bicycle.')):
            actor.destroy()

destroy_all_actors()  # 清除之前残留的 Actor

# ================== 生成车辆 ==================
def spawn_vehicles():
    vehicle_blueprints = world.get_blueprint_library().filter('vehicle.*')
    spawn_points = world.get_map().get_spawn_points()
    
    spawned_vehicles = []
    attempts = 0
    
    while len(spawned_vehicles) < VEHICLE_NUM and attempts < 100:
        bp = random.choice(vehicle_blueprints)
        spawn_point = random.choice(spawn_points)
        
        # 检查生成点是否被占用
        collision = False
        for v in spawned_vehicles:
            if spawn_point.location.distance(v.get_location()) < MIN_SPAWN_DISTANCE:
                collision = True
                break
        if collision:
            attempts +=1
            continue
            
        try:
            vehicle = world.spawn_actor(bp, spawn_point)
            vehicle.set_autopilot(True)  # 启用自动驾驶
            spawned_vehicles.append(vehicle)
            print(f"生成车辆 {vehicle.type_id} 于 {spawn_point.location}")
        except:
            attempts +=1
            
    return spawned_vehicles

# ================== 生成行人 ==================
def spawn_pedestrians():
    pedestrian_blueprints = world.get_blueprint_library().filter('walker.pedestrian.*')
    
    spawned_peds = []
    controllers = []
    
    for _ in range(PEDESTRIAN_NUM):
        bp = random.choice(pedestrian_blueprints)
        
        # 随机生成可行走位置
        spawn_transform = carla.Transform()
        spawn_transform.location = world.get_random_location_from_navigation()
        while spawn_transform.location is None:
            spawn_transform.location = world.get_random_location_from_navigation()
            
        try:
            pedestrian = world.spawn_actor(bp, spawn_transform)
            controller_bp = world.get_blueprint_library().find('controller.ai.walker')
            controller = world.spawn_actor(controller_bp, carla.Transform(), pedestrian)
            
            # 设置随机行走目标
            controller.start()
            controller.go_to_location(world.get_random_location_from_navigation())
            controller.set_max_speed(1.5 + random.random())  # 1.5-2.5 m/s
            
            spawned_peds.append(pedestrian)
            controllers.append(controller)
            print(f"生成行人 {pedestrian.type_id}")
        except Exception as e:
            print(f"行人生成失败: {str(e)}")
            
    return spawned_peds, controllers

# ================== 生成自行车 ==================
def spawn_bicycles():
    bicycle_bp = world.get_blueprint_library().filter('vehicle.bh.crossbike')
    spawn_points = world.get_map().get_spawn_points()
    
    spawned_bicycles = []
    
    for _ in range(BICYCLE_NUM):
        bp = random.choice(bicycle_bp)
        spawn_point = random.choice(spawn_points)
        
        try:
            bicycle = world.spawn_actor(bp, spawn_point)
            bicycle.set_autopilot(True)
            spawned_bicycles.append(bicycle)
            print(f"生成自行车 {bicycle.type_id}")
        except:
            pass
            
    return spawned_bicycles

# ================== 主程序 ==================
if __name__ == '__main__':
    try:
        print("===== 开始生成车辆 =====")
        vehicles = spawn_vehicles()
        
        print("\n===== 开始生成行人 =====")
        pedestrians, ped_controllers = spawn_pedestrians()
        
        print("\n===== 开始生成自行车 =====")
        bicycles = spawn_bicycles()
                
        # 获取所有 Actor
        actors = world.get_actors()

        # 分类存储
        vehicles = []
        pedestrians = []
        bycicles = []
        others = []
        print("\n===== 统计生成的 Actor 类型 =====")
        for actor in actors:
            type_id = actor.type_id.lower()  # 统一转为小写避免大小写问题
            
            print(f"--- Actor Type: {type_id}")
            
            if type_id.startswith("vehicle."):
                if type_id.endswith("crossbike"): # vehicle.bh.crossbike
                    bycicles.append(actor)
                else:
                    vehicles.append(actor)
            elif type_id.startswith("walker."):
                pedestrians.append(actor)
            else:
                others.append(actor)

        # 输出统计结果
        print(f"车辆数量: {len(vehicles)}")
        print(f"行人数量: {len(pedestrians)}")
        print(f"自行车数量: {len(bycicles)}")  # 自行车的数量是多少？
        print(f"其他类型 Actor 数量: {len(others)}")

        
        # 保持程序运行
        print("\n生成完成，按 Ctrl+C 退出...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n正在清理场景...")
        destroy_all_actors()
        print("清理完成")

