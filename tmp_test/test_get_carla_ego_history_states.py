import carla
import time
import csv
import math
from collections import deque

# 连接 Carla 服务端
client = carla.Client('localhost', 2000)
client.set_timeout(10.0)
world = client.get_world()

# 同步模式配置
settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = 0.1  # 固定步长 0.1 秒
world.apply_settings(settings)

# 创建车辆
vehicle = world.spawn_actor(
    world.get_blueprint_library().filter('model3')[0],
    world.get_map().get_spawn_points()[0]
)
vehicle.set_autopilot(True)

# CSV 文件配置
filename = f"sliding_window_{int(time.time())}.csv"
header = ['window_timestamp_us', 'x_seq', 'y_seq', 'heading_seq', 'vx_seq', 'vy_seq', 'ax_seq', 'ay_seq']

# 滑动窗口配置
WINDOW_SIZE = 22  # 2.1秒窗口（0.1秒 * 21）
data_buffer = deque(maxlen=WINDOW_SIZE)

def convert_to_vehicle_coords(world_vector, yaw_deg):
    """将世界坐标系向量转换到车辆坐标系"""
    yaw = math.radians(yaw_deg)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    x = world_vector.x * cos_yaw + world_vector.y * sin_yaw
    y = -world_vector.x * sin_yaw + world_vector.y * cos_yaw
    return x, y

try:
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        global last_timestamp_us
        last_timestamp_us = time.time()
        while True:
            start_time = time.time()
            world.tick()
            timestamp_us = time.time()
            timestamp_diff = timestamp_us - last_timestamp_us if last_timestamp_us else 0
            
            print(f"窗口填充完成，时间差：{timestamp_diff} s")
            last_timestamp_us = timestamp_us
            # 获取车辆状态
            transform = vehicle.get_transform()
            velocity = vehicle.get_velocity()
            acceleration = vehicle.get_acceleration()
            
            # 坐标系转换
            x, y = transform.location.x, transform.location.y
            heading = math.radians(transform.rotation.yaw)
            vx, vy = velocity.x, velocity.y
            ax, ay = acceleration.x, acceleration.y
            
            # 记录数据点
            data_buffer.append({
                'timestamp': timestamp_diff,
                'x': x,
                'y': y,
                'heading': heading,
                'vx': vx,
                'vy': vy,
                'ax': ax,
                'ay': ay
            })
            
            # 当窗口填满时写入CSV
            if len(data_buffer) == WINDOW_SIZE:
                # 提取序列数据（新数据在右侧）
                timestamp_us_seq = ";".join([str(d['timestamp']) for d in data_buffer])
                x_seq = ";".join([f"{d['x']:.3f}" for d in data_buffer])
                y_seq = ";".join([f"{d['y']:.3f}" for d in data_buffer])
                heading_seq = ";".join([f"{d['heading']:.2f}" for d in data_buffer])
                vx_seq = ";".join([f"{d['vx']:.3f}" for d in data_buffer])
                vy_seq = ";".join([f"{d['vy']:.3f}" for d in data_buffer])
                ax_seq = ";".join([f"{d['ax']:.3f}" for d in data_buffer])
                ay_seq = ";".join([f"{d['ay']:.3f}" for d in data_buffer])
                
                # 写入CSV
                writer.writerow([
                    timestamp_us_seq,
                    x_seq, y_seq, heading_seq,
                    vx_seq, vy_seq,
                    ax_seq, ay_seq
                ])
                print(f"记录窗口 @ {timestamp_us_seq}")

            # 计算剩余时间并休眠
            elapsed_time = time.time() - start_time
            sleep_time = max(0, 0.1 - elapsed_time)
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("用户终止记录")
finally:
    vehicle.destroy()
    settings.synchronous_mode = False
    world.apply_settings(settings)