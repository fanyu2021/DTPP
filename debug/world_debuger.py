

import carla
import numpy as np
import random
import math
import json
from nuplan.common.actor_state.state_representation import Point2D

from debug.debug_utils import transform_to_global_frame


colors = {"blue":carla.Color(b=255), "green":carla.Color(g=255),\
           "red":carla.Color(r=255), "orange":carla.Color(r=255,g=255), 
           "purple":carla.Color(r=255,b=255), "brown":carla.Color(r=125,g=125,b=10),
              "yellow":carla.Color(r=255,g=255),
                "white":carla.Color(r=255,g=255,b=255), "black":carla.Color(r=0,g=0,b=0),
                  "gray":carla.Color(r=125,g=125,b=125), "pink":carla.Color(r=255,g=125,b=125),
                    "cyan":carla.Color(r=125,g=255,b=125), "magenta":carla.Color(r=255,g=125,b=255),
                      "lime":carla.Color(r=125,g=255,b=0), "maroon":carla.Color(r=125,g=0,b=0),
                        "olive":carla.Color(r=125,g=125,b=0), "teal":carla.Color(r=0,g=125,b=125),
                          "navy":carla.Color(r=0,g=0,b=125), "silver":carla.Color(r=192,g=192,b=192),
                            "gold":carla.Color(r=255,g=215,b=0), "beige":carla.Color(r=245,g=245,b=220),
                              "turquoise":carla.Color(r=64,g=224,b=208), "lavender":carla.Color(r=230,g=230,b=250),
                                "coral":carla.Color(r=255,g=127,b=80), "aliceblue":carla.Color(r=240,g=248,b=255),
                                  "antiquewhite":carla.Color(r=250,g=235,b=215), "aquamarine":carla.Color(r=127,g=255,b=212),
                                    "azure":carla.Color(r=240,g=255,b=255), "bisque":carla.Color(r=255,g=228,b=196),
                                      "blanchedalmond":carla.Color(r=255,g=235,b=205), "blueviolet":carla.Color(r=138,g=43,b=226),
                                        "burlywood":carla.Color(r=222,g=184,b=135), "cadetblue":carla.Color(r=95,g=158,b=160),
                                          "chartreuse":carla.Color(r=127,g=255,b=0), "chocolate":carla.Color(r=210,g=105,b=30),
                                            "cornflowerblue":carla.Color(r=100,g=149,b=237), "cornsilk":carla.Color(r=255,g=248,b=220),
                                              "crimson":carla.Color(r=220,g=20,b=60), "darkblue":carla.Color(r=0,g=0,b=139),
                                                "darkcyan":carla.Color(r=0,g=139,b=139), "darkgoldenrod":carla.Color(r=184,g=134,b=11),
                                                  "darkgray":carla.Color(r=169,g=169,b=169), "darkgreen":carla.Color(r=0,g=100,b=0),
                                                    "darkkhaki":carla.Color(r=189,g=183,b=107), "darkmagenta":carla.Color(r=139,g=0,b=139),
                                                      "darkolivegreen":carla.Color(r=85,g=107,b=47), "darkorange":carla.Color(r=255,g=140,b=0),
                                                        "darkorchid":carla.Color(r=153,g=50,b=204), "darkred":carla.Color(r=139,g=0,b=0),
                                                          "darksalmon":carla.Color(r=233,g=150,b=122), "darkseagreen":carla.Color(r=143,g=188,b=143),
                                                            "darkslateblue":carla.Color(r=72,g=61,b=139), "darkslategray":carla.Color(r=47,g=79,b=79),
                                                              "darkturquoise":carla.Color(r=0,g=206,b=209), "darkviolet":carla.Color(r=148,g=0,b=211),
                                                                "deeppink":carla.Color(r=255,g=20,b=147), "deepskyblue":carla.Color(r=0,g=191,b=255),
                                                                  "dimgray":carla.Color(r=105,g=105,b=105), "dodgerblue":carla.Color(r=30,g=144,b=255),
                                                                    "firebrick":carla.Color(r=178,g=34,b=34), "floralwhite":carla.Color(r=255,g=250,b=240),
                                                                      "forestgreen":carla.Color(r=34,g=139,b=34), "gainsboro":carla.Color(r=220,g=220,b=220),
           }
# color_keys = ["blue", "green", "red", "orange", "purple", "brown"]
color_keys = list(colors.keys())

class SingletonMeta(type):
    _instances = {}  # 类属性，存储所有单例类的实例
    
    def __call__(cls, *args, **kwargs):
        # 当尝试通过ClassName()创建实例时触发
        if cls not in cls._instances:  # 检查是否已有该类的实例
            # 创建新实例并存储在字典中（仅首次创建）
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]  # 始终返回已存在的实例


class WorldDebuger(metaclass=SingletonMeta):
    # 这个声明会使WorldDebuger的实例创建受SingletonMeta控制
    # 无论调用多少次WorldDebuger()，返回的都是同一个实例
    def __init__(self, actor: carla.Actor):
        self._world = actor.get_world()
        self._actor = actor
        self._debug = self._world.debug
        self._time_list = []



    def draw_routing(self, routing):
        # 绘制路由点
        route_x = [wp_road_opt[0].transform.location.x for wp_road_opt in routing]
        route_y = [wp_road_opt[0].transform.location.y for wp_road_opt in routing]
        # self._p.scatter(route_x, route_y, size=5, color="yellow", alpha=0.5)
        z = 1.2
        for wp, road_opt in routing:
            loc = carla.Location(x = wp.transform.location.x, y = wp.transform.location.y, z = 0.1)
            # begin = loc + carla.Location(z=z)
            self._debug.draw_point(loc, size=0.1, color=colors['darksalmon'], life_time=0.1, type=carla.Location)
            # self._debug.draw_string(loc, text="o", color=colors['green'], life_time=0.1, persistent_lines=True)

        

    def plot_candiate_lanes(self, dtpp_map):
        # self.draw_routing(dtpp_map._routing)
        trim_lanes = dtpp_map.get_candidate_lanes(self._actor)
        # 遍历 trimmed_paths 并绘制每条路径
        for idx, path in enumerate(trim_lanes):
            color = colors[color_keys[idx % len(color_keys)]]
            # self._p.scatter(x, y, size=5, color=color, alpha=0.5)
            z = 0.1
            for e in path[2]:
                loc = carla.Location(x = e[0], y = e[1], z = z)
                begin = loc + carla.Location(z=z)
                # angle = math.radians(actor.get_transform().roation.yaw)
                # end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
                self._debug.draw_point(begin, size=0.05, color=color, life_time=1.0)
            


    def plot_generated_paths(self, g_paths, actor=None):
        # print(f"g_path.shape:{g_paths}")
        actor = actor or self._actor
        xs, ys = [], []
        for i, path in enumerate(g_paths):
            global_path = transform_to_global_frame(path, actor)
            x_list = [pt[0] for pt in global_path]
            y_list = [pt[1] for pt in global_path]
            xs.append(x_list)
            ys.append(y_list)

            color = colors[color_keys[i % len(color_keys)]]
            
            z = 0.12
            for e in global_path:
                loc = carla.Location(x = e[0], y = e[1], z = z)
                # begin = loc + carla.Location(z=z)
                self._debug.draw_point(loc, size=0.05, color=color, life_time=0.1)

    @staticmethod
    def draw_trajectory(world, trajectory):
        z = 0.5
        color = colors['red']
        for pt in trajectory:
            transform = carla.Transform()
            transform.location = carla.Location(x=pt.rear_axle.x,y=pt.rear_axle.y, z=0)
            transform.rotation = carla.Rotation(yaw = np.rad2deg(pt.rear_axle.heading))
            # logger.debug(f"pt:{pt.rear_axle.x},{pt.rear_axle.y},{np.rad2deg(pt.rear_axle.heading)}")
            loc = carla.Location(x=pt.rear_axle.x,y=pt.rear_axle.y, z=z)
            world.debug.draw_point(loc, size=0.25, color=color, life_time=0.1)

    @staticmethod
    def draw_states(world, states):
        z = 0.1
        # color = colors['orange']
        color = carla.Color(r=0,g=255,b=250)
        for pt in states:
            transform = carla.Transform()
            transform.location = carla.Location(x=pt.rear_axle.x,y=pt.rear_axle.y, z=0)
            transform.rotation = carla.Rotation(yaw = np.rad2deg(pt.rear_axle.heading))
            # logger.debug(f"pt:{pt.rear_axle.x},{pt.rear_axle.y},{np.rad2deg(pt.rear_axle.heading)}")
            loc = carla.Location(x=pt.rear_axle.x,y=pt.rear_axle.y, z=z)
            world.debug.draw_point(loc, size=0.25, color=color, life_time=0.1)




    def record_times(self, time_dict):
        print(f'\n first_rule_time:{int(time_dict["r1"])}ms,\
            first_predict_time:{int(time_dict["p1"])}ms,\
            second_rule_time:{int(time_dict["r2"])}ms,\
            second_predict_time:{int(time_dict["p2"])}ms,\
            cost_selected_time:{int(time_dict["cs"])}ms')
        self._time_list.append(time_dict)


    def write_times_json(self):      
        json_str = json.dumps(self._time_list)
        byte_data = json_str.encode()

        # 打开文件，使用 'w' 模式表示写入        
        # 使用 write 方法写入内容
        with open('debug/time_list.bin', 'wb') as file:
            # 将字典转换为 JSON 字符串
            file.write(byte_data)
