

import carla
import numpy as np
import random
import math
from nuplan.common.actor_state.state_representation import Point2D

from debug.debug_utils import transform_to_global_frame


colors = {"blue":carla.Color(b=255), "green":carla.Color(g=255),\
           "red":carla.Color(r=255), "orange":carla.Color(r=255,g=255), 
           "purple":carla.Color(r=255,b=255), "brown":carla.Color(r=125,g=125,b=10)}
color_keys = ["blue", "green", "red", "orange", "purple", "brown"]

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class WorldDebuger(metaclass=SingletonMeta):
    def __init__(self, actor: carla.Actor):
        self._world = actor.get_world()
        self._actor = actor
        self._debug = self._world.debug


    # def draw_topology(self, topology):
    #     for lane in topology:
    #         lane_line = [
    #             Point2D(wp.transform.location.x, wp.transform.location.y)
    #             for wp in lane["path"]
    #         ]
    #         color = "red" if lane["entry"].is_junction else "blue"
    #         line_type = "dashed" if lane["entry"].lane_id < 0 else "solid"
    #         plt.plot(
    #             [pt.x for pt in lane_line],
    #             [pt.y for pt in lane_line],
    #             color=color,
    #             linestyle=line_type,
    #         )
    #         last_theta = lane["entry"].transform.rotation.yaw
    #         l_arrow = 0.01
    #         dir = (
    #             np.array(
    #                 [np.cos(np.deg2rad(last_theta)), np.sin(np.deg2rad(last_theta))]
    #             )
    #             * l_arrow
    #         )

    #         plt.arrow(
    #             lane["entry"].transform.location.x,
    #             lane["entry"].transform.location.y,
    #             dir[0],
    #             dir[1],
    #             head_width=1,
    #             head_length=5,
    #             fc=color,
    #             ec=color,
    #         )
    #         plt.text(
    #             lane["entry"].transform.location.x + 0.5,
    #             lane["entry"].transform.location.y + 0.5,
    #             "s%d.r%d\nl%d.j%d"
    #             % (
    #                 lane["entry"].section_id,
    #                 lane["entry"].road_id,
    #                 lane["entry"].lane_id,
    #                 lane["entry"].junction_id,
    #             ),
    #             fontdict={"fontsize": 14, "color": color, "fontweight": "bold"},
    #         )

    def draw_routing(self, routing):
        pass

    def plot_candiate_lanes(self, dtpp_map):
        trim_lanes = dtpp_map.get_candidate_lanes(self._actor)
        # 遍历 trimmed_paths 并绘制每条路径
        for idx, path in enumerate(trim_lanes):
            x = path[2][:, 0].tolist()
            y = path[2][:, 1].tolist()
            color = colors[color_keys[idx % len(color_keys)]]
            # self._p.scatter(x, y, size=5, color=color, alpha=0.5)
            z = 0.1
            for e in path[2]:
                loc = carla.Location(x = e[0], y = e[1], z = z)
                begin = loc + carla.Location(z=z)
                # angle = math.radians(actor.get_transform().roation.yaw)
                # end = begin + carla.Location(x=math.cos(angle), y=math.sin(angle))
                self._debug.draw_point(begin, size=0.05, color=color, life_time=1.0)
            


    def plot_generated_paths(self, g_paths, actor):
        # print(f"g_path.shape:{g_paths}")
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