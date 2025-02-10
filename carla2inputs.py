'''
Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
Author: 范雨
Date: 2025-01-10 12:24:03
LastEditTime: 2025-02-10 12:52:19
LastEditors: 范雨
Description: 
'''

# import yaml
import numpy as np
import math

from common_utils import *

from dataclasses import dataclass, field
from collections import deque
from typing import Deque, List, Dict, Tuple, Any

# from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
# from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
# from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
# from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
# from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
# from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

# from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
# from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
# from nuplan.common.actor_state.ego_state import EgoState
# from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation, Sensors

import carla

from typing import Set
# from nuplan_adapter.nuplan_data_process import *

@dataclass
class EgoState:
    """
    ego state(x,y,heading,vx,vy,ax,ay)
    """
    x : float = 0.0
    y : float = 0.0
    heading : float = 0.0
    vx : float = 0.0
    vy : float = 0.0
    ax : float = 0.0
    ay : float = 0.0
    timestamp : float = 0.0
    

    
@dataclass
class AgentState:
    """
    id,vx,vy,heading,width,length,x,y,obstype（3-d）
    """
    id : int = 0
    x : float = 0.0
    y : float = 0.0
    heading : float = 0.0
    vx : float = 0.0
    vy : float = 0.0    
    width : float = 0.0
    length : float = 0.0
    type : str = ""
    
    @staticmethod
    def set_agent_state_from_actor(actor: Any) -> 'AgentState':
        # print("--- actor.id:", actor[0])
        # return AgentState(id=actor.id, x=actor.get_location().x, y=actor.get_location().y, heading=actor.get_transform().rotation.yaw*np.pi/180,
        #                   vx=actor.get_velocity().x, vy=actor.get_velocity().y, 
        #                   width=2*actor.bounding_box.extent.x, length=2*actor.bounding_box.extent.y, type=actor.type_id)
        return AgentState(id=actor[0], x = actor[1], y = actor[2], heading = actor[3], vx = actor[4], vy = actor[5], width = actor[6], length = actor[7], type = actor[8])
@dataclass
class CarlaScenario:
    # possible_static_obs_, possible_dynamic_obs_, \
    #         vehicle_loc_, pred_loc_, vehicle_v_, vehicle_a_, global_frenet_path_, match_point_list_
    def __init__(self, possible_static_obs, 
                 possible_dynamic_obs, 
                 vehicle_loc, pred_loc, 
                 local_frenet_path_opt,
                 global_frenet_path, match_point_list, road_ids):
        # 获取当前系统时间戳，微秒
        self.timestamp = time.time()*1e6
        self._set_ego_state(vehicle_loc)
        self.ego_state.timestamp = self.timestamp
        self._set_agents_state(possible_static_obs, possible_dynamic_obs)
        self._set_map_lanes(local_frenet_path_opt)
        self._set_route_lines(global_frenet_path)
        self.road_ids = list(road_ids)

        
    def _set_ego_state(self, vehicle_loc):
        # 获取当前系统时间戳
        self.ego_state = EgoState(*vehicle_loc)
    def _set_agents_state(self, possible_static_obs, possible_dynamic_obs):
        """
          neighbor_agents_past:(track_tocken,vx,vy,heading,width,length,x,y,obstype（3-d）)
          只提取 3 种类型的障碍物：车辆，行人，自行车 （目前这里传入都是车辆）
        """
        self.agents = []
        agent = AgentState()
        for actor in possible_static_obs:
            agent.type = "vehicle"
            # print("--- sactor.id:", actor)
            agent = AgentState.set_agent_state_from_actor(actor)
            self.agents.append(agent)
            
        for actor in possible_dynamic_obs:
            agent.type = "vehicle"
            # print("--- dactor.id:", actor)
            agent = AgentState.set_agent_state_from_actor(actor)
            self.agents.append(agent)
    


    def _set_map_lanes(self, local_frenet_path: List[Tuple[float, float, float]]):
        """
        将 local_frenet_path 中的路径点转换为车道信息，并存储到 self.map_lanes 中。

        Args:
            local_frenet_path: 局部 Frenet 路径，每个点为 (x, y, heading)。
        """
        # 初始化 Deque，最大长度为 40
        T = Tuple[float, float, float, str]
        # if not hasattr(self, 'map_lanes'):
        self.map_lanes: Deque[List[T]] = deque(maxlen=40)

        # 提取最多 50 个点
        max_points_per_lane = 50
        current_lane = [
            (x, y, heading, "GREEN")  # 暂时先设置为 GREEN
            for x, y, heading, _ in local_frenet_path[:max_points_per_lane]
        ]

        # 如果 current_lane 点数小于 50，则使用最后一个点重复填充
        if len(current_lane) < max_points_per_lane:
            last_point = current_lane[-1] if current_lane else (0.0, 0.0, 0.0, "GREEN")
            current_lane.extend([last_point] * (max_points_per_lane - len(current_lane)))

        # 将 current_lane 添加到 map_lanes 中
        self.map_lanes.appendleft(current_lane)
            
    # def _set_map_lanes(self, local_frenet_path):
    #     # (x,y,heading,trafficlight_type(4-d))
    #     self.map_lanes = Deque(40)
    #     current_lane = []
    #     left_lane = []
    #     right_lane = []
    #     for i in range(len(local_frenet_path)):
    #         if i >= 50 :
    #             break
    #         lane_pts = (local_frenet_path[i][0], local_frenet_path[i][1], local_frenet_path[i][2], "GREEN") # 暂时先设置为 GREEN
    #         current_lane.append(lane_pts)
    #     # 如果 current_lane 点数小于50，则使用最后一个点重复填充
    #     if len(current_lane) < 50:
    #         last_point = current_lane[-1]
    #         while len(current_lane) < 50:
    #             current_lane.append(last_point)            
            
    #     self.map_lanes.appendleft(current_lane)
    
    def _set_route_lines(self, local_frenet_path: List[Tuple[float, float, float]]):
        """
        将 local_frenet_path 中的路径点转换为路线信息，并存储到 self.route_lines 中。

        Args:
            local_frenet_path: 局部 Frenet 路径，每个点为 (x, y, heading)。
        """
        # 初始化 Deque，最大长度为 10
        if not hasattr(self, 'route_lines'):
            self.route_lines: Deque[List[Tuple[float, float, float]]] = deque(maxlen=10)

        # 提取最多 50 个点
        max_points = 50
        current_lane = [
            (x, y, heading)  # 每个点的格式为 (x, y, heading)
            for x, y, heading, _ in local_frenet_path[:max_points]
        ]

        # 如果 current_lane 点数小于 50，则使用最后一个点重复填充
        if len(current_lane) < max_points:
            last_point = current_lane[-1] if current_lane else (0.0, 0.0, 0.0)
            current_lane.extend([last_point] * (max_points - len(current_lane)))

        # 将 current_lane 添加到 route_lines 中
        self.route_lines.appendleft(current_lane)
        
    # def _set_route_lines(self, local_frenet_path):
    #     # (x,y,heading)
    #     self.route_lines = Deque(10)
    #     current_lane = []

    #     for i in range(len(local_frenet_path)):
    #         if i > 50:
    #             break
    #         lane_pts = (local_frenet_path[i][0], local_frenet_path[i][1], local_frenet_path[i][2]) # 暂时先设置为 GREEN
    #         current_lane.append(lane_pts)
    #     # 如果 current_lane 点数小于50，则使用最后一个点重复填充
    #     if len(current_lane) < 50:
    #         last_point = current_lane[-1]
    #         while len(current_lane) < 50:
    #             current_lane.append(last_point)            
            
    #     self.route_lines.appendleft(current_lane)
        
        
        


# def get_lane_line_points(waypoint, distance):
#     left_points = []
#     right_points = []
#     current_waypoint = waypoint
#     while distance > 0:
#         left_waypoint = current_waypoint.get_left_lane_waypoint()
#         if left_waypoint:
#             left_points.append(left_waypoint.transform.location)
#             current_waypoint = left_waypoint
#             distance -= 1.0
#         else:
#             break
#     current_waypoint = waypoint
#     distance = distance
#     while distance > 0:
#         right_waypoint = current_waypoint.get_right_lane_waypoint()
#         if right_waypoint:
#             right_points.append(right_waypoint.transform.location)
#             current_waypoint = right_waypoint
#             distance -= 1.0
#         else:
#             break
#     return left_points, right_points


# def main():
#     try:
#         map = world.get_map()
#         location = carla.Location(x = 0, y = 0, z = 0)  # 替换为实际想要查询的位置
#         waypoint = map.get_waypoint(location)
#         left_points, right_points = get_lane_line_points(waypoint, 10)  # 获取10米距离内的车道线点
#         print("Left lane line points:")
#         for point in left_points:
#             print(f"x: {point.x}, y: {point.y}, z: {point.z}")
#         print("Right lane line points:")
#         for point in right_points:
#             print(f"x: {point.x}, y: {point.y}, z: {point.z}")
#     except Exception as e:
#         print(f"Error: {e}")

        
    


@dataclass
class CarlaScenarioInput:
    time_series: Deque = field(default_factory=lambda: deque(maxlen=22))
    iteration: int = 0
    traffic_light_data: List = field(default_factory=list)
    ego_states: Deque = field(default_factory=lambda: deque(maxlen=22))
    map_lanes: Deque = field(default_factory=lambda: deque(maxlen=40))
    route_lines: Deque = field(default_factory=lambda: deque(maxlen=10))
    agents_map: Dict[int, List[AgentState]] = field(default_factory=dict)
    road_ids: List = field(default_factory=list)
    is_ready : bool = False

def extract_carla_data(actors):
    vehicle_states = []
    for actor in actors.filter('vehicle.*'):
        state = {
            'id': actor.id,
            'location': actor.get_location(),
           'velocity': actor.get_velocity(),
            'orientation': actor.get_transform().rotation
        }
        vehicle_states.append(state)

    return vehicle_states




       






    
    
    
    
                              