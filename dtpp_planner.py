'''
Author: fanyu fantiming@yeah.net
Date: 2024-07-17 16:39:38
LastEditors: 范雨
LastEditTime: 2025-02-14 16:07:51
FilePath: \DTPP_fy\planner.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import math
import time
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from planner_utils import *
# from obs_adapter import *
from trajectory_tree_planner import TreePlanner
from scenario_tree_prediction import *

from collections import deque
from typing import Deque, List, Optional, Tuple, Type, Union, Iterable

from dataclasses import dataclass

from carla2inputs import *
from nuplan_adapter.nuplan_data_process import *

  


class DTPPPlanner():
    def __init__(self, model_path, device):
        self._future_horizon = T # [s] 
        self._step_interval = DT # [s]
        self._N_points = int(T/DT)
        self._model_path = model_path
        self._device = device
        self.initialize()

    def name(self) -> str:
        return "DTPP Planner"
    
    # def observation_type(self):
    #     return DetectionsTracks

    def initialize(self):
        # self._map_api = initialization.map_api
        # self._goal = initialization.mission_goal
        # self._route_roadblock_ids = initialization.route_roadblock_ids
        self._route_roadblock_ids = []
        self._initialize_route_plan(self._route_roadblock_ids)
        self._initialize_model()
        self._trajectory_planner = TreePlanner(self._device, self._encoder, self._decoder)

    def _initialize_model(self):
        model = torch.load(self._model_path, map_location=self._device)
        self._encoder = Encoder()
        self._encoder.load_state_dict(model['encoder'])
        self._encoder.to(self._device)
        self._encoder.eval()
        self._decoder = Decoder()
        self._decoder.load_state_dict(model['decoder'])
        self._decoder.to(self._device)
        self._decoder.eval()

    def _initialize_route_plan(self, route_roadblock_ids):
        self._route_roadblocks = []

        # for id_ in route_roadblock_ids:
        #     block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
        #     block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)
        #     self._route_roadblocks.append(block)

        # self._candidate_lane_edge_ids = [
        #     edge.id for block in self._route_roadblocks if block for edge in block.interior_edges
        # ]
        self._route_roadblocks = []
        self._candidate_lane_edge_ids = []

    def compute_planner_trajectory(self, carla_scenario_input: CarlaScenarioInput):
        # Extract iteration, history, and traffic light
        iteration = carla_scenario_input.iteration
        # history = carla_scenario_input.history
        traffic_light_data = list(carla_scenario_input.traffic_light_data)
        ego_state, observation = carla_scenario_input.ego_states[-1], carla_scenario_input.agents_map

        # Construct input features
        start_time = time.perf_counter()
        # features = observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)
        features = create_feature_from_carla(carla_scenario_input=carla_scenario_input, device=self._device)

        # Get starting block
        starting_block = None
        cur_point = (ego_state.x, ego_state.y)
        closest_distance = math.inf

        # for block in self._route_roadblocks:
        #     for edge in block.interior_edges:
        #         distance = edge.polygon.distance(Point(cur_point))
        #         if distance < closest_distance:
        #             starting_block = block
        #             closest_distance = distance

        #     if np.isclose(closest_distance, 0):
        #         break
        
        # Get traffic light lanes
        traffic_light_lanes = []
        for data in traffic_light_data:
            id_ = str(data.lane_connector_id)
            if data.status == TrafficLightStatusType.RED and id_ in self._candidate_lane_edge_ids:
                # lane_conn = self._map_api.get_map_object(id_, SemanticMapLayer.LANE_CONNECTOR)
                # traffic_light_lanes.append(lane_conn)
                print(f"Todo(fanyu): add traffic light lane connector {id_} to candidate lane edge ids")

        # Tree policy planner
        # try:
        for k, v in features.items():
            print(f'---116---features {k}: {v}')    
        plan = self._trajectory_planner.plan(iteration, ego_state, features, carla_scenario_input, self._route_roadblocks, 
                                             self._candidate_lane_edge_ids, traffic_light_lanes, observation)
        # except Exception as e:
        #     print("Error in planning")
        #     print(e)
        #     plan = np.zeros((self._N_points, 3))
            
        # Convert relative poses to absolute states and wrap in a trajectory object
        states = transform_predictions_to_states(plan, carla_scenario_input.ego_states, self._future_horizon, DT)
        # trajectory = InterpolatedTrajectory(states)
        trajectory = states
        print(f'Step {iteration+1} Planning time: {time.perf_counter() - start_time:.3f} s')
        for state in states:
            print(state)

        return trajectory
