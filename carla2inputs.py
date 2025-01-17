'''
Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
Author: 范雨
Date: 2025-01-10 12:24:03
LastEditTime: 2025-01-17 18:54:51
LastEditors: 范雨
Description: 
'''

import yaml

from common_utils import *

from typing import Deque

# from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
# from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
# from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
# from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
# from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
# from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario

from nuplan.planning.simulation.planner.abstract_planner import PlannerInput
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks, Observation, Sensors

import carla

from typing import Set
from nuplan_adapter.nuplan_data_process import *


class CarlaScenario:
    # possible_static_obs_, possible_dynamic_obs_, \
    #         vehicle_loc_, pred_loc_, vehicle_v_, vehicle_a_, global_frenet_path_, match_point_list_
    def __init__(self, possible_static_obs, 
                 possible_dynamic_obs, 
                 vehicle_loc, pred_loc, 
                 vehicle_v, vehicle_a, 
                 global_frenet_path, match_point_list):
        self.possible_static_obs = possible_static_obs
        self.possible_dynamic_obs = possible_dynamic_obs
        self.vehicle_loc = vehicle_loc
        self.pred_loc = pred_loc
        self.vehicle_v = vehicle_v
        self.vehicle_a = vehicle_a
        self.global_frenet_path = global_frenet_path
        self.match_point_list = match_point_list
    


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


# def creat_nuplan_scenario(args):
#     map_version = "nuplan-maps-v1.0"
#     scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
#     builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version, scenario_mapping=scenario_mapping)
#     if args.load_test_set:
#         params = yaml.safe_load(open('test_scenario.yaml', 'r'))
#         scenario_filter = ScenarioFilter(**params)
#     else:
#         scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type))
#     worker = SingleMachineParallelExecutor(use_process_pool=False)
#     return builder.get_scenarios(scenario_filter, worker)[0]



def create_planner_input_from_carla(actors):
    pass

def create_ego_state_buffer_from_carla(vehicle_loc: carla.Location,
                                       vehicle_v: float,
                                       vehicle_a: float,
                                       vehicle_pred_loc: carla.Location) -> Deque[EgoState]:
    """
    从CARLA场景中创建EgoStateBuffer
    """
    

def create_observation_buffer_from_carla() -> Deque[Observation]:
    pass

def trans_to_ego_state(vehicle_loc: carla.Location,
                        vehicle_v: float,
                        vehicle_a: float,
                        vehicle_pred_loc: carla.Location) -> EgoState:
    ego_state = EgoState()
    ego_state.rear_axle.x = vehicle_loc.x
    ego_state.rear_axle.y = vehicle_loc.y
    ego_state.rear_axle.z = vehicle_loc.z
    
    ego_state.rear_axle.leading = vehicle_loc.heading

def create_history_buffer_from_carla(carla_scenario: CarlaScenario) -> SimulationHistoryBuffer:
    ego_state_buffer = create_ego_state_buffer_from_carla(carla_scenario.vehicle_loc, 
                                                          carla_scenario.vehicle_v, 
                                                          carla_scenario.vehicle_a, 
                                                          carla_scenario.pred_loc)

def create_model_input_from_carla(carla_scenario: CarlaScenario) -> PlannerInput:
    pass





def create_feature_from_carla(history_buffer, map_api, route_roadblock_ids, device):
    num_agents = 20
    past_time_steps = 21
    map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK'] # name of map features to be extracted.
    max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} # maximum number of elements to extract per feature layer.
    max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} # maximum number of points per feature to extract per feature layer.
    radius = 80 # [m] query radius scope relative to the current pose.
    interpolation_method = 'linear'

    ego_state_buffer = history_buffer.ego_state_buffer # Past ego state including the current
    observation_buffer = history_buffer.observation_buffer # Past observations including the current

    ego_agent_past = sampled_past_ego_states_to_tensor(ego_state_buffer)
    past_tracked_objects_tensor_list, past_tracked_objects_types = sampled_tracked_objects_to_tensor_list(observation_buffer)
    time_stamps_past = sampled_past_timestamps_to_tensor([state.time_point for state in ego_state_buffer])
    ego_state = history_buffer.current_state[0]
    ego_coords = Point2D(ego_state.rear_axle.x, ego_state.rear_axle.y)
    coords, traffic_light_data = get_neighbor_vector_set_map(
        map_api, map_features, ego_coords, radius, route_roadblock_ids, traffic_light_data
    )

    ego_agent_past, neighbor_agents_past = agent_past_process(
        ego_agent_past, time_stamps_past, past_tracked_objects_tensor_list, past_tracked_objects_types, num_agents
    )

    vector_map = map_process(ego_state.rear_axle, coords, traffic_light_data, map_features, 
                             max_elements, max_points, interpolation_method)

    data = {"ego_agent_past": ego_agent_past[1:], 
            "neighbor_agents_past": neighbor_agents_past[:, 1:]}
    data.update(vector_map)
    data = convert_to_model_inputs(data, device)

    data = {"ego_agent_past": ego_agent_past[1:], 
         "neighbor_agents_past": neighbor_agents_past[:, 1:]}
    return data

    
    
    
    
                              