'''
Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
Author: 范雨
Date: 2025-01-10 12:24:03
LastEditTime: 2025-01-14 10:11:42
LastEditors: fanyu fantiming@yeah.net
Description: 
'''

import yaml

from common_utils import *

from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario import NuPlanScenario



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


def creat_nuplan_scenario(args):
    map_version = "nuplan-maps-v1.0"
    scenario_mapping = ScenarioMapping(scenario_map=get_scenario_map(), subsample_ratio_override=0.5)
    builder = NuPlanScenarioBuilder(args.data_path, args.map_path, None, None, map_version, scenario_mapping=scenario_mapping)
    if args.load_test_set:
        params = yaml.safe_load(open('test_scenario.yaml', 'r'))
        scenario_filter = ScenarioFilter(**params)
    else:
        scenario_filter = ScenarioFilter(*get_filter_parameters(args.scenarios_per_type))
    worker = SingleMachineParallelExecutor(use_process_pool=False)
    return builder.get_scenarios(scenario_filter, worker)[0]



def create_scenario_from_carla():
    # from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioExtractionInfo
    # from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
    # scenario = NuPlanScenario(data_root=None, 
    #                           log_file_load_path=None, 
    #                           initial_lidar_token=None, 
    #                           initial_lidar_timestamp=None, 
    #                           scenario_type='from_carla', 
    #                           map_root=None,map_version=None, 
    #                           map_name='Town01', scenario_extraction_info=ScenarioExtractionInfo(), # or None
    #                           ego_vehicle_parameters=None,
    #                           cenario_extraction_info = ScenarioExtractionInfo(), # | None,
    #                           ego_vehicle_parameters = VehicleParameters(),
    #                           sensor_root=None)
    pass

def create_planner_input_from_carla(actors):
    pass

def create_model_input_from_carla(actors):
    
    
                              