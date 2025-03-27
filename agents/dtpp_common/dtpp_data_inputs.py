
from typing import Deque, List, Optional, Tuple, Dict, Union
from collections import deque

import time


import carla

from nuplan.common.actor_state.state_representation import Point2D
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatuses,
    TrafficLightStatusType,
)
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import *

from agents.dtpp_common.dtpp_map_utils import (

    get_neighbor_vector_set_map,
    )
from agents.dtpp_common.dtpp_map import DtppMap
from agents.dtpp_common.features_adapter import (
    DtppDataConfig,
    get_tracked_objects_from_actors,
    get_ego_state_list_from_actor,
    get_tracked_actors,
    sampled_tracked_agents_to_tensor_list,
    get_traffic_light_data,
    agent_past_process,
    map_process,
    convert_to_model_inputs)

from custom_format import *
logger = create_colored_logger(name=__name__)



class DtppInputs(object):
    def __init__(self, vehicle: carla.Actor) -> None:
        self._conf = DtppDataConfig()
        self._ego_state_buffer = deque(maxlen=DtppDataConfig.window_size) # EgoState
        self._observation_buffer = deque(maxlen=DtppDataConfig.window_size) # TrackedObjects
        # self.dtpp_map = DtppMap(map, topology, routing)
        self._vehicle = vehicle
        self._is_ready = False
        self._traffic_light_data: List[TrafficLightStatusData] = []
        self._carla_traffice_lights: List[carla.Actor] = []

    def _update_ego_state_buffer(self, timestamp_us: int, ego: carla.Actor) -> Deque[EgoState]:
        if not self._is_ready:
            logger.error("not ready")
        else:
            logger.error('ready!!!')

        ego_state = get_ego_state_list_from_actor(timestamp_us=timestamp_us, ego=ego)
        self._ego_state_buffer.append(ego_state)
        

    def _update_observation_buffer(self, tracked_actors: List[carla.Actor]) -> Deque[TrackedObjects]:
        tracked_objects = get_tracked_objects_from_actors(tracked_actors)
        observation = DetectionsTracks(tracked_objects=tracked_objects)
        self._observation_buffer.append(observation)

    def update(self, dtpp_map, device='cuda'):
        timestamp_us = time.time() * 1e6
        tracked_actors = get_tracked_actors(world=self._vehicle.get_world())
        self._update_ego_state_buffer(timestamp_us=timestamp_us, ego=self._vehicle)        
        self._update_observation_buffer(tracked_actors=tracked_actors)
        ego_agent_past = sampled_past_ego_states_to_tensor(self._ego_state_buffer)
        past_tracked_objects_tensor_list, past_tracked_objects_types = sampled_tracked_agents_to_tensor_list(self._observation_buffer)
        time_stamps_past = sampled_past_timestamps_to_tensor([state.time_point for state in self._ego_state_buffer])
        # logger.debug(f"--- ego_time_diff: {np.diff(np.array([state.time_point.time_s for state in self._ego_state_buffer]))}")
        # logger.debug(f"--- time_stamps_past: {[state.time_point.time_s for state in self._ego_state_buffer]}")
        self._ego_state = self._ego_state_buffer[-1]
        self._observation = self._observation_buffer[-1]
        ego_coords = Point2D(self._ego_state.rear_axle.x, self._ego_state.rear_axle.y)

        self._traffic_light_data, self._carla_traffice_lights = get_traffic_light_data(self._vehicle.get_world())

        # dtpp_map = DtppMap(self.map, self.topology)
        coords, traffic_light_data = get_neighbor_vector_set_map(
            dtpp_map, self._conf.map_features, ego_coords, self._conf.radius, self._traffic_light_data
        )
        # tensor 处理，待检查
        ego_agent_past, neighbor_agents_past = agent_past_process(
        ego_agent_past, time_stamps_past, past_tracked_objects_tensor_list, past_tracked_objects_types, self._conf.num_agents)
        vector_map = map_process(self._ego_state.rear_axle, coords, traffic_light_data, config=self._conf, device=device)

        data = {"ego_agent_past": ego_agent_past[1:], 
                "neighbor_agents_past": neighbor_agents_past[:, 1:]}
        data.update(vector_map)
        data = convert_to_model_inputs(data, device)
        if len(self._ego_state_buffer) == self._conf.window_size and len(self._observation_buffer) == self._conf.window_size:
            self._is_ready = True
        else:
            self._is_ready = False
            return None
        return data

    @property
    def traffic_light_data(self):
        return self._traffic_light_data

    @property
    def carla_traffice_lights(self):
        return self._carla_traffice_lights

    @property
    def ego_state(self):
        return self._ego_state
    
    @property
    def observation(self):
        return self._observation

    def get_traffic_light_lane(self, dtpp_map: DtppMap) -> List[Dict]:
        traffic_light_lanes: List[Dict] = []
        candidate_traffic_lanes = dtpp_map.get_candidate_traffic_lanes(self._vehicle)
        logger.debug(
            "candidate_lanes entry_points id list: {}".format(
                [
                    (
                        lane["entry"].road_id,
                        lane["entry"].lane_id,
                        lane["entry"].junction_id,
                    )
                    for lane in candidate_traffic_lanes
                ]
            )
        )

        for lane in candidate_traffic_lanes:
            for tfl in self.carla_traffice_lights:
                if tfl.state != TrafficLightStatusType.RED:
                    continue
                affected_wp = tfl.get_affected_lane_waypoints()
                id_pairs = [
                    (wp.road_id, wp.lane_id, wp.junction_id) for wp in affected_wp
                ]
                entry = lane["entry"]
                if (entry.road_id, entry.lane_id, entry.junction_id) in id_pairs:
                    traffic_light_lanes.append(lane)
                    logger.debug(
                        f"Found traffic light lane entry_point: ({entry.road_id}, {entry.lane_id}, {entry.junction_id})"
                    )
        # if not traffic_light_lanes:
        #     logger.debug("No traffic light lane found")
        return traffic_light_lanes


