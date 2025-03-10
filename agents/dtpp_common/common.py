# from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple, Dict
from scipy.interpolate import interp1d

import time
import math
# import logging
from custom_format import *
from dataclasses import dataclass, field


import carla


from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.training.preprocessing.features.agents import Agents

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D
from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import *
from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType, TrafficLightStatuses, Transform

from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
    LaneSegmentTrafficLightData,
    VectorFeatureLayer,
)


from agents.dtpp_common.common_utils import get_neighbor_vector_set_map, DtppMap, DtppLane


import torch


WINDOW_SIZE = 22

@dataclass(frozen=True)
class DtppDataConfig:
    window_size: int = WINDOW_SIZE
    num_agents = 20
    past_time_steps = 21
    # name of map features to be extracted.
    map_features = ['LANE', 'ROUTE_LANES', 'CROSSWALK']
    # maximum number of elements to extract per feature layer.
    max_elements = {'LANE': 40, 'ROUTE_LANES': 10, 'CROSSWALK': 5} 
    # maximum number of points per feature to extract per feature layer.
    max_points = {'LANE': 50, 'ROUTE_LANES': 50, 'CROSSWALK': 30} 
    radius = 80 # [m] query radius scope relative to the current pose.
    interpolation_method = 'linear'


def get_state_list_from_actor(timestamp_us: int, actor: carla.Actor) -> List[float]:
    x = actor.get_location().x
    y = actor.get_location().y
    heading = math.radians(actor.get_transform().rotation.yaw)
    vx = actor.get_velocity().x
    vy = actor.get_velocity().y
    ax = actor.get_acceleration().x
    ay = actor.get_acceleration().y
    tire_steering_angle = 0 # TODO: get from actor
    return [timestamp_us, x, y, heading, vx, vy, ax, ay, tire_steering_angle]

def get_ego_state_list_from_actor(timestamp_us: int, ego: carla.Actor) -> EgoState:
    # time_us = time.time() * 1e6 放在外面，避免每次调用都计算
    state_vector_list = get_state_list_from_actor(timestamp_us=timestamp_us, actor=ego)
    vehicle_params = get_vehicle_params_from_actor(actor=ego)
    return EgoState.deserialize(vector=state_vector_list, vehicle=vehicle_params)


# def get_state_list_from_actor(timestamp_us: int, actor: carla.Actor) -> List[float]:
#     x = actor.get_location().x
#     y = actor.get_location().y
#     heading = math.radians(actor.get_transform().rotation.yaw)
#     vx = actor.get_velocity().x
#     vy = actor.get_velocity().y
#     ax = actor.get_acceleration().x
#     ay = actor.get_acceleration().y
#     tire_steering_angle = 0 # TODO: get from actor
#     return [timestamp_us, x, y, heading, vx, vy, ax, ay, tire_steering_angle]

def get_vehicle_params_from_actor(actor: carla.Actor) -> VehicleParameters:
    # TODO: (fanyu)这是tesla model 3的数据，需要改成其他车型的数据
    bounding_box = actor.bounding_box
    width = bounding_box.extent.x * 2
    front_length = bounding_box.extent.y * 2
    rear_length = bounding_box.extent.y
    cog_position_from_rear_axle = 0.8
    wheel_base = 2.875
    vehicle_name = actor.type_id
    vehicle_type = actor.type_id
    height = bounding_box.extent.z * 2

    return VehicleParameters(
        width=width,
        front_length=front_length,
        rear_length=rear_length,
        cog_position_from_rear_axle=cog_position_from_rear_axle,
        wheel_base=wheel_base,
        vehicle_name=vehicle_name,
        vehicle_type=vehicle_type,
        height=height,
    )

def get_tracked_object_type(actor: carla.Actor) -> TrackedObjectType:
    if actor.type_id.startswith('vehicle.*'):
        if actor.type_id.endswith('crossbike'): # vehicle.bh.crossbike
            return TrackedObjectType.BICYCLE
        else:
            return TrackedObjectType.VEHICLE
    elif actor.type_id.startswith("walker.*"):
        return TrackedObjectType.PEDESTRIAN
    else:
        raise ValueError("Unsupported actor type")

def get_tracked_object_oriented_box(actor: carla.Actor) -> OrientedBox:
    centor = StateSE2.deserialize([actor.get_location().x, actor.get_location().y, math.radians(actor.get_transform().get_rotation().yaw)])
    length = actor.bounding_box.extent.x * 2
    width = actor.bounding_box.extent.y * 2
    high = actor.bounding_box.extent.z * 2
    return OrientedBox(center=centor, length=length, width=width, height=high)

def get_agent_from_actor(actor: carla.Actor) -> Agent:
    tracked_object_type = get_tracked_object_type(actor=actor)
    oriented_box = get_tracked_object_oriented_box(actor=actor)
    velocity = StateVector2D(x=actor.get_velocity().x, y=actor.get_velocity().y)
    metadata = SceneObjectMetadata(
        timestamp_us=0, # TODO: (fanyu) 暂时不给时间戳，看看后面怎么处理
        token="", # TODO: (fanyu) 暂时不给时间戳，看看后面怎么处理
        track_id=actor.id,
        track_token="", # TODO: (fanyu) 暂时不给时间戳，看看后面怎么处理
        category_name=actor.type_id
    )
    
    return Agent(
        tracked_object_type=tracked_object_type,
        oriented_box=oriented_box,
        velocity=velocity,
        metadata=metadata,
        angular_velocity=None,
        predictions=None,
        past_trajectory=None
    )

def get_tracked_objects_from_actors(actors: List[carla.Actor]) -> TrackedObjects:
    tracked_agents = []
    for actor in actors:
        # tracked_objects.append(TrackedObjects(tracked_objects=[actor]))
        agent = get_agent_from_actor(actor=actor)
        tracked_agents.append(agent)
        
    return TrackedObjects(tracked_objects=tracked_agents)

def convert_to_model_inputs(features, device) -> Dict[str, torch.Tensor]:
    tensor_data = {}
    for k, v in features.items():
        tensor_data[k] = v.float().unsqueeze(0).to(device)
    return tensor_data

def get_tracked_actors(world: carla.World) -> List[carla.Actor]:
    func = lambda a: a.type_id.startswith('vehicle.*') or a.type_id.startswith('walker.*') or a.type_id.endswith('crossbike')
    return list(filter(func, world.get_actors()))

def get_traffic_light_data(world: carla.World) -> Tuple[List[TrafficLightStatusData], List[carla.Actor]]:
    traffic_lights = world.get_actors().filter('traffic.traffic_light')
    traffic_light_status_list = []
    traffice_light_state_map = {
        carla.TrafficLightState.Red: TrafficLightStatusType.RED,
        carla.TrafficLightState.Green: TrafficLightStatusType.GREEN,
        carla.TrafficLightState.Yellow: TrafficLightStatusType.YELLOW,
        carla.TrafficLightState.Off: TrafficLightStatusType.UNKNOWN,
        carla.TrafficLightState.Unknown: TrafficLightStatusType.UNKNOWN,      
    }
    
    # status: TrafficLightStatusType  # Status: green, red
    # lane_connector_id: int  # lane connector id, where this traffic light belongs to
    # timestamp: int  # Timestamp
    for traffic_light in traffic_lights:
        status = traffice_light_state_map[traffic_light.get_state()]
        traffic_light_status = TrafficLightStatusData(
            # id=traffic_light.get_pole_index(), # TODO(fanyu): 这里需要修改，get_opendrive_id()目前是opendrive_id
            status=status,
            # location=Point2D(traffic_light.get_location().x, traffic_light.get_location().y),
            # orientation=math.radians(traffic_light.get_transform().get_rotation().yaw)
            lane_connector_id=traffic_light.get_affected_lane_waypoints()[0].junction_id,
            timestamp=time.time()*1e6
        )
        traffic_light_status_list.append(traffic_light_status)
    return traffic_light_status_list, traffic_lights


def global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * torch.cos(anchor_heading) + velocity[:, 1] * torch.sin(anchor_heading)
    velocity_y = velocity[:, 1] * torch.cos(anchor_heading) - velocity[:, 0] * torch.sin(anchor_heading)

    return torch.stack([velocity_x, velocity_y], dim=-1)

def convert_absolute_quantities_to_relative(agent_state, ego_state, agent_type='ego'):
    """
    Converts the agent' poses and relative velocities from absolute to ego-relative coordinates.
    :param agent_state: The agent states to convert, in the AgentInternalIndex schema.
    :param ego_state: The ego state to convert, in the EgoInternalIndex schema.
    :return: The converted states, in AgentInternalIndex schema.
    """
    ego_pose = torch.tensor(
        [
            float(ego_state[EgoInternalIndex.x()].item()),
            float(ego_state[EgoInternalIndex.y()].item()),
            float(ego_state[EgoInternalIndex.heading()].item()),
        ],
        dtype=torch.float64,
    )

    if agent_type == 'ego':
        agent_global_poses = agent_state[:, [EgoInternalIndex.x(), EgoInternalIndex.y(), EgoInternalIndex.heading()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        agent_state[:, EgoInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, EgoInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, EgoInternalIndex.heading()] = transformed_poses[:, 2].float()
    else:
        agent_global_poses = agent_state[:, [AgentInternalIndex.x(), AgentInternalIndex.y(), AgentInternalIndex.heading()]]
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        transformed_velocities = global_velocity_to_local(agent_global_velocities, ego_pose[-1])
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()

    return agent_state

def agent_past_process(past_ego_states, past_time_stamps, past_tracked_objects, tracked_objects_types, num_agents):
    agents_states_dim = Agents.agents_states_dim()
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects

    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    logger.warning(f"--- ego_history.shape: {ego_history.shape}")
    logger.info(f"---- anchor_ego_state: {anchor_ego_state}")
    ego_tensor = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    agent_history = filter_agents_tensor(agents, reverse=True)
    agent_types = tracked_objects_types[-1]

    if agent_history[-1].shape[0] == 0:
        # Return zero tensor when there are no agents in the scene
        logger.info("--- There is no agent in deque.")
        agents_tensor = torch.zeros((len(agent_history), 0, agents_states_dim)).float()
    else:
        local_coords_agent_states = []
        padded_agent_states = pad_agent_states(agent_history, reverse=True)

        for agent_state in padded_agent_states:
            local_coords_agent_states.append(convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))
    
        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
    
        agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)

    agents = torch.zeros((num_agents, agents_tensor.shape[0], agents_tensor.shape[-1]+3), dtype=torch.float32)

    # sort agents according to distance to ego
    logger.info(f'--- agents_tensor.shape = {agents_tensor.shape}')
    distance_to_ego = torch.norm(agents_tensor[-1, :, :2], dim=-1)
    indices = list(torch.argsort(distance_to_ego).numpy())[:num_agents]

    # fill agent features into the array
    added_agents = 0
    for i in indices:
        if added_agents >= num_agents:
            break
        
        if agents_tensor[-1, i, 0] < -6.0:
            continue

        agents[added_agents, :, :agents_tensor.shape[-1]] = agents_tensor[:, i, :agents_tensor.shape[-1]]

        if agent_types[i] == TrackedObjectType.VEHICLE:
            agents[added_agents, :, agents_tensor.shape[-1]:] = torch.tensor([1, 0, 0])
        elif agent_types[i] == TrackedObjectType.PEDESTRIAN:
            agents[added_agents, :, agents_tensor.shape[-1]:] = torch.tensor([0, 1, 0])
        else:
            agents[added_agents, :, agents_tensor.shape[-1]:] = torch.tensor([0, 0, 1])

        added_agents += 1

    return ego_tensor, agents

def interpolate_points(coords: torch.Tensor, max_points: int, interpolation: str) -> torch.Tensor:
    """
    Interpolate points within map element to maintain fixed size.
    :param coords: Sequence of coordinate points representing map element. <torch.Tensor: num_points, 2>
    :param max_points: Desired size to interpolate to.
    :param interpolation: Torch interpolation mode. Available options: 'linear' and 'area'.
    :return: Coordinate points interpolated to max_points size.
    :raise ValueError: If coordinates dimensions are not valid.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}. Expected shape: (*, 2)")

    x_coords = coords[:, 0].unsqueeze(0).unsqueeze(0)
    y_coords = coords[:, 1].unsqueeze(0).unsqueeze(0)
    align_corners = True if interpolation == 'linear' else None
    x_coords = torch.nn.functional.interpolate(x_coords, max_points, mode=interpolation, align_corners=align_corners)
    y_coords = torch.nn.functional.interpolate(y_coords, max_points, mode=interpolation, align_corners=align_corners)
    coords = torch.stack((x_coords, y_coords), dim=-1).squeeze()

    return coords


def convert_feature_layer_to_fixed_size(ego_pose, feature_coords, feature_tl_data, max_elements, max_points,
                                         traffic_light_encoding_dim, interpolation):
    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        raise ValueError(f"Size between feature coords and traffic light data inconsistent: {len(feature_coords)}, {len(feature_tl_data)}")

    # trim or zero-pad elements to maintain fixed size
    coords_tensor = torch.zeros((max_elements, max_points, 2), dtype=torch.float32)
    avails_tensor = torch.zeros((max_elements, max_points), dtype=torch.bool)
    tl_data_tensor = (
        torch.zeros((max_elements, max_points, traffic_light_encoding_dim), dtype=torch.float32)
        if feature_tl_data is not None else None
    )

    # get elements according to the mean distance to the ego pose
    mapping = {}
    for i, e in enumerate(feature_coords):
        dist = torch.norm(e - ego_pose[None, :2], dim=-1).min()
        mapping[i] = dist

    mapping = sorted(mapping.items(), key=lambda item: item[1])
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    for idx, element_idx in enumerate(sorted_elements):
        element_coords = feature_coords[element_idx[0]]
    
        # interpolate to maintain fixed size if the number of points is not enough
        element_coords = interpolate_points(element_coords, max_points, interpolation=interpolation)
        coords_tensor[idx] = element_coords
        avails_tensor[idx] = True  # specify real vs zero-padded data

        if tl_data_tensor is not None and feature_tl_data is not None:
            tl_data_tensor[idx] = feature_tl_data[element_idx[0]]

    return coords_tensor, tl_data_tensor, avails_tensor

def _validate_state_se2_tensor_shape(tensor: torch.Tensor, expected_first_dim: Optional[int] = None) -> None:
    """
    Validates that a tensor is of the proper shape for a tensorized StateSE2.
    :param tensor: The tensor to validate.
    :param expected_first_dim: The expected first dimension. Can be one of three values:
        * 1: Tensor is expected to be of shape (3,)
        * 2: Tensor is expected to be of shape (N, 3)
        * None: Either shape is acceptable
    """
    expected_feature_dim = 3
    if len(tensor.shape) == 2 and tensor.shape[1] == expected_feature_dim:
        if expected_first_dim is None or expected_first_dim == 2:
            return
    if len(tensor.shape) == 1 and tensor.shape[0] == expected_feature_dim:
        if expected_first_dim is None or expected_first_dim == 1:
            return

    raise ValueError(f"Improper se2 tensor shape: {tensor.shape}")

def state_se2_tensor_to_transform_matrix(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transforms a state of the form [x, y, heading] into a 3x3 transform matrix.
    :param input_data: the input data as a 3-d tensor.
    :return: The output 3x3 transformation matrix.
    """
    _validate_state_se2_tensor_shape(input_data, expected_first_dim=1)

    if precision is None:
        precision = input_data.dtype

    x: float = float(input_data[0].item())
    y: float = float(input_data[1].item())
    h: float = float(input_data[2].item())

    cosine: float = math.cos(h)
    sine: float = math.sin(h)

    return torch.tensor(
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]], dtype=precision, device=input_data.device
    )

def coordinates_to_local_frame(
    coords: torch.Tensor, anchor_state: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transform a set of [x, y] coordinates without heading to the the given frame.
    :param coords: <torch.Tensor: num_coords, 2> Coordinates to be transformed, in the form [x, y].
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param precision: The precision with which to allocate the intermediate tensors. If None, then it will be inferred from the input precisions.
    :return: <torch.Tensor: num_coords, 2> Transformed coordinates.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}")

    if precision is None:
        if coords.dtype != anchor_state.dtype:
            raise ValueError("Mixed datatypes provided to coordinates_to_local_frame without precision specifier.")
        precision = coords.dtype

    # torch.nn.functional.pad will crash with 0-length inputs.
    # In that case, there are no coordinates to transform.
    if coords.shape[0] == 0:
        return coords

    # Extract transform
    transform = state_se2_tensor_to_transform_matrix(anchor_state, precision=precision)
    transform = torch.linalg.inv(transform)

    # Transform the incoming coordinates to homogeneous coordinates
    #  So translation can be done with a simple matrix multiply.
    #
    # [x1, y1]  => [x1, y1, 1]
    # [x2, y2]     [x2, y2, 1]
    # ...          ...
    # [xn, yn]     [xn, yn, 1]
    coords = torch.nn.functional.pad(coords, (0, 1, 0, 0), "constant", value=1.0)

    # Perform the transformation, transposing so the shapes match
    coords = torch.matmul(transform, coords.transpose(0, 1))

    # Transform back from homogeneous coordinates to standard coordinates.
    #   Get rid of the scaling dimension and transpose so output shape matches input shape.
    result = coords.transpose(0, 1)
    result = result[:, :2]

    return result

def vector_set_coordinates_to_local_frame(
    coords: torch.Tensor,
    avails: torch.Tensor,
    anchor_state: torch.Tensor,
    output_precision: Optional[torch.dtype] = torch.float32,
) -> torch.Tensor:
    """
    Transform the vector set map element coordinates from global frame to ego vehicle frame, as specified by
        anchor_state.
    :param coords: Coordinates to transform. <torch.Tensor: num_elements, num_points, 2>.
    :param avails: Availabilities mask identifying real vs zero-padded data in coords.
        <torch.Tensor: num_elements, num_points>.
    :param anchor_state: The coordinate frame to transform to, in the form [x, y, heading].
    :param output_precision: The precision with which to allocate output tensors.
    :return: Transformed coordinates.
    :raise ValueError: If coordinates dimensions are not valid or don't match availabilities.
    """
    if len(coords.shape) != 3 or coords.shape[2] != 2:
        raise ValueError(f"Unexpected coords shape: {coords.shape}. Expected shape: (*, *, 2)")

    if coords.shape[:2] != avails.shape:
        raise ValueError(f"Mismatching shape between coords and availabilities: {coords.shape[:2]}, {avails.shape}")

    # Flatten coords from (num_map_elements, num_points_per_element, 2) to
    #   (num_map_elements * num_points_per_element, 2) for easier processing.
    num_map_elements, num_points_per_element, _ = coords.size()
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    # Apply transformation using adequate precision
    coords = coordinates_to_local_frame(coords.double(), anchor_state.double(), precision=torch.float64)

    # Reshape to original dimensionality
    coords = coords.reshape(num_map_elements, num_points_per_element, 2)

    # Output with specified precision
    coords = coords.to(output_precision)

    # ignore zero-padded data
    coords[~avails] = 0.0

    return coords

def polyline_process(polylines, avails, traffic_light=None):
    dim = 3 if traffic_light is None else 7
    new_polylines = torch.zeros((polylines.shape[0], polylines.shape[1], dim), dtype=torch.float32)

    for i in range(polylines.shape[0]):
        if avails[i][0]:
            polyline = polylines[i]
            polyline_heading = torch.atan2(polyline[1:, 1]-polyline[:-1, 1], polyline[1:, 0]-polyline[:-1, 0])
            polyline_heading = torch.fmod(polyline_heading, 2*torch.pi)
            polyline_heading = torch.cat([polyline_heading, polyline_heading[-1].unsqueeze(0)], dim=0).unsqueeze(-1)
            if traffic_light is None:
                new_polylines[i] = torch.cat([polyline, polyline_heading], dim=-1)
            else:
                new_polylines[i] = torch.cat([polyline, polyline_heading, traffic_light[i]], dim=-1)

    return new_polylines

def map_process(anchor_state, coords, traffic_light_data, config: DtppDataConfig, device:str = 'cuda'): # map_features, max_elements, max_points, interpolation_method
    # convert data to tensor list
    anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float32)
    list_tensor_data = {}

    for feature_name, feature_coords in coords.items():
        list_feature_coords = []

        # Pack coords into tensor list
        for element_coords in feature_coords.to_vector():
            list_feature_coords.append(torch.tensor(element_coords, dtype=torch.float32))
        list_tensor_data[f"coords.{feature_name}"] = list_feature_coords

        # Pack traffic light data into tensor list if it exists
        if feature_name in traffic_light_data:
            list_feature_tl_data = []

            for element_tl_data in traffic_light_data[feature_name].to_vector():
                list_feature_tl_data.append(torch.tensor(element_tl_data, dtype=torch.float32))
            list_tensor_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data

    tensor_output = {}
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    for feature_name in config.map_features:
        if f"coords.{feature_name}" in list_tensor_data:
            feature_coords = list_tensor_data[f"coords.{feature_name}"]

            feature_tl_data = (
                list_tensor_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in list_tensor_data
                else None
            )

            coords, tl_data, avails = convert_feature_layer_to_fixed_size(
                    anchor_state_tensor,
                    feature_coords,
                    feature_tl_data,
                    config.max_elements[feature_name],
                    config.max_points[feature_name],
                    traffic_light_encoding_dim,
                    interpolation=config.interpolation_method  # apply interpolation only for lane features
                    if feature_name
                    in [
                        VectorFeatureLayer.LANE.name,
                        VectorFeatureLayer.LEFT_BOUNDARY.name,
                        VectorFeatureLayer.RIGHT_BOUNDARY.name,
                        VectorFeatureLayer.ROUTE_LANES.name,
                        VectorFeatureLayer.CROSSWALK.name
                    ]
                    else None,
            )

            coords = vector_set_coordinates_to_local_frame(coords, avails, anchor_state_tensor)

            tensor_output[f"vector_set_map.coords.{feature_name}"] = coords
            tensor_output[f"vector_set_map.availabilities.{feature_name}"] = avails

            if tl_data is not None:
                tensor_output[f"vector_set_map.traffic_light_data.{feature_name}"] = tl_data

    for feature_name in config.map_features:
        if feature_name == "LANE":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}']
            traffic_light_state = tensor_output[f'vector_set_map.traffic_light_data.{feature_name}']
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}']
            vector_map_lanes = polyline_process(polylines, avails, traffic_light_state)

        elif feature_name == "CROSSWALK":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}']
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}']
            vector_map_crosswalks = polyline_process(polylines, avails)

        elif feature_name == "ROUTE_LANES":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}']
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}']
            vector_map_route_lanes = polyline_process(polylines, avails)

        else:
            pass

    vector_map_output = {'map_lanes': vector_map_lanes, 'map_crosswalks': vector_map_crosswalks, 'route_lanes': vector_map_route_lanes}

    return vector_map_output

def extract_agent_tensor(tracked_objects, track_token_ids, object_types):
    agents = tracked_objects.get_tracked_objects_of_types(object_types)
    agent_types = []
    output = torch.zeros((len(agents), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)

    for idx, agent in enumerate(agents):
        if agent.track_token not in track_token_ids:
            track_token_ids[agent.track_token] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent.track_token]

        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent.velocity.x
        output[idx, AgentInternalIndex.vy()] = agent.velocity.y
        output[idx, AgentInternalIndex.heading()] = agent.center.heading
        output[idx, AgentInternalIndex.width()] = agent.box.width
        output[idx, AgentInternalIndex.length()] = agent.box.length
        output[idx, AgentInternalIndex.x()] = agent.center.x
        output[idx, AgentInternalIndex.y()] = agent.center.y
        print(f"------ output: {output}")
        agent_types.append(agent.tracked_object_type)

    return output, track_token_ids, agent_types

def sampled_tracked_agents_to_tensor_list(past_tracked_objects):
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids, agent_types = extract_agent_tensor(past_tracked_objects[i].tracked_objects, track_token_ids, object_types)
        output.append(tensorized)
        output_types.append(agent_types)

    return output, output_types


def _get_fixed_timesteps(state: EgoState, future_horizon: float, step_interval: float) -> List[float]:
    """
    Get a fixed array of timesteps starting from a state's time.

    :param state: input state
    :param future_horizon: [s] future time horizon
    :param step_interval: [s] interval between steps in the array
    :return: constructed timestep list
    """
    timesteps = np.arange(0.0, future_horizon, step_interval) + step_interval
    timesteps += state.time_point.time_s

    return list(timesteps.tolist())

def matrix_from_pose(pose: StateSE2) -> npt.NDArray[np.float64]:
    """
    Converts a 2D pose to a 3x3 transformation matrix

    :param pose: 2D pose (x, y, yaw)
    :return: 3x3 transformation matrix
    """
    return np.array(
        [
            [np.cos(pose.heading), -np.sin(pose.heading), pose.x],
            [np.sin(pose.heading), np.cos(pose.heading), pose.y],
            [0, 0, 1],
        ]
    )
    
    
def pose_from_matrix(transform_matrix: npt.NDArray[np.float32]) -> StateSE2:
    """
    Converts a 3x3 transformation matrix to a 2D pose
    :param transform_matrix: 3x3 transformation matrix
    :return: 2D pose (x, y, yaw)
    """
    if transform_matrix.shape != (3, 3):
        raise RuntimeError(f"Expected a 3x3 transformation matrix, got {transform_matrix.shape}")

    heading = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

    return StateSE2(transform_matrix[0, 2], transform_matrix[1, 2], heading)
    
def relative_to_absolute_poses(origin_pose: StateSE2, relative_poses: List[StateSE2]) -> List[StateSE2]:
    """
    Converts a list of SE2 poses from relative to absolute coordinates using an origin pose.
    :param origin_pose: Reference origin pose
    :param relative_poses: list of relative poses to convert
    :return: list of converted absolute poses
    """
    relative_transforms: npt.NDArray[np.float64] = np.array([matrix_from_pose(pose) for pose in relative_poses])
    origin_transform = matrix_from_pose(origin_pose)
    absolute_transforms: npt.NDArray[np.float32] = origin_transform @ relative_transforms
    absolute_poses = [pose_from_matrix(transform_matrix) for transform_matrix in absolute_transforms]

    return absolute_poses

def _project_from_global_to_ego_centric_ds(
    ego_poses: npt.NDArray[np.float32], values: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Project value from the global xy frame to the ego centric ds frame.

    :param ego_poses: [x, y, heading] with size [planned steps, 3].
    :param values: values in global frame with size [planned steps, 2]
    :return: values projected onto the new frame with size [planned steps, 2]
    """
    headings = ego_poses[:, -1:]

    values_lon = values[:, :1] * np.cos(headings) + values[:, 1:2] * np.sin(headings)
    values_lat = values[:, :1] * np.sin(headings) - values[:, 1:2] * np.cos(headings)
    values = np.concatenate((values_lon, values_lat), axis=1)
    return values

def _get_velocity_and_acceleration(
    ego_poses: List[StateSE2], ego_history: Deque[EgoState], timesteps: List[float]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    """
    Given the past, current and planned ego poses, estimate the velocity and acceleration by taking the derivatives.

    :param ego_poses: a list of the planned ego poses
    :param ego_history: the ego history that includes the current
    :param timesteps: [s] timesteps of the planned ego poses
    :return: the approximated velocity and acceleration in ego centric frame
    """
    ego_history_len = len(ego_history)
    current_ego_state = ego_history[-1]

    # Past and current
    timesteps_past_current = [state.time_point.time_s for state in ego_history]
    ego_poses_past_current: npt.NDArray[np.float32] = np.stack(
        [np.array(state.rear_axle.serialize()) for state in ego_history]
    )

    # Planned
    dt = current_ego_state.time_point.time_s - ego_history[-2].time_point.time_s
    timesteps_current_planned: npt.NDArray[np.float32] = np.array([current_ego_state.time_point.time_s] + timesteps)
    ego_poses_current_planned: npt.NDArray[np.float32] = np.stack(
        [current_ego_state.rear_axle.serialize()] + [pose.serialize() for pose in ego_poses]
    )

    # Interpolation to have equal space for derivatives
    ego_poses_interpolate = interp1d(
        timesteps_current_planned, ego_poses_current_planned, axis=0, fill_value='extrapolate'
    )
    timesteps_current_planned_interp = np.arange(
        start=current_ego_state.time_point.time_s, stop=timesteps[-1] + 1e-6, step=dt
    )
    ego_poses_current_planned_interp = ego_poses_interpolate(timesteps_current_planned_interp)

    # Combine past current and planned
    timesteps_past_current_planned = [*timesteps_past_current, *timesteps_current_planned_interp[1:]]
    ego_poses_past_current_planned: npt.NDArray[np.float32] = np.concatenate(
        [ego_poses_past_current, ego_poses_current_planned_interp[1:]], axis=0
    )

    # Take derivatives
    ego_velocity_past_current_planned = approximate_derivatives(
        ego_poses_past_current_planned[:, :2], timesteps_past_current_planned, axis=0
    )
    ego_acceleration_past_current_planned = approximate_derivatives(
        ego_poses_past_current_planned[:, :2], timesteps_past_current_planned, axis=0, deriv_order=2
    )

    # Only take the planned for output
    ego_velocity_planned_xy = ego_velocity_past_current_planned[ego_history_len:]
    ego_acceleration_planned_xy = ego_acceleration_past_current_planned[ego_history_len:]

    # Projection
    ego_velocity_planned_ds = _project_from_global_to_ego_centric_ds(
        ego_poses_current_planned_interp[1:], ego_velocity_planned_xy
    )
    ego_acceleration_planned_ds = _project_from_global_to_ego_centric_ds(
        ego_poses_current_planned_interp[1:], ego_acceleration_planned_xy
    )

    # Interpolate back
    ego_velocity_interp_back = interp1d(
        timesteps_past_current_planned[ego_history_len:], ego_velocity_planned_ds, axis=0, fill_value='extrapolate'
    )
    ego_acceleration_interp_back = interp1d(
        timesteps_past_current_planned[ego_history_len:],
        ego_acceleration_planned_ds,
        axis=0,
        fill_value='extrapolate',
    )

    ego_velocity_planned_ds = ego_velocity_interp_back(timesteps)
    ego_acceleration_planned_ds = ego_acceleration_interp_back(timesteps)

    return ego_velocity_planned_ds, ego_acceleration_planned_ds



def _se2_vel_acc_to_ego_state(
    state: StateSE2,
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    timestamp: float,
    vehicle: VehicleParameters,
) -> EgoState:
    """
    Convert StateSE2, velocity and acceleration to EgoState given a timestamp.

    :param state: input SE2 state
    :param velocity: [m/s] longitudinal velocity, lateral velocity
    :param acceleration: [m/s^2] longitudinal acceleration, lateral acceleration
    :param timestamp: [s] timestamp of state
    :return: output agent state
    """
    return EgoState.build_from_rear_axle(
        rear_axle_pose=state,
        rear_axle_velocity_2d=StateVector2D(*velocity),
        rear_axle_acceleration_2d=StateVector2D(*acceleration),
        tire_steering_angle=0.0,
        time_point=TimePoint(int(timestamp * 1e6)),
        vehicle_parameters=vehicle,
        is_in_auto_mode=True,
    )

def _get_absolute_agent_states_from_numpy_poses(
    poses: npt.NDArray[np.float32], ego_history: Deque[EgoState], timesteps: List[float]
) -> List[EgoState]:
    """
    Converts an array of relative numpy poses to a list of absolute EgoState objects.

    :param poses: input relative poses
    :param ego_history: the history of the ego state, including the current
    :param timesteps: timestamps corresponding to each state
    :return: list of agent states
    """
    ego_state = ego_history[-1]
    relative_states = [StateSE2.deserialize(pose) for pose in poses]
    absolute_states = relative_to_absolute_poses(ego_state.rear_axle, relative_states)
    velocities, accelerations = _get_velocity_and_acceleration(absolute_states, ego_history, timesteps)
    agent_states = [
        _se2_vel_acc_to_ego_state(state, velocity, acceleration, timestep, ego_state.car_footprint.vehicle_parameters)
        for state, velocity, acceleration, timestep in zip(absolute_states, velocities, accelerations, timesteps)
    ]

    return agent_states

def transform_predictions_to_states(
    predicted_poses: npt.NDArray[np.float32],
    ego_history: Deque[EgoState],
    future_horizon: float,
    step_interval: float,
    include_ego_state: bool = True,
) -> List[EgoState]:
    """
    Transform an array of pose predictions to a list of EgoState.

    :param predicted_poses: input relative poses
    :param ego_history: the history of the ego state, including the current
    :param future_horizon: [s] future time horizon
    :param step_interval: [s] interval between steps in the array
    :param include_ego_state: whether to include the current ego state as the initial state
    :return: transformed absolute states
    """
    ego_state = ego_history[-1]
    timesteps = _get_fixed_timesteps(ego_state, future_horizon, step_interval)
    states = _get_absolute_agent_states_from_numpy_poses(predicted_poses, ego_history, timesteps)

    if include_ego_state:
        states.insert(0, ego_state)

    return states

class DtppInputs(object):
    def __init__(self, vehicle: carla.Actor) -> None:
        self._conf = DtppDataConfig()
        self._ego_state_buffer = deque(maxlen=WINDOW_SIZE) # EgoState
        self._observation_buffer = deque(maxlen=WINDOW_SIZE) # TrackedObjects
        # self.dtpp_map = DtppMap(map, topology, routing)
        self._vehicle = vehicle
        self._is_ready = False
        self._traffic_light_data: List[TrafficLightStatusData] = []
        self._carla_traffice_lights: List[carla.Actor] = []

    def _update_ego_state_buffer(self, timestamp_us: int, ego: carla.Actor) -> Deque[EgoState]:
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
        if not traffic_light_lanes:
            logger.debug("No traffic light lane found")
        return traffic_light_lanes
