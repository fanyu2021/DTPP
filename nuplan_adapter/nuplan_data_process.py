'''
Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
Author: 范雨
Date: 2025-01-17 17:25:28
LastEditTime: 2025-01-17 17:39:10
LastEditors: 范雨
Description: 
'''

import torch
from typing import List, Tuple, Dict, Set, Optional

from nuplan_adapter.nuplan_data import EgoInternalIndex, AgentInternalIndex, TrackedObjectType, Point2D


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


def map_process(anchor_state, coords, traffic_light_data, map_features, max_elements, max_points, interpolation_method):
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

    for feature_name in map_features:
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
                    max_elements[feature_name],
                    max_points[feature_name],
                    traffic_light_encoding_dim,
                    interpolation=interpolation_method  # apply interpolation only for lane features
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

    for feature_name in map_features:
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


def convert_to_model_inputs(data, device):
    tensor_data = {}
    for k, v in data.items():
        tensor_data[k] = v.float().unsqueeze(0).to(device)

    return tensor_data

def sampled_past_timestamps_to_tensor(past_time_stamps: List[TimePoint]) -> torch.Tensor:
    """
    Converts a list of N past timestamps into a 1-d tensor of shape [N]. The field is the timestamp in uS.
    :param past_time_stamps: The time stamps to convert.
    :return: The converted tensor.
    """
    flat = [t.time_us for t in past_time_stamps]
    return torch.tensor(flat, dtype=torch.int64)

def get_neighbor_vector_set_map(
    map_api: AbstractMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    route_roadblock_ids: List[str],
    traffic_light_status_data: List[TrafficLightStatusData],
) -> Tuple[Dict[str, MapObjectPolylines], Dict[str, LaneSegmentTrafficLightData]]:
    """
    Extract neighbor vector set map information around ego vehicle.
    :param map_api: map to perform extraction on.
    :param map_features: Name of map features to extract.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about vector map query range.
    :param route_roadblock_ids: List of ids of roadblocks/roadblock connectors (lane groups) within goal route.
    :param traffic_light_status_data: A list of all available data at the current time step.
    :return:
        coords: Dictionary mapping feature name to polyline vector sets.
        traffic_light_data: Dictionary mapping feature name to traffic light info corresponding to map elements
            in coords.
    :raise ValueError: if provided feature_name is not a valid VectorFeatureLayer.
    """
    coords: Dict[str, MapObjectPolylines] = {}
    traffic_light_data: Dict[str, LaneSegmentTrafficLightData] = {}
    feature_layers: List[VectorFeatureLayer] = []

    for feature_name in map_features:
        try:
            feature_layers.append(VectorFeatureLayer[feature_name])
        except KeyError:
            raise ValueError(f"Object representation for layer: {feature_name} is unavailable")

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(map_api, point, radius)

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # lane traffic light data
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_status_data
        )

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lanes_left.polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lanes_right.polylines)

    # extract route
    if VectorFeatureLayer.ROUTE_LANES in feature_layers:
        route_polylines = get_route_lane_polylines_from_roadblock_ids(map_api, point, radius, route_roadblock_ids)
        coords[VectorFeatureLayer.ROUTE_LANES.name] = route_polylines

    # extract generic map objects
    for feature_layer in feature_layers:
        if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
            polygons = get_map_object_polygons(
                map_api, point, radius, VectorFeatureLayerMapping.semantic_map_layer(feature_layer)
            )
            coords[feature_layer.name] = polygons

    return coords, traffic_light_data

def sampled_past_ego_states_to_tensor(past_ego_states: List[EgoState]) -> torch.Tensor:
    """
    Converts a list of N ego states into a N x 7 tensor. The 7 fields are as defined in `EgoInternalIndex`
    :param past_ego_states: The ego states to convert.
    :return: The converted tensor.
    """
    output = torch.zeros((len(past_ego_states), EgoInternalIndex.dim()), dtype=torch.float32)
    for i in range(0, len(past_ego_states), 1):
        output[i, EgoInternalIndex.x()] = past_ego_states[i].rear_axle.x
        output[i, EgoInternalIndex.y()] = past_ego_states[i].rear_axle.y
        output[i, EgoInternalIndex.heading()] = past_ego_states[i].rear_axle.heading
        output[i, EgoInternalIndex.vx()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.x
        output[i, EgoInternalIndex.vy()] = past_ego_states[i].dynamic_car_state.rear_axle_velocity_2d.y
        output[i, EgoInternalIndex.ax()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.x
        output[i, EgoInternalIndex.ay()] = past_ego_states[i].dynamic_car_state.rear_axle_acceleration_2d.y

    return output

def sampled_tracked_objects_to_tensor_list(past_tracked_objects):
    object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.PEDESTRIAN, TrackedObjectType.BICYCLE]
    output = []
    output_types = []
    track_token_ids = {}

    for i in range(len(past_tracked_objects)):
        tensorized, track_token_ids, agent_types = extract_agent_tensor(past_tracked_objects[i].tracked_objects, track_token_ids, object_types)
        output.append(tensorized)
        output_types.append(agent_types)

    return output, output_types

def agent_past_process(past_ego_states, past_time_stamps, past_tracked_objects, tracked_objects_types, num_agents):
    agents_states_dim = Agents.agents_states_dim()
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects

    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    ego_tensor = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    agent_history = filter_agents_tensor(agents, reverse=True)
    agent_types = tracked_objects_types[-1]

    if agent_history[-1].shape[0] == 0:
        # Return zero tensor when there are no agents in the scene
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