"""
Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
Author: 范雨
Date: 2025-01-17 17:25:28
LastEditTime: 2025-02-12 12:15:44
LastEditors: 范雨
Description: 
"""

import torch
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

from typing import List, Tuple, Dict, Set, Optional
import numpy as np
import numpy.typing as npt

import copy

from carla2inputs import *

from nuplan_adapter.nuplan_data import *

# from obs_adapter import *

# from nuplan.common.geometry.torch_geometry import *


def create_model_input_from_carla(
    carla_scenario: CarlaScenario, carla_scenario_input: CarlaScenarioInput
) -> CarlaScenarioInput:
    carla_scenario_input.time_series.appendleft(time.time() * 1e6)  # us
    carla_scenario_input.iteration += 1
    carla_scenario_input.ego_states.appendleft(carla_scenario.ego_state)
    carla_scenario_input.map_lanes = carla_scenario.map_lanes
    carla_scenario_input.route_lines = carla_scenario.route_lines
    carla_scenario_input.road_ids = carla_scenario.road_ids
    # TODO(fanyu): size
    # 如果 ego_states size < 21, 将尾部填充相同的数据

    if len(carla_scenario_input.ego_states) < 22:
        # 计算需要添加的状态数量
        num_states_to_add = 22 - len(carla_scenario_input.ego_states)
        new_states = []
        for i in range(num_states_to_add):
            # 创建 carla_scenario.ego_state 的深拷贝
            new_state = copy.deepcopy(carla_scenario.ego_state)
            new_state.timestamp = carla_scenario.ego_state.timestamp - 100000 * (i + 1)
            # print(f'------- {i}: {new_state.timestamp}')
            new_states.append(new_state)
        carla_scenario_input.ego_states.extend(new_states)
        carla_scenario_input.ego_states.reverse()
        # for state in carla_scenario_input.ego_states:
        #     print(f"----B---- state.timestamp = {state.timestamp}\n")

    for agent in carla_scenario.agents:
        # 如果距离超过20米，则不添加
        is_in_range = (np.sqrt((agent.x - carla_scenario.ego_state.x) ** 2 + (agent.y - carla_scenario.ego_state.y) ** 2) <= 100)
        if agent.id not in carla_scenario_input.agents_map and is_in_range:
            carla_scenario_input.agents_map[agent.id] = Deque(maxlen=22)
            carla_scenario_input.agents_map[agent.id].appendleft(agent)
        elif agent.id not in carla_scenario_input.agents_map and not is_in_range:
            continue
        elif agent.id in carla_scenario_input.agents_map and is_in_range:
            carla_scenario_input.agents_map[agent.id].appendleft(agent)
        elif agent.id in carla_scenario_input.agents_map and not is_in_range:
            carla_scenario_input.agents_map.pop(agent.id)
            continue
        if len(carla_scenario_input.agents_map[agent.id]) < 22:
            num_states_to_add = 22 - len(carla_scenario_input.agents_map[agent.id])
            new_states = []
            for i in range(num_states_to_add):
                new_state = copy.deepcopy(agent)
                new_state.timestamp = agent.timestamp - 100000 * (i + 1)
                new_states.append(new_state)
            carla_scenario_input.agents_map[agent.id].extend(new_states)
            carla_scenario_input.agents_map[agent.id].reverse()
            print(f'------ agent traj size: {len(carla_scenario_input.agents_map[agent.id])}\n')
    print(f'------ agents size: {len(carla_scenario_input.agents_map)}\n')
    return carla_scenario_input


def polyline_process(polylines, avails, traffic_light=None):
    dim = 3 if traffic_light is None else 7
    new_polylines = torch.zeros(
        (polylines.shape[0], polylines.shape[1], dim), dtype=torch.float32
    )

    for i in range(polylines.shape[0]):
        if avails[i][0]:
            polyline = polylines[i]
            polyline_heading = torch.atan2(
                polyline[1:, 1] - polyline[:-1, 1], polyline[1:, 0] - polyline[:-1, 0]
            )
            polyline_heading = torch.fmod(polyline_heading, 2 * torch.pi)
            polyline_heading = torch.cat(
                [polyline_heading, polyline_heading[-1].unsqueeze(0)], dim=0
            ).unsqueeze(-1)
            if traffic_light is None:
                new_polylines[i] = torch.cat([polyline, polyline_heading], dim=-1)
            else:
                new_polylines[i] = torch.cat(
                    [polyline, polyline_heading, traffic_light[i]], dim=-1
                )

    return new_polylines


def interpolate_points(
    coords: torch.Tensor, max_points: int, interpolation: str
) -> torch.Tensor:
    """
    Interpolate points within map element to maintain fixed size.
    :param coords: Sequence of coordinate points representing map element. <torch.Tensor: num_points, 2>
    :param max_points: Desired size to interpolate to.
    :param interpolation: Torch interpolation mode. Available options: 'linear' and 'area'.
    :return: Coordinate points interpolated to max_points size.
    :raise ValueError: If coordinates dimensions are not valid.
    """
    if len(coords.shape) != 2 or coords.shape[1] != 2:
        raise ValueError(
            f"Unexpected coords shape: {coords.shape}. Expected shape: (*, 2)"
        )

    x_coords = coords[:, 0].unsqueeze(0).unsqueeze(0)
    y_coords = coords[:, 1].unsqueeze(0).unsqueeze(0)
    align_corners = True if interpolation == "linear" else None
    x_coords = torch.nn.functional.interpolate(
        x_coords, max_points, mode=interpolation, align_corners=align_corners
    )
    y_coords = torch.nn.functional.interpolate(
        y_coords, max_points, mode=interpolation, align_corners=align_corners
    )
    coords = torch.stack((x_coords, y_coords), dim=-1).squeeze()

    return coords

def convert_feature_layer_to_fixed_size(ego_pose, feature_coords, feature_tl_data, max_elements, max_points,
                                         traffic_light_encoding_dim, interpolation):
    if feature_tl_data is not None and len(feature_coords) != len(feature_tl_data):
        print(f'--- 141 --- feature_coords size: {len(feature_coords)}, feature_tl_data size: {len(feature_tl_data)}')
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
    print(f'---154---feature_coords size: {feature_coords}')
    for i, e in enumerate(feature_coords):
        dist = torch.norm(e - ego_pose[None, :2], dim=-1).min()
        mapping[i] = dist

    mapping = sorted(mapping.items(), key=lambda item: item[1])
    sorted_elements = mapping[:max_elements]

    # pad or trim waypoints in a map element
    for idx, element_idx in enumerate(sorted_elements):
        element_coords = feature_coords[element_idx[0]]

        # interpolate to maintain fixed size if the number of points is not enough
        element_coords = interpolate_points(
            element_coords, max_points, interpolation=interpolation
        )
        coords_tensor[idx] = element_coords
        avails_tensor[idx] = True  # specify real vs zero-padded data

        if tl_data_tensor is not None and feature_tl_data is not None:
            tl_data_tensor[idx] = feature_tl_data[element_idx[0]]
    # print(f'---174---coords_tensor: {len(coords_tensor)}, tl_data_tensor: {tl_data_tensor.shape}, avails_tensor: {avails_tensor.shape}')
    return coords_tensor, tl_data_tensor, avails_tensor


def _validate_state_se2_tensor_shape(
    tensor: torch.Tensor, expected_first_dim: Optional[int] = None
) -> None:
    """
    Validates that a tensor is of the proper shape for a tensorized EgoState.
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
        [[cosine, -sine, x], [sine, cosine, y], [0.0, 0.0, 1.0]],
        dtype=precision,
        device=input_data.device,
    )


def coordinates_to_local_frame(
    coords: torch.Tensor,
    anchor_state: torch.Tensor,
    precision: Optional[torch.dtype] = None,
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
            raise ValueError(
                "Mixed datatypes provided to coordinates_to_local_frame without precision specifier."
            )
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
        raise ValueError(
            f"Unexpected coords shape: {coords.shape}. Expected shape: (*, *, 2)"
        )

    if coords.shape[:2] != avails.shape:
        raise ValueError(
            f"Mismatching shape between coords and availabilities: {coords.shape[:2]}, {avails.shape}"
        )

    # Flatten coords from (num_map_elements, num_points_per_element, 2) to
    #   (num_map_elements * num_points_per_element, 2) for easier processing.
    num_map_elements, num_points_per_element, _ = coords.size()
    coords = coords.reshape(num_map_elements * num_points_per_element, 2)

    # Apply transformation using adequate precision
    coords = coordinates_to_local_frame(
        coords.double(), anchor_state.double(), precision=torch.float64
    )

    # Reshape to original dimensionality
    coords = coords.reshape(num_map_elements, num_points_per_element, 2)

    # Output with specified precision
    coords = coords.to(output_precision)

    # ignore zero-padded data
    coords[~avails] = 0.0

    return coords

def map_process(anchor_state, coords, traffic_light_data, map_features, max_elements, max_points, interpolation_method):
    # convert data to tensor list
    anchor_state_tensor = torch.tensor([anchor_state.x, anchor_state.y, anchor_state.heading], dtype=torch.float32)
    list_tensor_data = {}
    
    print(f'---331--- coords.size:{coords}')

    for feature_name, feature_coords in coords.items():
        list_feature_coords = []
        print(f'---335---feature_name: {feature_name}, feature_coords: {feature_coords}')
        # Pack coords into tensor list
        for element_coords in feature_coords.to_vector():
            list_feature_coords.append(torch.tensor(element_coords, dtype=torch.float32))
        list_tensor_data[f"coords.{feature_name}"] = list_feature_coords
        print(f'---338--- list_feature_coords: {list_feature_coords}')

        # Pack traffic light data into tensor list if it exists
        if feature_name in traffic_light_data:
            list_feature_tl_data = []

            for element_tl_data in traffic_light_data[feature_name].to_vector():
                list_feature_tl_data.append(torch.tensor(element_tl_data, dtype=torch.float32))
            list_tensor_data[f"traffic_light_data.{feature_name}"] = list_feature_tl_data
            print(f'---349--- list_feature_tl_data: {list_feature_tl_data}')

    tensor_output = {}
    traffic_light_encoding_dim = LaneSegmentTrafficLightData.encoding_dim()

    for feature_name in map_features:
        if f"coords.{feature_name}" in list_tensor_data:
            print(f'--- 352 --- feature name: {feature_name}')

            # print(f'---358--- len(feature_coords): {len(feature_coords[0].shape)}')
            

            feature_tl_data = (
                list_tensor_data[f"traffic_light_data.{feature_name}"]
                if f"traffic_light_data.{feature_name}" in list_tensor_data
                else None
            )
            # TODO(fanyu): not use traffic light data for now
            feature_tl_data = None
            
            # feature_coords = list_tensor_data[f"coords.{feature_name}"]*len(feature_tl_data)
            feature_coords = list_tensor_data[f"coords.{feature_name}"]
            print(f'---357--- feature_coords: {feature_coords}')
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
            # TODO(fanyu): not use traffic light data for now
            if tl_data is not None:
                tensor_output[f"vector_set_map.traffic_light_data.{feature_name}"] = tl_data

    for feature_name in map_features:
        if feature_name == "LANE":
            polylines = tensor_output[f'vector_set_map.coords.{feature_name}']
            # 目标形状
            original_tensor = torch.tensor([0,0,0,1])
            target_shape = (40, 50, 4)

            # 扩展原始张量的维度，使其与目标形状匹配
            traffic_light_state = original_tensor.unsqueeze(0).unsqueeze(0)  # 扩展为 [1, 1, 4]
            traffic_light_state = traffic_light_state.expand(target_shape)       # 扩展为 [40, 50, 4]
            
            # traffic_light_state = tensor_output[f'vector_set_map.traffic_light_data.{feature_name}']
            avails = tensor_output[f'vector_set_map.availabilities.{feature_name}']
            vector_map_lanes = polyline_process(polylines, avails, traffic_light_state)

        elif feature_name == "CROSSWALK":
            # polylines = tensor_output[f'vector_set_map.coords.{feature_name}']
            # avails = tensor_output[f'vector_set_map.availabilities.{feature_name}']
            # vector_map_crosswalks = polyline_process(polylines, avails)
            vector_map_crosswalks = None
            
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
        # print(f'---437--- k: {k}, v.shape: {v.shape}')
        if k == 'map_crosswalks':
            continue
        tensor_data[k] = v.float().unsqueeze(0).to(device)
        print(f'---441--- k: {k}, v.shape: {tensor_data[k].shape}')

    return tensor_data


def sampled_past_timestamps_to_tensor(past_time_stamps: List[float]) -> torch.Tensor:
    """
    Converts a list of N past timestamps into a 1-d tensor of shape [N]. The field is the timestamp in uS.
    :param past_time_stamps: The time stamps to convert.
    :return: The converted tensor.
    """
    flat = [t for t in past_time_stamps]
    return torch.tensor(flat, dtype=torch.int64)


def get_lane_polylines(
    carla_scenario_input: CarlaScenarioInput, point: Point2D, radius: float
) -> Tuple[
    MapObjectPolylines, MapObjectPolylines, MapObjectPolylines, LaneSegmentLaneIDs
]:
    """
    Extract ids, baseline path polylines, and boundary polylines of neighbor lanes and lane connectors around ego vehicle.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :return:
        lanes_mid: extracted lane/lane connector baseline polylines.
        lanes_left: extracted lane/lane connector left boundary polylines.
        lanes_right: extracted lane/lane connector right boundary polylines.
        lane_ids: ids of lanes/lane connector associated polylines were extracted from.
    """
    lanes_mid: List[List[Point2D]] = (
        []
    )  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_left: List[List[Point2D]] = (
        []
    )  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_right: List[List[Point2D]] = (
        []
    )  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lane_ids: List[str] = []  # shape: [num_lanes]
    # layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    # layers = map_api.get_proximal_map_objects(point, radius, layer_names)

    # map_objects: List[MapObject] = []

    # for layer_name in layer_names:
    #     map_objects += layers[layer_name]
    # # sort by distance to query point
    # map_objects.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))

    # for map_obj in map_objects:
    #     # center lane
    #     baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.baseline_path.discrete_path]
    #     lanes_mid.append(baseline_path_polyline)

    #     # boundaries
    #     lanes_left.append([Point2D(node.x, node.y) for node in map_obj.left_boundary.discrete_path])
    #     lanes_right.append([Point2D(node.x, node.y) for node in map_obj.right_boundary.discrete_path])

    #     # lane ids
    #     lane_ids.append(map_obj.id)

    # center lane
    baseline_path_polyline = [
        Point2D(node[0], node[1])
        for lane in carla_scenario_input.map_lanes
        for node in lane
    ]
    lanes_mid.append(baseline_path_polyline)

    # 根据车道中心线的点序列，和车道宽度，计算左右边界的点序列，注意左右边界与中心线是平行的
    lanes_width = 5.0
    for i, node in enumerate(baseline_path_polyline):
        if i == 0:
            # 计算斜率
            dir = np.array(
                [baseline_path_polyline[i + 1].x, baseline_path_polyline[i + 1].y]
            ) - np.array([node.x, node.y])
        else:
            # 计算斜率
            dir = np.array([node.x, node.y]) - np.array(
                [baseline_path_polyline[i - 1].x, baseline_path_polyline[i - 1].y]
            )

        dir = dir / np.linalg.norm(dir)
        # 计算左边界点，dir 逆时针旋转90度，然后乘以车道宽度的一半，再加到中心线点上
        dir = np.array([-dir[1], dir[0]])
        dir = dir * lanes_width / 2
        left_node = np.array([node.x, node.y]) + dir
        lanes_left.append([Point2D(left_node[0], left_node[1])])
        # 计算右边界点，dir 顺时针旋转90度，然后乘以车道宽度的一半，再加到中心线点上
        dir = np.array([dir[1], -dir[0]])
        dir = dir * lanes_width / 2
        right_node = np.array([node.x, node.y]) + dir
        lanes_right.append([Point2D(right_node[0], right_node[1])])
        lane_ids.append(i)

    return (
        MapObjectPolylines(lanes_mid),
        MapObjectPolylines(lanes_left),
        MapObjectPolylines(lanes_right),
        LaneSegmentLaneIDs(lane_ids),
    )


def get_traffic_light_encoding(
    lane_seg_ids: LaneSegmentLaneIDs, traffic_light_data: List[TrafficLightStatusData]
) -> LaneSegmentTrafficLightData:
    """
    Encode the lane segments with traffic light data.
    :param lane_seg_ids: The lane_segment ids [num_lane_segment].
    :param traffic_light_data: A list of all available data at the current time step.
    :returns: Encoded traffic light data per segment.
    """
    # Initialize with all segment labels with UNKNOWN status
    traffic_light_encoding = np.full(
        (len(lane_seg_ids.lane_ids), len(TrafficLightStatusType)),
        LaneSegmentTrafficLightData.encode(TrafficLightStatusType.UNKNOWN),
    )
    # print(f'------ traffic_light_data: {traffic_light_data}')

    # Extract ids of red and green lane connectors
    green_lane_connectors = [
        str(data.lane_connector_id)
        for data in traffic_light_data
        if data.status == TrafficLightStatusType.GREEN
    ]
    red_lane_connectors = [
        str(data.lane_connector_id)
        for data in traffic_light_data
        if data.status == TrafficLightStatusType.RED
    ]

    # print(f'------ green_lane_connectors: {green_lane_connectors}')
    # print(f'------ lane_seg_ids.lane_ids: {lane_seg_ids.lane_ids}')

    # Assign segments with corresponding traffic light status
    for tl_id in green_lane_connectors:
        indices = np.argwhere(np.array(lane_seg_ids.lane_ids) == int(tl_id))
        # print(f'------ indices: {indices}, tl_id: {tl_id}')
        traffic_light_encoding[indices] = LaneSegmentTrafficLightData.encode(
            TrafficLightStatusType.GREEN
        )

    for tl_id in red_lane_connectors:
        indices = np.argwhere(np.array(lane_seg_ids.lane_ids) == int(tl_id))
        traffic_light_encoding[indices] = LaneSegmentTrafficLightData.encode(
            TrafficLightStatusType.RED
        )
    # print(f'------ traffic_light_encoding: {traffic_light_encoding}')
    return LaneSegmentTrafficLightData(list(map(tuple, traffic_light_encoding)))  # type: ignore


def get_route_lane_polylines_from_roadblock_ids(
    carla_scenario_input: CarlaScenarioInput, point: Point2D, radius: float
) -> MapObjectPolylines:
    """
    Extract route polylines from map for route specified by list of roadblock ids. Route is represented as collection of
        baseline polylines of all children lane/lane connectors or roadblock/roadblock connectors encompassing route.
    :param map_api: map to perform extraction on.
    :param point: [m] x, y coordinates in global frame.
    :param radius: [m] floating number about extraction query range.
    :param route_roadblock_ids: ids of roadblocks/roadblock connectors specifying route.
    :return: A route as sequence of lane/lane connector polylines.
    """
    route_lane_polylines: List[List[Point2D]] = (
        []
    )  # shape: [num_lanes, num_points_per_lane (variable), 2]

    route_lane_polylines = [
        [
            Point2D(node[0], node[1])
            for lane in carla_scenario_input.route_lines
            for node in lane
        ]
    ]

    return MapObjectPolylines(route_lane_polylines)

# def get_map_object_polygons(
#     map_api: CarlaScenarioInput, point: Point2D, radius: float, layer_name: SemanticMapLayer
# ) -> MapObjectPolylines:
#     """
#     Extract polygons of neighbor map object around ego vehicle for specified semantic layers.
#     :param map_api: map to perform extraction on.
#     :param point: [m] x, y coordinates in global frame.
#     :param radius: [m] floating number about extraction query range.
#     :param layer_name: semantic layer to query.
#     :return extracted map object polygons.
#     """
#     map_objects = map_api.get_proximal_map_objects(point, radius, [layer_name])[layer_name]
#     # sort by distance to query point
#     map_objects.sort(key=lambda map_obj: get_distance_between_map_object_and_point(point, map_obj))
#     polygons = [extract_polygon_from_map_object(map_obj) for map_obj in map_objects]

#     return MapObjectPolylines(polygons)

def get_neighbor_vector_set_map(
    map_features: List[str],
    origin_point: Point2D,
    radius: float,
    carla_scenario_input: CarlaScenarioInput,
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
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(
            carla_scenario_input,
            origin_point, radius)

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = MapObjectPolylines(lanes_mid.polylines)
        
        
        # TODO(fanyu): 临时构造一个交通灯数据，用于测试
        traffic_light_status_data: List[TrafficLightStatusData] = []
        tl_num = 8
        for i in range(len(lane_ids.lane_ids)):
            traffic_light_status_data.append(TrafficLightStatusData(
                status = TrafficLightStatusType.GREEN,
                lane_connector_id = lane_ids.lane_ids[i], # 最好与 lane id 一致
                timestamp = time.time()*1e6 # 时间戳，单位微秒
            ))
        

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
        route_polylines = get_route_lane_polylines_from_roadblock_ids(carla_scenario_input, origin_point, radius)
        coords[VectorFeatureLayer.ROUTE_LANES.name] = route_polylines

    # # extract generic map objects
    # for feature_layer in feature_layers:
    #     if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
    #         polygons = get_map_object_polygons(
    #             carla_scenario_input, origin_point, radius, VectorFeatureLayerMapping.semantic_map_layer(feature_layer)
    #         )
    #         coords[feature_layer.name] = polygons
    
    
    # print(f'---695---coords[VectorFeatureLayer.LANE.name] = {coords[VectorFeatureLayer.LANE.name]}')
    # print(f'---696---traffic_light_data[VectorFeatureLayer.LANE.name] = {traffic_light_data[VectorFeatureLayer.LANE.name]}')
    
    print(f'---698---lane size: {len(coords[VectorFeatureLayer.LANE.name].polylines[0])}')
    print(f'---699---traffic_light_data[VectorFeatureLayer.LANE.name] size: {len(traffic_light_data[VectorFeatureLayer.LANE.name].traffic_lights)}')
    return coords, traffic_light_data


def sampled_past_ego_states_to_tensor(past_ego_states: Deque[EgoState]) -> torch.Tensor:
    """
    Converts a list of N ego states into a N x 7 tensor. The 7 fields are as defined in `EgoInternalIndex`
    :param past_ego_states: The ego states to convert.
    :return: The converted tensor.
    """
    print(f"------ size of past_ego_states: {len(past_ego_states)}")
    output = torch.zeros(
        (len(past_ego_states), EgoInternalIndex.dim()), dtype=torch.float32
    )
    for i in range(0, len(past_ego_states), 1):
        output[i, EgoInternalIndex.x()] = past_ego_states[i].x
        output[i, EgoInternalIndex.y()] = past_ego_states[i].y
        output[i, EgoInternalIndex.heading()] = past_ego_states[i].heading
        output[i, EgoInternalIndex.vx()] = past_ego_states[i].vx
        output[i, EgoInternalIndex.vy()] = past_ego_states[i].vy
        output[i, EgoInternalIndex.ax()] = past_ego_states[i].ax
        output[i, EgoInternalIndex.ay()] = past_ego_states[i].ay

    return output


def get_tracted_objects_of_types(past_tracked_objects: dict, object_types: list):
    ret_types = []
    for k, v in past_tracked_objects.items():
        if v.type and v.type != "" and v.type in object_types:
            ret_types.append(v.type)
    return ret_types


def filter_agents_tensor(agents_tensor, reverse=False):
    filtered_agents = []
    for agent_tensor in agents_tensor:
        if agent_tensor.shape[0] > 0:
            filtered_agents.append(agent_tensor)

    if reverse:
        filtered_agents.reverse()

    return filtered_agents


def pad_agent_states(
    agent_trajectories: List[torch.Tensor], reverse: bool
) -> List[torch.Tensor]:
    """
    Pads the agent states with the most recent available states. The order of the agents is also
    preserved. Note: only agents that appear in the current time step will be computed for. Agents appearing in the
    future or past will be discarded.

     t1      t2           t1      t2
    |a1,t1| |a1,t2|  pad |a1,t1| |a1,t2|
    |a2,t1| |a3,t2|  ->  |a2,t1| |a2,t1| (padded with agent 2 state at t1)
    |a3,t1| |     |      |a3,t1| |a3,t2|


    If reverse is True, the padding direction will start from the end of the trajectory towards the start

     tN-1    tN             tN-1    tN
    |a1,tN-1| |a1,tN|  pad |a1,tN-1| |a1,tN|
    |a2,tN  | |a2,tN|  <-  |a3,tN-1| |a2,tN| (padded with agent 2 state at tN)
    |a3,tN-1| |a3,tN|      |       | |a3,tN|

    :param agent_trajectories: agent trajectories [num_frames, num_agents, AgentInternalIndex.dim()], corresponding to the AgentInternalIndex schema.
    :param reverse: if True, the padding direction will start from the end of the list instead
    :return: A trajectory of extracted states
    """
    for traj in agent_trajectories:
        _validate_agent_internal_shape(traj)

    track_id_idx = AgentInternalIndex.track_token()
    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    key_frame = agent_trajectories[0]

    id_row_mapping: Dict[int, int] = {}
    for idx, val in enumerate(key_frame[:, track_id_idx]):
        id_row_mapping[int(val.item())] = idx

    current_state = torch.zeros((key_frame.shape[0], key_frame.shape[1]), dtype=torch.float32)
    for idx in range(len(agent_trajectories)):
        frame = agent_trajectories[idx]

        # Update current frame
        for row_idx in range(frame.shape[0]):
            mapped_row: int = id_row_mapping[int(frame[row_idx, track_id_idx].item())]
            current_state[mapped_row, :] = frame[row_idx, :]

        # Save current state
        agent_trajectories[idx] = torch.clone(current_state)

    if reverse:
        agent_trajectories = agent_trajectories[::-1]

    return agent_trajectories

def extract_agent_tensor(i: int, past_tracked_objects: dict, track_token_ids: dict, object_types:list):
    # agent_types = get_tracted_objects_of_types(past_tracked_objects, object_types)
    agent_types = []
    output = torch.zeros((len(past_tracked_objects), AgentInternalIndex.dim()), dtype=torch.float32)
    max_agent_id = len(track_token_ids)
    # print(f'--- 831 --- past_tracked_objects.size: {len(past_tracked_objects)}')
    # for idx, agent in enumerate(past_tracked_objects):
    idx = 0
    for key, agent in past_tracked_objects.items():
        # print(f'idx: {idx}, agent.size: {len(agent)}')
        if i >= len(agent):
            break
        if agent[i].id not in track_token_ids:
            track_token_ids[agent[i].id] = max_agent_id
            max_agent_id += 1
        track_token_int = track_token_ids[agent[i].id]

        output[idx, AgentInternalIndex.track_token()] = float(track_token_int)
        output[idx, AgentInternalIndex.vx()] = agent[i].vx
        output[idx, AgentInternalIndex.vy()] = agent[i].vy
        output[idx, AgentInternalIndex.heading()] = agent[i].heading
        output[idx, AgentInternalIndex.width()] = agent[i].width
        output[idx, AgentInternalIndex.length()] = agent[i].length
        output[idx, AgentInternalIndex.x()] = agent[i].x
        output[idx, AgentInternalIndex.y()] = agent[i].y
        agent_types.append(agent[i].type)
        idx += 1

    return output, track_token_ids, agent_types


def sampled_tracked_objects_to_tensor_list(past_tracked_objects: dict):
    object_types = [
        TrackedObjectType.VEHICLE,
        TrackedObjectType.PEDESTRIAN,
        TrackedObjectType.BICYCLE,
    ]
    output = []
    output_types = []
    track_token_ids = {}

    # for i in range(len(past_tracked_objects)):
    rem_dim = 22
    for i in range(rem_dim):
        # for k, agent_states in past_tracked_objects.items():
        tensorized, track_token_ids, agent_types = extract_agent_tensor(i, past_tracked_objects, track_token_ids, object_types)
        output.append(tensorized)
        # print(f'--- 871 --- agent type: {agent_types}')
        # TODO(fanyu): fill without any reason, just for test.
        agent_types = [object_types[0] for _ in range(21) ]
        output_types.append(agent_types)
    # print(f'---876--- output types: {output_types}')

    return output, output_types


def global_velocity_to_local(velocity, anchor_heading):
    velocity_x = velocity[:, 0] * torch.cos(anchor_heading) + velocity[
        :, 1
    ] * torch.sin(anchor_heading)
    velocity_y = velocity[:, 1] * torch.cos(anchor_heading) - velocity[
        :, 0
    ] * torch.sin(anchor_heading)

    return torch.stack([velocity_x, velocity_y], dim=-1)


def _validate_state_se2_tensor_batch_shape(tensor: torch.Tensor) -> None:
    """
    Validates that a tensor is of the proper shape for a batch of tensorized StateSE2.
    :param tensor: The tensor to validate.
    """
    expected_feature_dim = 3
    if len(tensor.shape) == 2 and tensor.shape[1] == expected_feature_dim:
        return

    raise ValueError(f"Improper se2 tensor batch shape: {tensor.shape}")


def state_se2_tensor_to_transform_matrix_batch(
    input_data: torch.Tensor, precision: Optional[torch.dtype] = None
) -> torch.Tensor:
    """
    Transforms a tensor of states of the form Nx3 (x, y, heading) into a Nx3x3 transform tensor.
    :param input_data: the input data as a Nx3 tensor.
    :param precision: The precision with which to create the output tensor. If None, then it will be inferred from the input tensor.
    :return: The output Nx3x3 batch transformation tensor.
    """
    _validate_state_se2_tensor_batch_shape(input_data)

    if precision is None:
        precision = input_data.dtype

    # Transform the incoming coordinates so transformation can be done with a simple matrix multiply.
    #
    # [x1, y1, phi1]  => [x1, y1, cos1, sin1, 1]
    # [x2, y2, phi2]     [x2, y2, cos2, sin2, 1]
    # ...          ...
    # [xn, yn, phiN]     [xn, yn, cosN, sinN, 1]
    processed_input = torch.column_stack(
        (
            input_data[:, 0],
            input_data[:, 1],
            torch.cos(input_data[:, 2]),
            torch.sin(input_data[:, 2]),
            torch.ones_like(input_data[:, 0], dtype=precision),
        )
    )

    # See below for reshaping example
    reshaping_tensor = torch.tensor(
        [
            [0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, -1, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1],
        ],
        dtype=precision,
        device=input_data.device,
    )
    # Builds the transform matrix
    # First computes the components of each transform as rows of a Nx9 tensor, and then reshapes to a Nx3x3 tensor
    # Below is outlined how the Nx9 representation looks like (s1 and c1 are cos1 and sin1)
    # [x1, y1, c1, s1, 1]  => [c1, -s1, x1, s1, c1, y1, 0, 0, 1]  =>  [[c1, -s1, x1], [s1, c1, y1], [0, 0, 1]]
    # [x2, y2, c2, s2, 1]     [c2, -s2, x2, s2, c2, y2, 0, 0, 1]  =>  [[c2, -s2, x2], [s2, c2, y2], [0, 0, 1]]
    # ...          ...
    # [xn, yn, cN, sN, 1]     [cN, -sN, xN, sN, cN, yN, 0, 0, 1]
    return (processed_input @ reshaping_tensor).reshape(-1, 3, 3)


def _validate_transform_matrix_batch_shape(tensor: torch.Tensor) -> None:
    """
    Validates that a tensor has the proper shape for a 3x3 transform matrix.
    :param tensor: the tensor to validate.
    """
    if len(tensor.shape) == 3 and tensor.shape[1] == 3 and tensor.shape[2] == 3:
        return

    raise ValueError(f"Improper transform matrix shape: {tensor.shape}")


def transform_matrix_to_state_se2_tensor_batch(
    input_data: torch.Tensor,
) -> torch.Tensor:
    """
    Converts a Nx3x3 batch transformation matrix into a Nx3 tensor of [x, y, heading] rows.
    :param input_data: The 3x3 transformation matrix.
    :return: The converted tensor.
    """
    _validate_transform_matrix_batch_shape(input_data)

    # Picks the entries, the third column will be overwritten with the headings [x, y, _]
    first_columns = input_data[:, :, 0].reshape(-1, 3)
    angles = torch.atan2(first_columns[:, 1], first_columns[:, 0])

    result = input_data[:, :, 2]
    result[:, 2] = angles

    return result


def global_state_se2_tensor_to_local(
    global_states: torch.Tensor,
    local_state: torch.Tensor,
    precision: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Transforms the StateSE2 in tensor from to the frame of reference in local_frame.

    :param global_states: A tensor of Nx3, where the columns are [x, y, heading].
    :param local_state: A tensor of [x, y, h] of the frame to which to transform.
    :param precision: The precision with which to allocate the intermediate tensors. If None, then it will be inferred from the input precisions.
    :return: The transformed coordinates.
    """
    _validate_state_se2_tensor_shape(global_states, expected_first_dim=2)
    _validate_state_se2_tensor_shape(local_state, expected_first_dim=1)

    if precision is None:
        if global_states.dtype != local_state.dtype:
            raise ValueError(
                "Mixed datatypes provided to coordinates_to_local_frame without precision specifier."
            )
        precision = global_states.dtype

    local_xform = state_se2_tensor_to_transform_matrix(local_state, precision=precision)
    local_xform_inv = torch.linalg.inv(local_xform)

    transforms = state_se2_tensor_to_transform_matrix_batch(
        global_states, precision=precision
    )

    transforms = torch.matmul(local_xform_inv, transforms)

    output = transform_matrix_to_state_se2_tensor_batch(transforms)

    return output


def convert_absolute_quantities_to_relative(agent_state, ego_state, agent_type="ego"):
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
        # print(f'---1054--- agent_global_poses.shape: {agent_global_poses.shape}')
        agent_global_velocities = agent_state[:, [AgentInternalIndex.vx(), AgentInternalIndex.vy()]]
        transformed_poses = global_state_se2_tensor_to_local(agent_global_poses, ego_pose, precision=torch.float64)
        transformed_velocities = global_velocity_to_local(agent_global_velocities, ego_pose[-1])
        agent_state[:, AgentInternalIndex.x()] = transformed_poses[:, 0].float()
        agent_state[:, AgentInternalIndex.y()] = transformed_poses[:, 1].float()
        agent_state[:, AgentInternalIndex.heading()] = transformed_poses[:, 2].float()
        agent_state[:, AgentInternalIndex.vx()] = transformed_velocities[:, 0].float()
        agent_state[:, AgentInternalIndex.vy()] = transformed_velocities[:, 1].float()

    return agent_state

def _validate_agent_internal_shape(feature: torch.Tensor) -> None:
    """
    Validates the shape of the provided tensor if it's expected to be an AgentInternal.
    :param feature: the tensor to validate.
    """
    if len(feature.shape) != 2 or feature.shape[1] != AgentInternalIndex.dim():
        raise ValueError(f"Improper agent internal shape: {feature.shape}")
    
def unwrap(angles: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    This unwraps a signal p by changing elements which have an absolute difference from their
    predecessor of more than Pi to their period-complementary values.
    It is meant to mimic numpy.unwrap (https://numpy.org/doc/stable/reference/generated/numpy.unwrap.html)
    :param angles: The tensor to unwrap.
    :param dim: Axis where the unwrap operation is performed.
    :return: Unwrapped tensor.
    """
    pi = torch.tensor(math.pi, dtype=torch.float64)
    angle_diff = torch.diff(angles, dim=dim)

    # Insert a single 0 at the front of the dimension corresponding to 'dim'.
    # This could be  written a bit more succintly using pure python, but
    #   torchscript doesn't support many of the constructs:
    #
    # * Itertools.chain() can't be scripted.
    # * Multiple generator expressions can't be scripted.
    # * sum() gets translated to aten::sum, which doesn't support Tuple.
    nn_functional_pad_args = [(0, 0) for _ in range(len(angles.shape))]
    nn_functional_pad_args[dim] = (1, 0)

    # Counter-intuitively, torch.nn.functional.pad reverses the indexes.
    #   e.g.:
    # >> x = torch.zeros((3, 3, 3), dtype=torch.float32)
    # >> torch.nn.functional.pad(x, (0, 0, 0, 0, 1, 0)).shape
    # torch.Size([4, 3, 3])
    # >> torch.nn.functional.pad(x, (1, 0, 0, 0, 0, 0)).shape
    # torch.Size([3, 3, 4])
    pad_arg: List[int] = []
    for value in nn_functional_pad_args[::-1]:
        pad_arg.append(value[0])
        pad_arg.append(value[1])

    dphi = torch.nn.functional.pad(angle_diff, pad_arg)

    dphi_m = ((dphi + pi) % (2.0 * pi)) - pi
    dphi_m[(dphi_m == -pi) & (dphi > 0)] = pi
    phi_adj = dphi_m - dphi
    phi_adj[dphi.abs() < pi] = 0
    return angles + phi_adj.cumsum(dim)


def _torch_savgol_filter(
    y: torch.Tensor, window_length: int, poly_order: int, deriv_order: int, delta: float
) -> torch.Tensor:
    """
    Perform Savinsky Golay filtering on the given tensor.
    This is adapted from the scipy method `scipy.signal.savgol_filter`
        However, it currently only works with window_length of 3.
    :param y: The tensor to filter. Should be of dimension 2.
    :param window_length: The window length to use.
        Currently provided as a parameter, but for now must be 3.
    :param poly_order: The polynomial order to use.
    :param deriv_order: The order of derivitave to use.
    :coefficients: The Savinsky Golay coefficients to use.
    :return: The filtered tensor.
    """
    # TODO: port np.polyfit and remove this restriction
    if window_length != 3:
        raise ValueError(
            "This method has unexpected edge behavior for window_length != 3."
        )

    if len(y.shape) != 2:
        raise ValueError(
            f"Unexpected input tensor shape to _torch_savgol_filter(): {y.shape}"
        )

    # Compute the coefficients in conv format
    halflen, rem = divmod(window_length, 2)
    if rem == 0:
        pos = halflen - 0.5
    else:
        pos = float(halflen)

    # For convolution in scipy, there is a horizontal flip of x
    # But, the weight ordering between torch.nn.functional.conv1d and
    #   scipy.ndimage.convolv1d is also flipped.
    #
    # (that is, given an input tensor x and a conv filter of [1, 0, -1],
    #   the output of scipy.ndimage.conv1d will be [a, b, c, d, ...]
    #   and the output of torch.nn.functional.conv1d will be [-a, -b, -c, -d])
    #
    # So they cancel out.
    x = torch.arange(-pos, window_length - pos, dtype=torch.float64)
    order = torch.arange(poly_order + 1).reshape(-1, 1)

    yy = torch.zeros(poly_order + 1, dtype=torch.float64)
    A = x**order
    yy[deriv_order] = math.factorial(deriv_order) / (delta**deriv_order)

    coeffs, _, _, _ = torch.linalg.lstsq(A, yy)

    # Perform the filtering
    y_in = y.unsqueeze(1)
    coeffs_in = coeffs.reshape(1, 1, -1)
    result = torch.nn.functional.conv1d(y_in, coeffs_in, padding="same").reshape(
        y.shape
    )

    # Fix the edges
    # This only works for window_length == 3
    # A more general solution would require porting np.polyfit
    n = result.shape[1]
    result[:, 0] = y[:, 1] - y[:, 0]
    result[:, n - 1] = y[:, n - 1] - y[:, n - 2]

    return result


def _validate_approximate_derivatives_shapes(y: torch.Tensor, x: torch.Tensor) -> None:
    """
    Validates that the shapes for approximate_derivatives_tensor are correct.
    :param y: The Y input.
    :param x: The X input.
    """
    print(
        f"y.shape: {y.shape}, x.shape: {x.shape}, y.shape[1]: {y.shape[1]}, x.shape[0]: {x.shape[0]}"
    )
    print(f"len(y.shape): {len(y.shape)}, len(x.shape): {len(x.shape)})")
    if len(y.shape) == 2 and len(x.shape) == 1 and y.shape[1] == x.shape[0]:
        return

    raise ValueError(
        f"Unexpected tensor shapes in approximate_derivatives_tensor: y.shape = {y.shape}, x.shape = {x.shape}"
    )


def approximate_derivatives_tensor(
    y: torch.Tensor,
    x: torch.Tensor,
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
) -> torch.Tensor:
    """
    Given a time series [y], and [x], approximate [dy/dx].
    :param y: Input tensor to filter.
    :param x: Time dimension for tensor to filter.
    :param window_length: The size of the window to use.
    :param poly_order: The order of polymonial to use when filtering.
    :deriv_order: The order of derivitave to use when filtering.
    :return: The differentiated tensor.
    """
    _validate_approximate_derivatives_shapes(y, x)

    window_length = min(window_length, x.shape[0])

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = torch.diff(x)
    # print(f"############ dx: {dx}")
    # print(f'------ x.shape: {x.shape}, dx.shape: {dx.shape}')
    min_increase = float(torch.min(dx).item())
    if min_increase <= 0:
        raise RuntimeError('dx is not monotonically increasing!')
        # print("[TODO(fanyu)] --- WARNING: dx is not monotonically increasing._---")

    dx = dx.mean()

    derivative: torch.Tensor = _torch_savgol_filter(
        y,
        poly_order=poly_order,
        window_length=window_length,
        deriv_order=deriv_order,
        delta=dx,
    )

    return derivative


def compute_yaw_rate_from_state_tensors(
    agent_states: List[torch.Tensor],
    time_stamps: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the yaw rate of all agents over the trajectory from heading
    :param agent_states_horizon: Agent trajectories [num_frames, num_agent, AgentsInternalBuffer.dim()]
    :param time_stamps: The time stamps of each frame.
    :return: <torch.Tensor: num_frames, num_agents> of yaw rates
    """
    # Convert time_stamps to seconds
    # Shift the min timestamp to 0 to avoid loss of precision
    if len(time_stamps.shape) != 1:
        raise ValueError(f"Unexpected timestamps shape: {time_stamps.shape}")
    time_stamps_s = (time_stamps - int(torch.min(time_stamps).item())).double() * 1e-6

    yaws: List[torch.Tensor] = []
    print(f"---1264--- agent_states.size: {len(agent_states)}")
    for i in range(len(agent_states)):
        _validate_agent_internal_shape(agent_states[i])
        yaws.append(agent_states[i][:, AgentInternalIndex.heading()].squeeze().double())

    # Convert to agent x frame
    yaws_tensor = torch.vstack(yaws)
    yaws_tensor = yaws_tensor.transpose(0, 1)
    # Remove [-pi, pi] yaw bounds to make the signal smooth
    yaws_tensor = unwrap(yaws_tensor, dim=-1)

    yaw_rate_horizon = approximate_derivatives_tensor(
        yaws_tensor, time_stamps_s, window_length=3
    )

    print(
        f"------ yaw_rate_horizon.shape: {yaw_rate_horizon.shape}, yaws_tensor.shape: {yaws_tensor.shape}, time_stamps_s.shape: {time_stamps_s.shape}"
    )

    # Convert back to frame x agent
    return yaw_rate_horizon.transpose(0, 1)

def pack_agents_tensor(padded_agents_tensors: List[torch.Tensor], yaw_rates: torch.Tensor) -> torch.Tensor:
    """
    Combines the local padded agents states and the computed yaw rates into the final output feature tensor.
    :param padded_agents_tensors: The padded agent states for each timestamp.
        Each tensor is of shape <num_agents, len(AgentInternalIndex)> and conforms to the AgentInternalIndex schema.
    :param yaw_rates: The computed yaw rates. The tensor is of shape <num_timestamps, agent>
    :return: The final feature, a tensor of shape [timestamp, num_agents, len(AgentsFeatureIndex)] conforming to the AgentFeatureIndex Schema
    """
    if yaw_rates.shape != (len(padded_agents_tensors), padded_agents_tensors[0].shape[0]):
        print(f'------ yaw_rates.shape: {yaw_rates.shape}, padded_agents_tensors[0].shape: {padded_agents_tensors[0].shape}')
        
        raise ValueError(f"Unexpected yaw_rates tensor shape: {yaw_rates.shape}")

    agents_tensor = torch.zeros(
        (len(padded_agents_tensors), padded_agents_tensors[0].shape[0], AgentFeatureIndex.dim())
    )

    for i in range(len(padded_agents_tensors)):
        _validate_agent_internal_shape(padded_agents_tensors[i])
        agents_tensor[i, :, AgentFeatureIndex.x()] = padded_agents_tensors[i][:, AgentInternalIndex.x()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.y()] = padded_agents_tensors[i][:, AgentInternalIndex.y()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.heading()] = padded_agents_tensors[i][
            :, AgentInternalIndex.heading()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.vx()] = padded_agents_tensors[i][:, AgentInternalIndex.vx()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.vy()] = padded_agents_tensors[i][:, AgentInternalIndex.vy()].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.yaw_rate()] = yaw_rates[i, :].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.width()] = padded_agents_tensors[i][
            :, AgentInternalIndex.width()
        ].squeeze()
        agents_tensor[i, :, AgentFeatureIndex.length()] = padded_agents_tensors[i][
            :, AgentInternalIndex.length()
        ].squeeze()

    return agents_tensor
def agent_past_process(past_ego_states, past_time_stamps, past_tracked_objects, tracked_objects_types, num_agents):
    # agents_states_dim = Agents.agents_states_dim()
    agents_states_dim = 8
    ego_history = past_ego_states
    time_stamps = past_time_stamps
    agents = past_tracked_objects

    # print(f"------ time stamps: {time_stamps}")

    anchor_ego_state = ego_history[-1, :].squeeze().clone()
    ego_tensor = convert_absolute_quantities_to_relative(ego_history, anchor_ego_state)
    agent_history = filter_agents_tensor(agents, reverse=True)
    print(f"---1326--- agent_history size = {len(agent_history)}")
    agent_types = tracked_objects_types[-1]

    if agent_history[-1].shape[0] == 0:
        # Return zero tensor when there are no agents in the scene
        agents_tensor = torch.zeros((len(agent_history), 0, agents_states_dim)).float()
    else:
        local_coords_agent_states = []
        padded_agent_states = pad_agent_states(agent_history, reverse=True)
        for i in range(len(padded_agent_states)):
            # print(f'------ agent_state.shape: {agent_state.shape}')
            if padded_agent_states[i].shape[0] < 21:
                # 将剩余的后面的数据补全为 21 帧，填充的内容为最后一帧的数据
                state = torch.cat([padded_agent_states[i], padded_agent_states[i][-1].unsqueeze(0).repeat(21-padded_agent_states[i].shape[0], 1)], dim=0)
                padded_agent_states[i] = state
                # print(f'---1341--- agent_state.shape after repeat: {padded_agent_states[i].shape}')
            
            

        for agent_state in padded_agent_states:
            local_coords_agent_states.append(convert_absolute_quantities_to_relative(agent_state, anchor_ego_state, 'agent'))
        print(f"------- local_coords_agent_states length = {len(local_coords_agent_states)}")
        print(f"------- local_coords_agent_states[0].shape = {local_coords_agent_states[0].shape}")
    
        # Calculate yaw rate
        yaw_rate_horizon = compute_yaw_rate_from_state_tensors(padded_agent_states, time_stamps)
        print(f"--- yaw_rate_horizon.shape = {yaw_rate_horizon.shape}")
        print(f"------- padded_agent_states length = {len(padded_agent_states)}, type = {type(padded_agent_states)}")
        print(f"------- padded_agent_states[0].shape = {padded_agent_states[0].shape}")
        agents_tensor = pack_agents_tensor(local_coords_agent_states, yaw_rate_horizon)
        print(f"--- agents_tensor.shape = {agents_tensor.shape}")

    agents = torch.zeros(
        (num_agents, agents_tensor.shape[0], agents_tensor.shape[-1] + 3),
        dtype=torch.float32,
    )

    # sort agents according to distance to ego
    # 根据距离ego车辆的距离进行排序
    print(f"--- agents_tensor.shape = {agents_tensor.shape}")
    distance_to_ego = torch.norm(agents_tensor[-1, :, :2], dim=-1)
    indices = list(torch.argsort(distance_to_ego).numpy())[:num_agents]

    # fill agent features into the array
    added_agents = 0
    print(f" ---1371--- indices = {indices}")
    for i in indices:
        if added_agents >= num_agents:
            break

        if agents_tensor[-1, i, 0] < -6.0:
            continue

        agents[added_agents, :, : agents_tensor.shape[-1]] = agents_tensor[:, i, : agents_tensor.shape[-1]]

        # print(f"---1381--- agent types: {agent_types}")
        print(f"---1382--- i = {i}, distance_to_ego = {distance_to_ego[i]}, agent_type = {agent_types[i]}, added_agents = {added_agents}")

        if agent_types[i] == TrackedObjectType.VEHICLE:
            agents[added_agents, :, agents_tensor.shape[-1] :] = torch.tensor([1, 0, 0])
        elif agent_types[i] == TrackedObjectType.PEDESTRIAN:
            agents[added_agents, :, agents_tensor.shape[-1] :] = torch.tensor([0, 1, 0])
        else:
            agents[added_agents, :, agents_tensor.shape[-1] :] = torch.tensor([0, 0, 1])

        added_agents += 1

    return ego_tensor, agents


def create_feature_from_carla(carla_scenario_input: CarlaScenarioInput, device):
    num_agents = 20
    past_time_steps = 22
    map_features = [
        "LANE",
        "ROUTE_LANES",
        "CROSSWALK",
    ]  # name of map features to be extracted.
    max_elements = {
        "LANE": 40,
        "ROUTE_LANES": 10,
        "CROSSWALK": 5,
    }  # maximum number of elements to extract per feature layer.
    max_points = {
        "LANE": 50,
        "ROUTE_LANES": 50,
        "CROSSWALK": 30,
    }  # maximum number of points per feature to extract per feature layer.
    radius = 80  # [m] query radius scope relative to the current pose.
    interpolation_method = "linear"

    ego_states = carla_scenario_input.ego_states  # Past ego state including the current
    agents_map = (carla_scenario_input.agents_map)  # Past observations including the current
    ego_agent_past = sampled_past_ego_states_to_tensor(ego_states)
    past_tracked_objects_tensor_list, past_tracked_objects_types = (sampled_tracked_objects_to_tensor_list(agents_map))
    print(f"-------- ego_agent_past.size: {len(carla_scenario_input.ego_states)}")

    time_stamps_past = sampled_past_timestamps_to_tensor(
        [state.timestamp for state in carla_scenario_input.ego_states]
    )
    print(f"-------- time_stamps_past.shape: {time_stamps_past.shape}")
    ego_state = carla_scenario_input.ego_states[-1]
    ego_coords = Point2D(ego_state.x, ego_state.y)
    coords, traffic_light_data = get_neighbor_vector_set_map(map_features, ego_coords, radius, carla_scenario_input)

    ego_agent_past, neighbor_agents_past = agent_past_process(
        ego_agent_past, time_stamps_past, past_tracked_objects_tensor_list, past_tracked_objects_types, num_agents
    )

    vector_map = map_process(ego_state, coords, traffic_light_data, map_features, 
                             max_elements, max_points, interpolation_method)

    data = {"ego_agent_past": ego_agent_past[1:], 
            "neighbor_agents_past": neighbor_agents_past[:, 1:]}
    data.update(vector_map)
    data = convert_to_model_inputs(data, device)

    # data = {"ego_agent_past": ego_agent_past[1:], 
    #      "neighbor_agents_past": neighbor_agents_past[:, 1:]}
    return data


def _get_fixed_timesteps(
    state: EgoState, future_horizon: float, step_interval: float
) -> List[float]:
    """
    Get a fixed array of timesteps starting from a state's time.

    :param state: input state
    :param future_horizon: [s] future time horizon
    :param step_interval: [s] interval between steps in the array
    :return: constructed timestep list
    """
    timesteps = np.arange(0.0, future_horizon, step_interval) + step_interval
    # timesteps += state.timestamp
    timesteps += state.timestamp*1e-6

    return list(timesteps.tolist())


def matrix_from_pose(pose: EgoState) -> npt.NDArray[np.float64]:
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


def pose_from_matrix(transform_matrix: npt.NDArray[np.float32]) -> EgoState:
    """
    Converts a 3x3 transformation matrix to a 2D pose
    :param transform_matrix: 3x3 transformation matrix
    :return: 2D pose (x, y, yaw)
    """
    if transform_matrix.shape != (3, 3):
        raise RuntimeError(
            f"Expected a 3x3 transformation matrix, got {transform_matrix.shape}"
        )

    heading = np.arctan2(transform_matrix[1, 0], transform_matrix[0, 0])

    return EgoState(transform_matrix[0, 2], transform_matrix[1, 2], heading)


def relative_to_absolute_poses(
    origin_pose: EgoState, relative_poses: List[EgoState]
) -> List[EgoState]:
    """
    Converts a list of SE2 poses from relative to absolute coordinates using an origin pose.
    :param origin_pose: Reference origin pose
    :param relative_poses: list of relative poses to convert
    :return: list of converted absolute poses
    """
    relative_transforms: npt.NDArray[np.float64] = np.array(
        [matrix_from_pose(pose) for pose in relative_poses]
    )
    origin_transform = matrix_from_pose(origin_pose)
    absolute_transforms: npt.NDArray[np.float32] = (
        origin_transform @ relative_transforms
    )
    absolute_poses = [
        pose_from_matrix(transform_matrix) for transform_matrix in absolute_transforms
    ]

    return absolute_poses


def approximate_derivatives(
    y: npt.NDArray[np.float32],
    x: npt.NDArray[np.float32],
    window_length: int = 5,
    poly_order: int = 2,
    deriv_order: int = 1,
    axis: int = -1,
) -> npt.NDArray[np.float32]:
    """
    Given two equal-length sequences y and x, compute an approximation to the n-th
    derivative of some function interpolating the (x, y) data points, and return its
    values at the x's.  We assume the x's are increasing and equally-spaced.
    :param y: The dependent variable (say of length n)
    :param x: The independent variable (must have the same length n).  Must be strictly
        increasing and equally-spaced.
    :param window_length: The order (default 5) of the Savitsky-Golay filter used.
        (Ignored if the x's are not equally-spaced.)  Must be odd and at least 3
    :param poly_order: The degree (default 2) of the filter polynomial used.  Must
        be less than the window_length
    :param deriv_order: The order of derivative to compute (default 1)
    :param axis: The axis of the array x along which the filter is to be applied. Default is -1.
    :return Derivatives.
    """
    window_length = min(window_length, len(x))

    if not (poly_order < window_length):
        raise ValueError(f"{poly_order} < {window_length} does not hold!")

    dx = np.diff(x)
    if not (dx > 0).all():
        raise RuntimeError("dx is not monotonically increasing!")

    dx = dx.mean()
    derivative: npt.NDArray[np.float32] = savgol_filter(
        y,
        polyorder=poly_order,
        window_length=window_length,
        deriv=deriv_order,
        delta=dx,
        axis=axis,
    )
    return derivative


def _project_from_global_to_ego_centric_ds(
    ego_poses: npt.NDArray[np.float32], values: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    """
    Project value from the global xy frame to the ego centric ds frame.

    :param ego_poses: [x, y, heading] with size [planned steps, 3].
    :param values: values in global frame with size [planned steps, 2]
    :return: values projected onto the new frame with size [planned steps, 2]
    """
    print(f'---1570---ego_poses: {ego_poses}')
    print(f'---1571---values: {values}')
    headings = ego_poses[:, -1:]

    values_lon = values[:, :1] * np.cos(headings) + values[:, 1:2] * np.sin(headings)
    values_lat = values[:, :1] * np.sin(headings) - values[:, 1:2] * np.cos(headings)
    values = np.concatenate((values_lon, values_lat), axis=1)
    return values


def _get_velocity_and_acceleration(
    ego_poses: List[EgoState], ego_history: Deque[EgoState], timesteps: List[float]
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
    timesteps_past_current = [state.timestamp for state in ego_history]
    ego_poses_past_current: npt.NDArray[np.float32] = np.stack(
        # [np.array(state.rear_axle.serialize()) for state in ego_history]
        [np.array(state.searialize()) for state in ego_history]
    )

    # Planned
    dt = current_ego_state.timestamp - ego_history[-2].timestamp
    timesteps_current_planned: npt.NDArray[np.float32] = np.array(
        [current_ego_state.timestamp] + timesteps
    )
    ego_poses_current_planned: npt.NDArray[np.float32] = np.stack(
        [current_ego_state.searialize()]
        + [pose.searialize() for pose in ego_poses]
    )

    # Interpolation to have equal space for derivatives
    ego_poses_interpolate = interp1d(
        timesteps_current_planned,
        ego_poses_current_planned,
        axis=0,
        fill_value="extrapolate",
    )
    timesteps_current_planned_interp = np.arange(
        start=current_ego_state.timestamp, stop=timesteps[-1] + 1e-6, step=dt
    )
    ego_poses_current_planned_interp = ego_poses_interpolate(
        timesteps_current_planned_interp
    )

    # Combine past current and planned
    timesteps_past_current_planned = [
        *timesteps_past_current,
        *timesteps_current_planned_interp[1:],
    ]
    ego_poses_past_current_planned: npt.NDArray[np.float32] = np.concatenate(
        [ego_poses_past_current, ego_poses_current_planned_interp[1:]], axis=0
    )

    # Take derivatives
    ego_velocity_past_current_planned = approximate_derivatives(
        ego_poses_past_current_planned[:, :2], timesteps_past_current_planned, axis=0
    )
    ego_acceleration_past_current_planned = approximate_derivatives(
        ego_poses_past_current_planned[:, :2],
        timesteps_past_current_planned,
        axis=0,
        deriv_order=2,
    )

    # Only take the planned for output
    ego_velocity_planned_xy = ego_velocity_past_current_planned[ego_history_len:]
    ego_acceleration_planned_xy = ego_acceleration_past_current_planned[
        ego_history_len:
    ]

    # Projection
    ego_velocity_planned_ds = _project_from_global_to_ego_centric_ds(
        ego_poses_current_planned_interp[1:], ego_velocity_planned_xy
    )
    ego_acceleration_planned_ds = _project_from_global_to_ego_centric_ds(
        ego_poses_current_planned_interp[1:], ego_acceleration_planned_xy
    )

    # Interpolate back
    print(f"--- 1658 --- timesteps_past_current_planned: {timesteps_past_current_planned}")
    print(f"--- 1659 --- ego_velocity_planned_ds: {ego_velocity_planned_ds}")
    ego_velocity_interp_back = interp1d(
        timesteps_past_current_planned[ego_history_len:],
        ego_velocity_planned_ds,
        axis=0,
        fill_value="extrapolate",
    )
    ego_acceleration_interp_back = interp1d(
        timesteps_past_current_planned[ego_history_len:],
        ego_acceleration_planned_ds,
        axis=0,
        fill_value="extrapolate",
    )

    ego_velocity_planned_ds = ego_velocity_interp_back(timesteps)
    ego_acceleration_planned_ds = ego_acceleration_interp_back(timesteps)

    return ego_velocity_planned_ds, ego_acceleration_planned_ds


def _se2_vel_acc_to_ego_state(
    state: EgoState,
    velocity: npt.NDArray[np.float32],
    acceleration: npt.NDArray[np.float32],
    timestamp: float,
) -> EgoState:
    """
    Convert EgoState, velocity and acceleration to EgoState given a timestamp.

    :param state: input SE2 state
    :param velocity: [m/s] longitudinal velocity, lateral velocity
    :param acceleration: [m/s^2] longitudinal acceleration, lateral acceleration
    :param timestamp: [s] timestamp of state
    :return: output agent state
    """
    # return EgoState.build_from_rear_axle(
    #     rear_axle_pose=state,
    #     rear_axle_velocity_2d=StateVector2D(*velocity),
    #     rear_axle_acceleration_2d=StateVector2D(*acceleration),
    #     tire_steering_angle=0.0,
    #     time_point=TimePoint(int(timestamp * 1e6)),
    #     vehicle_parameters=vehicle,
    #     is_in_auto_mode=True,
    # )
    return EgoState(
        state.x,
        state.y,
        state.heading,
        velocity[0],
        velocity[1],
        acceleration[0],
        acceleration[1],
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
    relative_states = [
        EgoState(x=pose[0], y=pose[1], heading=pose[2]) for pose in poses
    ]
    absolute_states = relative_to_absolute_poses(ego_state, relative_states)
    velocities, accelerations = _get_velocity_and_acceleration(
        absolute_states, ego_history, timesteps
    )
    agent_states = [
        _se2_vel_acc_to_ego_state(state, velocity, acceleration, timestep)
        for state, velocity, acceleration, timestep in zip(
            absolute_states, velocities, accelerations, timesteps
        )
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
    states = _get_absolute_agent_states_from_numpy_poses(
        predicted_poses, ego_history, timesteps
    )

    if include_ego_state:
        states.insert(0, ego_state)

    return states
