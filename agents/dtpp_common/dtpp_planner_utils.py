# from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Tuple, Dict, Union
from scipy.interpolate import interp1d

# import torch

# import shapely.geometry as geom
import numpy as np

# import time
# import math
# import logging
from custom_format import *
logger = create_colored_logger(name=__name__)

import carla


from nuplan.common.actor_state.ego_state import EgoState
# from nuplan.planning.training.preprocessing.features.agents import Agents

from nuplan.common.actor_state.vehicle_parameters import VehicleParameters
# from nuplan.common.actor_state.tracked_objects import TrackedObjects
# from nuplan.common.actor_state.agent import Agent
# from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import Point2D, StateSE2, StateVector2D
# from nuplan.common.actor_state.scene_object import SceneObject, SceneObjectMetadata
from nuplan.planning.training.preprocessing.utils.agents_preprocessing import *
# from nuplan.common.maps.maps_datatypes import TrafficLightStatusData, TrafficLightStatusType, TrafficLightStatuses, Transform

# from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import (
#     LaneSegmentTrafficLightData,
#     VectorFeatureLayer,
#     MapObjectPolylines, MapObjectPolylines, MapObjectPolylines, LaneSegmentLaneIDs,
#     SemanticMapLayer,
#     get_traffic_light_encoding
# )


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
    logger.debug(f"--- dt = {dt}")
    dt = 0.1 if dt > 0.12 else dt
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

def get_rear_axle_world_coordinates(vehicle: carla.Actor) -> carla.Location:
        """
        计算车辆后轴中心在世界坐标系下的坐标。

        参数:
            vehicle (carla.Actor): 车辆 Actor 对象。
            cr (float): 后轴中心到车辆几何中心的纵向距离（沿车辆前进方向）。

        返回:
            carla.Location: 后轴中心的世界坐标。
        """
        # 获取车辆的 Transform 和 Bounding Box
        transform = vehicle.get_transform()
        bbox = vehicle.bounding_box

        # 车辆几何中心在局部坐标系中的位置（通常是 Bounding Box 的中心）
        geometric_center_local = bbox.location

        # 后轴中心在局部坐标系中的位置
        # 假设 cr 是后轴中心到几何中心的纵向距离（沿车辆前进方向）
        vehicle_param =  get_vehicle_params_from_actor(actor=vehicle)
        rear_axle_local = carla.Location(
            x=geometric_center_local.x - vehicle_param.cog_position_from_rear_axle,  # 沿车辆前进方向负方向
            y=geometric_center_local.y,
            z=geometric_center_local.z
        )

        # 将局部坐标转换为世界坐标
        rear_axle_world = transform.transform(rear_axle_local)

        return rear_axle_world