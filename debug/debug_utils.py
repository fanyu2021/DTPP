
import numpy as np
from agents.dtpp_common.dtpp_planner_utils import get_vehicle_params_from_actor, get_rear_axle_world_coordinates



def transform_to_ego_frame(path, actor):
        transform = actor.get_transform()
        x, y, heading = transform.location.x, transform.location.y, np.deg2rad(transform.rotation.yaw)
        x = path[:, 0] - x
        y = path[:, 1] - y

        # x_e = x * np.cos(-heading) - y * np.sin(-heading)
        # y_e = x * np.sin(-heading) + y * np.cos(-heading)
        # 将上述计算转换为矩阵运算
        rotation_matrix = np.array([[np.cos(-heading), -np.sin(-heading)],
                                    [np.sin(-heading), np.cos(-heading)]])
        path_transformed = np.dot(rotation_matrix, np.vstack([x,y]))
        x_e, y_e = path_transformed[0, :], path_transformed[1, :]
        # path = np.column_stack([x_e, y_e])
        return np.column_stack([x_e, y_e])
    
def transform_to_global_frame(path, actor):
    rear_axis_location = get_rear_axle_world_coordinates(actor)
    transform = actor.get_transform()
    x, y, heading = rear_axis_location.x, rear_axis_location.y, np.deg2rad(transform.rotation.yaw)
    rotation_matrix = np.array([[np.cos(heading), -np.sin(heading)],
                                [np.sin(heading), np.cos(heading)]])
    path_transformed = np.dot(rotation_matrix, path[:,:2].T).T
    x = path_transformed[:, 0] + x
    y = path_transformed[:, 1] + y
    return np.column_stack([x ,y])