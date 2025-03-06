"""
Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
Author: 范雨
Date: 2025-02-27 11:25:25
LastEditTime: 2025-03-05 16:38:46
LastEditors: 范雨
Description: 
"""

from typing import Any, List, Dict, Callable, Union
from collections import defaultdict
import shapely.geometry as geom
from scipy.spatial.distance import cdist

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# 用bokeh画出 trimmed paths
from bokeh.plotting import figure, show
from bokeh.models import Arrow, OpenHead, Text, ColumnDataSource
from bokeh.io import output_notebook  # 如果在 Jupyter Notebook 中使用


import carla

from custom_format import *

from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatuses,
    TrafficLightStatusType,
)

# MapObject = Union[Lane, LaneConnector, RoadBlockGraphEdgeMapObject, PolygonMapObject, Intersection, StopLine]


@dataclass
class DtppLane(object):
    id: int = None
    mid_line: List[Point2D] = None
    left_boundary: List[Point2D] = None
    right_boundary: List[Point2D] = None


@dataclass
class DtppCrossWalk(object):
    id: int = None
    cross_walk_line: List[Point2D] = None


@dataclass
class DtppRoutLane(object):
    id: int = None
    route_lanes_line: List[Point2D] = None


DtppMapObject = Union[DtppLane, DtppCrossWalk, DtppRoutLane]


class DtppMap(object):
    def __init__(self, map: carla.Map, topology: List[Dict], routing) -> None:
        self._map = map
        self._topology = topology
        self._routing = routing
        self._road_block_ids = self._get_road_block_ids(routing)
        self._map_object_getter: Dict[
            SemanticMapLayer, Callable[[geom.Polygon], DtppMapObject]
        ] = {
            SemanticMapLayer.LANE: self._get_lane,
            SemanticMapLayer.LANE_CONNECTOR: self._get_lane_connector,
            SemanticMapLayer.ROADBLOCK: self._get_roadblock,
            SemanticMapLayer.ROADBLOCK_CONNECTOR: self._get_roadblock_connector,
            SemanticMapLayer.STOP_LINE: self._get_stop_line,
            SemanticMapLayer.CROSSWALK: self._get_crosswalk,
            SemanticMapLayer.INTERSECTION: self._get_intersection,
            SemanticMapLayer.WALKWAYS: self._get_walkway,
            SemanticMapLayer.CARPARK_AREA: self._get_carpark_area,
        }

    def _get_road_block_ids(self, routing):
        ids = [wp[0].lane_id for wp in routing]
        return list(set(ids))

    def draw_dtpp_map(self, actor, bokeh: bool = True):
        if not bokeh:
            self._draw_map_top(self._routing)
        else:
            self._draw_map_top_bokeh(self._routing, actor)

    def _draw_map_top(self, routing, vehicle:carla.Actor=None):
        # from tmp_test.test_2_road_graph_and_routing import draw_map

        plt.figure(figsize=(10, 10))
        plt.axis("equal")
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        plt.scatter(
            [wp_road_opt[0].transform.location.x for wp_road_opt in routing],
            [wp_road_opt[0].transform.location.y for wp_road_opt in routing],
            s=1,
            color="y",
            alpha=0.5,
            # label="route points",
            linewidths=8,
        )
        # 分别画出起点和终点
        plt.scatter(
            routing[0][0].transform.location.x,
            routing[0][0].transform.location.y,
            c="r",
            marker="o",
        )
        plt.scatter(
            routing[-1][0].transform.location.x,
            routing[-1][0].transform.location.y,
            c="g",
            marker="o",
        )

        # draw_map(self._world, self._map)
        self._draw_topology()
        # import time
        # plt.savefig(f'./routing_{time.time()}.png')
        plt.show()

    def _draw_topology(self):
        for lane in self._topology:
            lane_line = [
                Point2D(wp.transform.location.x, wp.transform.location.y)
                for wp in lane["path"]
            ]
            color = "red" if lane["entry"].is_junction else "blue"
            line_type = "dashed" if lane["entry"].lane_id < 0 else "solid"
            plt.plot(
                [pt.x for pt in lane_line],
                [pt.y for pt in lane_line],
                color=color,
                linestyle=line_type,
            )
            last_theta = lane["entry"].transform.rotation.yaw
            l_arrow = 0.01
            dir = (
                np.array(
                    [np.cos(np.deg2rad(last_theta)), np.sin(np.deg2rad(last_theta))]
                )
                * l_arrow
            )

            plt.arrow(
                lane["entry"].transform.location.x,
                lane["entry"].transform.location.y,
                dir[0],
                dir[1],
                head_width=1,
                head_length=5,
                fc=color,
                ec=color,
            )
            plt.text(
                lane["entry"].transform.location.x + 0.5,
                lane["entry"].transform.location.y + 0.5,
                "s%d.r%d\nl%d.j%d"
                % (
                    lane["entry"].section_id,
                    lane["entry"].road_id,
                    lane["entry"].lane_id,
                    lane["entry"].junction_id,
                ),
                fontdict={"fontsize": 14, "color": color, "fontweight": "bold"},
            )
        # plt.show()

    def _draw_map_top_bokeh(self, routing, actor: carla.Actor=None):


        p = figure(
            title="Dtpp Map",
            width=1280,
            height=1080,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            match_aspect=True,
        )
        p.grid.visible = True
        p.xaxis.axis_label = "X"
        p.yaxis.axis_label = "Y"

        # 绘制路由点
        route_x = [wp_road_opt[0].transform.location.x for wp_road_opt in routing]
        route_y = [wp_road_opt[0].transform.location.y for wp_road_opt in routing]
        p.scatter(route_x, route_y, size=5, color="yellow", alpha=0.5)


                # 定义颜色列表，用于区分不同的路径
        colors = ["blue", "green", "red", "orange", "purple", "brown"]
        
        # 绘制起点和终点
        p.scatter(
            [routing[0][0].transform.location.x],
            [routing[0][0].transform.location.y],
            size=15,
            color="red",
            marker="circle",
        )
        p.scatter(
            [routing[-1][0].transform.location.x],
            [routing[-1][0].transform.location.y],
            size=15,
            color="green",
            marker="circle",
        )

        # 绘制拓扑结构
        self._draw_topology_bokeh(p)
        # 绘制候选车道线
        if actor:
            self.plot_candiate_lanes_bokeh(actor, p, colors)
        show(p)

    def plot_candiate_lanes_bokeh(self, actor, p, colors):
        trim_lanes = self.get_candidate_paths(actor)
        # 遍历 trimmed_paths 并绘制每条路径
        for idx, path in enumerate(trim_lanes):
            x = path[:, 0].tolist()
            y = path[:, 1].tolist()
            # p.line(
            #     x,
            #     y,
            #     line_width=2,
            #     color=colors[idx % len(colors)],
            #     legend_label=f"Path {idx+1}",
            # )
            p.scatter(x, y, size=5, color=colors[idx % len(colors)], alpha=0.5)

        # 添加图例位置
        p.legend.location = "top_left"

    def _draw_topology_bokeh(self, p):
        from bokeh.models import Arrow, OpenHead, Segment, Text, Label

        for lane in self._topology:
            lane_line = [
                (wp.transform.location.x, wp.transform.location.y)
                for wp in lane["path"]
            ]
            color = "red" if lane["entry"].is_junction else "blue"
            line_type = "dashed" if lane["entry"].lane_id < 0 else "solid"

            # 绘制车道线
            p.multi_line(
                [[pt[0] for pt in lane_line]],
                [[pt[1] for pt in lane_line]],
                line_color=color,
                line_dash="dotdash" if line_type == "dashed" else "solid",
                line_width=1,
            )

            # 绘制箭头
            start_x = lane["entry"].transform.location.x
            start_y = lane["entry"].transform.location.y
            theta = np.deg2rad(lane["entry"].transform.rotation.yaw)
            l_arrow = 5.0
            end_x = start_x + np.cos(theta) * l_arrow
            end_y = start_y + np.sin(theta) * l_arrow

            p.add_layout(
                Arrow(
                    end=OpenHead(size=10),
                    x_start=start_x,
                    y_start=start_y,
                    x_end=end_x,
                    y_end=end_y,
                    line_color=color,
                )
            )

            # 添加文本标签
            p.add_layout(
                Label(
                    x=start_x + 0.5,
                    y=start_y + 0.5,
                    text=f"s{lane['entry'].section_id}.r{lane['entry'].road_id}\nl{lane['entry'].lane_id}.j{lane['entry'].junction_id}",
                    text_font_size="14px",
                    text_color=color,
                    text_font_style="bold",
                )
            )

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """Inherited, see superclass."""
        return list(self._map_object_getter.keys())

    def _get_lane(self, patch: geom.Polygon) -> List[DtppMapObject]:
        dtpp_lanes: List[DtppMapObject] = []
        for lane in self._topology:
            # lane_line = [Point2D(wp.transform.location.x, wp.transform.location.y) for wp in lane['path']]
            # lane = [Point2D(wp.transform.location.x, wp.transform.location.y) for wp in lane['path']]
            if patch.contains(
                geom.Point(
                    lane["entry"].transform.location.x,
                    lane["entry"].transform.location.y,
                )
            ):
                lane_line = [
                    Point2D(wp.transform.location.x, wp.transform.location.y)
                    for wp in lane["path"]
                ]
                dtpp_lanes.append(
                    DtppLane(lane["entry"].lane_id, lane_line)
                )  # TODO(fanyu): 是否替换为 DtppMapObject
        return dtpp_lanes

    def _get_lane_connector(self, patch: geom.Polygon):
        lane_connector_lines: List[DtppMapObject] = []
        for lane in self._topology:
            if not lane["entry"].is_junction:
                continue
            if patch.contains(
                geom.Point(
                    lane["entry"].transform.location.x,
                    lane["entry"].transform.location.y,
                )
            ):
                lane_connector_line = [
                    Point2D(wp.transform.location.x, wp.transform.location.y)
                    for wp in lane["path"]
                ]
                lane_connector_lines.append(
                    DtppLane(lane["entry"].lane_id, lane_connector_line)
                )
        return lane_connector_lines

    def _get_roadblock(self, patch: geom.Polygon):
        pass

    def _get_roadblock_connector(self, patch: geom.Polygon):
        pass

    def _get_stop_line(self, patch: geom.Polygon):
        pass

    def _get_crosswalk(self, patch: geom.Polygon):
        pass

    def _get_intersection(self, patch: geom.Polygon):
        pass

    def _get_walkway(self, patch: geom.Polygon):
        pass

    def _get_carpark_area(self, patch: geom.Polygon):
        pass

    def _get_proximity_map_object(
        self, patch: geom.Polygon, layer: SemanticMapLayer
    ) -> List[MapObject]:
        """
        Gets nearby lanes within the given patch.
        :param patch: The area to be checked.
        :param layer: desired layer to check.
        :return: A list of map objects.
        """
        # layer_df = self._get_vector_map_layer(layer)
        # map_object_ids = layer_df[layer_df['geometry'].intersects(patch)]['fid']

        # return [self.get_map_object(map_object_id, layer) for map_object_id in map_object_ids]

        # 通过给定的patch和layer获取对应的地图对象
        return self._map_object_getter[layer](patch)

    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[SemanticMapLayer]
    ) -> Dict[SemanticMapLayer, List[MapObject]]:
        """Inherited, see superclass."""
        x_min, x_max = point.x - radius, point.x + radius
        y_min, y_max = point.y - radius, point.y + radius
        patch = geom.box(x_min, y_min, x_max, y_max)

        supported_layers = self.get_available_map_objects()
        unsupported_layers = [
            layer for layer in layers if layer not in supported_layers
        ]

        assert (
            len(unsupported_layers) == 0
        ), f"Object representation for layer(s): {unsupported_layers} is unavailable"

        object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

        for layer in layers:
            object_map[layer] = self._get_proximity_map_object(patch, layer)

        return object_map

    def get_candidate_traffic_lanes(
        self, actor: carla.Actor
    ) -> List[Dict]:
        """
        Get candidate lanes based on a location.
        Args:
            loc: Location to search for lanes.
        Returns:
            List of candidate lanes.
        """
        # self.draw_dtpp_map(actor)
        candidates: List[Dict] = []
        carla_map = self._map

        cur_wp = carla_map.get_waypoint(
            actor.get_location(), project_to_road=True, lane_type=carla.LaneType.Driving
        )
        for lane in self._topology:
            entry_wp = lane["entry"]
            dis = entry_wp.s - cur_wp.s
            is_near_front = dis > 0 and dis < 30
            if (
                entry_wp.road_id != cur_wp.road_id
                or entry_wp.lane_id < 0
                or not is_near_front
            ):
                continue  # 跳过非同一道路的lane, 以及负向lane
            candidates.append(lane)
        return candidates

    def get_candidate_paths(self, vehicle, max_length=200, interval=0.25):
        """
        基于 CARLA 地图和自车位置生成候选路径
        :param vehicle: 自车对象（需包含位置信息）
        :param carla_map: CARLA 地图对象
        :param max_length: 路径最大长度（米）
        :param interval: 路径点采样间隔（米）
        :return: 候选路径列表，每条路径为带航向的 NumPy 数组
        """
        # 1. 获取自车当前位置的 Waypoint
        vehicle_location = vehicle.get_location()
        # vehicle_location = location
        current_wp = self._map.get_waypoint(
            vehicle_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving
        )

        # 2. 生成候选车道（当前车道及相邻车道）
        candidate_lanes = []
        lane_changes = [
            carla.LaneChange.NONE,  # 当前车道
            carla.LaneChange.Left,  # 左车道（如果存在）
            carla.LaneChange.Right # 右车道（如果存在）
        ]

        for lane_change in lane_changes:
            # 检查是否允许变道
            # if current_wp.lane_change & lane_change:
            if True:
                next_wps = current_wp.next_until_lane_end(interval)
                prev_wps = current_wp.previous_until_lane_start(interval).reverse()
                prev_wps = prev_wps + [current_wp] if prev_wps else [current_wp]
                if lane_change != carla.LaneChange.NONE:
                    # 获取相邻车道的 Waypoint
                    adjacent_wp = current_wp.get_left_lane() if lane_change == carla.LaneChange.Left \
                        else current_wp.get_right_lane()
                    if adjacent_wp is not None and adjacent_wp.lane_type == carla.LaneType.Driving and adjacent_wp.lane_id > 0:
                        next_wps = adjacent_wp.next_until_lane_end(interval)
                        prev_wps = adjacent_wp.previous_until_lane_start(interval).reverse()
                        prev_wps = prev_wps + [adjacent_wp] if prev_wps else [adjacent_wp]
                        
                while len(next_wps) * interval < max_length:
                    next_lane_spt = next_wps[-1].next(interval)[-1] # TODO(fanyu): next找到的是一个list，暂时取最后一个元素
                    next_wps_i = [next_lane_spt]+next_lane_spt.next_until_lane_end(interval)
                    next_wps += next_wps_i
                # logger.debug(f"Adjacent lane found: {len(next_wps)}")
                # prev_wps = prev_wps if prev_wps else [current_wp]
                candidate_lanes.append(prev_wps + next_wps)

        # 3. 生成路径点序列
        candidate_paths = []
        for lane in candidate_lanes:
            path = []
            for wp in lane:
                path.append([wp.transform.location.x, wp.transform.location.y])
                if len(path) * interval >= max_length:
                    break
            if len(path) < 3:
                continue  # 跳过过短路径

            # 转换为 NumPy 数组并计算航向
            path = np.array(path)
            headings = np.arctan2(np.diff(path[:,1]), np.diff(path[:,0]))
            headings = np.append(headings, headings[-1])  # 补全最后一个航向
            path = np.column_stack((path, headings))

            candidate_paths.append(path)
            # logger.debug(f"--- Found candidate path: {len(candidate_paths)}")

        # 4. 根据自车位置修剪路径
        ego_point = np.array([[vehicle_location.x, vehicle_location.y]])
        trimmed_paths = []
        for path in candidate_paths:
            # 找到距离自车最近的路径点
            distances = cdist(ego_point, path[:, :2])
            logger.debug(f'--- shape:{distances.shape}, distances: {distances}')
            closest_idx = np.argmin(distances)
            closest_dis = distances[0, closest_idx]
            trimmed = path[closest_idx:]
            size = trimmed.shape[0]
            if size >= 3:
                trimmed_paths.append((size*interval, closest_dis, trimmed))

        return trimmed_paths


def get_distance_between_dtpp_list2d_and_point(
    point: Point2D, map_object: DtppRoutLane
) -> float:
    # print(f"--- map_object: {map_object}")
    # polygon = geom.Polygon([(pt.x, pt.y) for pt in map_object.route_lanes_line]).buffer(0)
    # return float(geom.Point(point.x, point.y).distance(polygon))
    mid_pt = Point2D(
        sum([pt.x for pt in map_object.route_lanes_line])
        / len(map_object.route_lanes_line),
        sum([pt.y for pt in map_object.route_lanes_line])
        / len(map_object.route_lanes_line),
    )
    return np.linalg.norm(np.array([point.x, point.y]) - np.array([mid_pt.x, mid_pt.y]))


def get_distance_between_dtpp_lane_and_point(
    point: Point2D, map_object: Union[DtppLane, DtppCrossWalk, DtppRoutLane]
) -> float:
    if isinstance(map_object, DtppLane):
        polygon = geom.Polygon([(pt.x, pt.y) for pt in map_object.mid_line]).buffer(0)
        return float(geom.Point(point.x, point.y).distance(polygon))
    if isinstance(map_object, DtppCrossWalk):
        polygon = geom.Polygon([(pt.x, pt.y) for pt in map_object.cross_walk_line])
        return float(geom.Point(point.x, point.y).distance(polygon))
    # if isinstance(map_object, DtppRoutLane):
    #     polygon = geom.Polygon([(pt.x, pt.y) for pt in map_object.route_lanes_line]).buffer(0)
    #     return float(geom.Point(point.x, point.y).distance(polygon))
    else:
        raise ValueError(
            f"Map object type {type(map_object)} is not supported for distance calculation."
        )


def get_lane_polylines(
    map_api: DtppMap, point: Point2D, radius: float
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
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)

    map_objects: List[DtppLane] = []

    for layer_name in layer_names:
        map_objects += layers[layer_name]
    # sort by distance to query point
    map_objects.sort(
        key=lambda map_obj: float(
            get_distance_between_dtpp_lane_and_point(point, map_obj)
        )
    )

    for map_obj in map_objects:
        # center lane
        # baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj]
        lanes_mid.append(map_obj.mid_line)

        # # boundaries
        # lanes_left.append([Point2D(node.x, node.y) for node in map_obj.left_boundary.discrete_path])
        # lanes_right.append([Point2D(node.x, node.y) for node in map_obj.right_boundary.discrete_path])

        # lane ids
        lane_ids.append(map_obj.id)

    return (
        MapObjectPolylines(lanes_mid),
        MapObjectPolylines(lanes_left),
        MapObjectPolylines(lanes_right),
        LaneSegmentLaneIDs(lane_ids),
    )


def prune_route_by_connectivity(
    route_roadblock_ids: List[str], roadblock_ids: Set[str]
) -> List[str]:
    """
    Prune route by overlap with extracted roadblock elements within query radius to maintain connectivity in route
    feature. Assumes route_roadblock_ids is ordered and connected to begin with.
    :param route_roadblock_ids: List of roadblock ids representing route.
    :param roadblock_ids: Set of ids of extracted roadblocks within query radius.
    :return: List of pruned roadblock ids (connected and within query radius).
    """
    pruned_route_roadblock_ids: List[str] = []
    route_start = False  # wait for route to come into query radius before declaring broken connection

    for roadblock_id in route_roadblock_ids:

        if roadblock_id in roadblock_ids:
            pruned_route_roadblock_ids.append(roadblock_id)
            route_start = True

        elif route_start:  # connection broken
            break

    return pruned_route_roadblock_ids


def get_route_lane_polylines_from_roadblock_ids(
    map_api: DtppMap, point: Point2D, radius: float
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
    # route_lane_polylines: List[DtppRoutLane] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    # map_objects = []

    # extract roadblocks/connectors within query radius to limit route consideration
    # layer_names = [SemanticMapLayer.ROADBLOCK, SemanticMapLayer.ROADBLOCK_CONNECTOR]
    # layers = map_api.get_proximal_map_objects(point, radius, layer_names)
    # roadblock_ids: Set[int] = set()

    # for layer_name in layer_names:
    #     roadblock_ids = roadblock_ids.union({map_object.id for map_object in layers[layer_name]})
    # # prune route by connected roadblocks within query radius
    # route_roadblock_ids = prune_route_by_connectivity(map_api._road_block_ids, roadblock_ids)

    # for route_roadblock_id in route_roadblock_ids:
    #     # roadblock
    #     roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK)

    #     # roadblock connector
    #     if not roadblock_obj:
    #         roadblock_obj = map_api.get_map_object(route_roadblock_id, SemanticMapLayer.ROADBLOCK_CONNECTOR)

    #     # represent roadblock/connector by interior lanes/connectors
    #     if roadblock_obj:
    #         map_objects += roadblock_obj.interior_edges

    # # sort by distance to query point
    # map_objects.sort(key=lambda map_obj: float(get_distance_between_map_object_and_point(point, map_obj)))

    # for map_obj in map_objects:
    #     baseline_path_polyline = [Point2D(node.x, node.y) for node in map_obj.baseline_path.discrete_path]
    #     route_lane_polylines.append(baseline_path_polyline)
    # lane_id = last_lane_id = -1

    # lane_center_line = List[Point2D]
    # last_lane_id = -1000
    # for wp in map_api._routing:
    #     if wp[0].lane_id != last_lane_id and lane_center_line:
    #         route_lane = DtppRoutLane(id = last_lane_id, route_lanes_line = lane_center_line)
    #         route_lane_polylines.append(route_lane)
    #         lane_center_line = []
    #     lane_center_line.append(Point2D(wp[0].transform.location.x, wp[0].transform.location.y))
    #     last_lane_id = wp[0].lane_id
    # route_lane = DtppRoutLane(id = last_lane_id, route_lanes_line = lane_center_line)
    # route_lane_polylines.append(route_lane)

    route_lane_polylines: List[DtppRoutLane] = []
    rout_lane: DtppRoutLane = None
    last_id = None

    for wp in map_api._routing:
        if last_id and wp[0].lane_id != last_id:
            route_lane_polylines.append(rout_lane)
            rout_lane = DtppRoutLane(
                id=wp[0].lane_id,
                route_lanes_line=[
                    Point2D(wp[0].transform.location.x, wp[0].transform.location.y)
                ],
            )
        else:
            wpt = Point2D(wp[0].transform.location.x, wp[0].transform.location.y)
            distance_to_point = np.linalg.norm(
                np.array([wpt.x, wpt.y]) - np.array([point.x, point.y])
            )
            if distance_to_point < radius:
                if rout_lane:
                    rout_lane.route_lanes_line.append(wpt)
                else:
                    rout_lane = DtppRoutLane(id=wp[0].lane_id, route_lanes_line=[wpt])
        last_id = wp[0].lane_id
        # print(f"--- last_id: {last_id}, rout_lane: {rout_lane}")

    if rout_lane.route_lanes_line:
        route_lane_polylines.append(rout_lane)

    route_lane_polylines.sort(
        key=lambda lane: float(get_distance_between_dtpp_list2d_and_point(point, lane))
    )
    poly_lines = [poly.route_lanes_line for poly in route_lane_polylines]

    return MapObjectPolylines(poly_lines)


def get_crosswalk_polygons(
    map_api: DtppMap, point: Point2D, radius: float
) -> MapObjectPolylines:
    corsswalks = map_api._map.get_crosswalks()
    corsswalks_info: List[DtppCrossWalk] = []
    crosswalk = DtppCrossWalk(id=0, cross_walk_line=[])
    last_loc = None

    for loc in corsswalks:
        if (
            last_loc
            and loc.distance(last_loc) < 30
            and len(crosswalk.cross_walk_line) < 5
        ):
            crosswalk.cross_walk_line.append(Point2D(loc.x, loc.y))
        else:

            def distance(p1: Point2D, cw: List[Point2D]) -> float:
                # 计算平均距离
                return sum(
                    [
                        np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y]))
                        for p2 in cw
                    ]
                ) / len(cw)

            if (
                crosswalk.cross_walk_line
                and distance(point, crosswalk.cross_walk_line) < radius
            ):
                crosswalk.id = len(corsswalks_info) + 1
                corsswalks_info.append(crosswalk)
            crosswalk.cross_walk_line = [Point2D(loc.x, loc.y)]
            last_loc = loc

    if crosswalk.cross_walk_line:
        crosswalk.id = len(corsswalks_info) + 1
        corsswalks_info.append(crosswalk)

    corsswalks_info.sort(
        key=lambda crosswalk: float(
            get_distance_between_dtpp_lane_and_point(point, crosswalk)
        )
    )
    polylines = [poly.cross_walk_line for poly in corsswalks_info]

    return MapObjectPolylines(polylines)


def get_neighbor_vector_set_map(
    map_api: DtppMap,
    map_features: List[str],
    point: Point2D,
    radius: float,
    # route_roadblock_ids: List[str],
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
            raise ValueError(
                f"Object representation for layer: {feature_name} is unavailable"
            )

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(
            map_api, point, radius
        )

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # lane traffic light data
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(
            lane_ids, traffic_light_status_data
        )

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(
                lanes_left.polylines
            )
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(
                lanes_right.polylines
            )

    # extract route
    if VectorFeatureLayer.ROUTE_LANES in feature_layers:
        route_polylines = get_route_lane_polylines_from_roadblock_ids(
            map_api, point, radius
        )
        coords[VectorFeatureLayer.ROUTE_LANES.name] = route_polylines

    # extract generic map objects
    # for feature_layer in feature_layers:
    #     if feature_layer in VectorFeatureLayerMapping.available_polygon_layers():
    #         polygons = get_map_object_polygons(
    #             map_api, point, radius, VectorFeatureLayerMapping.semantic_map_layer(feature_layer)
    #         )
    #         coords[feature_layer.name] = polygons

    if VectorFeatureLayer.CROSSWALK in feature_layers:
        polygons = get_crosswalk_polygons(map_api, point, radius)
        coords[VectorFeatureLayer.CROSSWALK.name] = polygons

    return coords, traffic_light_data
