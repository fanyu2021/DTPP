'''
Copyright (c) 2025 by GAC R&D Center, All Rights Reserved.
Author: 范雨
Date: 2025-02-27 11:25:25
LastEditTime: 2025-03-03 20:30:20
LastEditors: 范雨
Description: 
'''

from typing import Any, List, Dict, Callable, Union
from collections import defaultdict
import shapely.geometry as geom

import carla

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
    id : int = None
    mid_line : List[Point2D] = None
    left_boundary : List[Point2D] = None
    right_boundary : List[Point2D] = None

@dataclass
class DtppCrossWalk(object):
    id:int = None
    cross_walk_line : List[Point2D] = None

@dataclass
class DtppRoutLane(object):
    id:int = None
    route_lanes_line : List[Point2D] = None

DtppMapObject = Union[DtppLane, DtppCrossWalk, DtppRoutLane]

class DtppMap(object):
    def __init__(self, map: carla.Map, topology: List[Dict], routing) -> None:
      self._map = map
      self._topology = topology
      self._routing = routing
    #   self.draw_map_top(routing)
      self._road_block_ids = self._get_road_block_ids(routing)
      self._map_object_getter: Dict[SemanticMapLayer, Callable[[geom.Polygon], DtppMapObject]] = {
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
      
    def draw_map_top(self, routing):
        from tmp_test.test_2_road_graph_and_routing import draw_map
        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.axis("equal")
        plt.grid()
        plt.scatter(
            [wp_road_opt[0].transform.location.x for wp_road_opt in routing],
            [wp_road_opt[0].transform.location.y for wp_road_opt in routing],
            s=1,
            color="r",
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
        draw_map(self._world, self._map)
        # import time
        # plt.savefig(f'./routing_{time.time()}.png')
        plt.show()
      
    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        """Inherited, see superclass."""
        return list(self._map_object_getter.keys())
      
    def _get_lane(self, patch: geom.Polygon) -> List[DtppMapObject]:
        dtpp_lanes:List[DtppMapObject] = []
        for lane in self._topology:
            # lane_line = [Point2D(wp.transform.location.x, wp.transform.location.y) for wp in lane['path']]
            # lane = [Point2D(wp.transform.location.x, wp.transform.location.y) for wp in lane['path']]
            if patch.contains(geom.Point(lane['entry'].transform.location.x, lane['entry'].transform.location.y)):
                lane_line = [Point2D(wp.transform.location.x, wp.transform.location.y) for wp in lane['path']]
                dtpp_lanes.append(DtppLane(lane['entry'].lane_id, lane_line)) # TODO(fanyu): 是否替换为 DtppMapObject
        return dtpp_lanes
        
    def _get_lane_connector(self, patch: geom.Polygon):
        lane_connector_lines:List[DtppMapObject] = []
        for lane in self._topology:
            if not lane['entry'].is_junction:
                continue
            if patch.contains(geom.Point(lane['entry'].transform.location.x, lane['entry'].transform.location.y)):
                lane_connector_line = [Point2D(wp.transform.location.x, wp.transform.location.y) for wp in lane['path']]
                lane_connector_lines.append(DtppLane(lane['entry'].lane_id, lane_connector_line))
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
    def _get_proximity_map_object(self, patch: geom.Polygon, layer: SemanticMapLayer) -> List[MapObject]:
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

    def get_proximal_map_objects(self, point: Point2D, radius: float, layers: List[SemanticMapLayer]) -> Dict[SemanticMapLayer, List[MapObject]]:
        """Inherited, see superclass."""
        x_min, x_max = point.x - radius, point.x + radius
        y_min, y_max = point.y - radius, point.y + radius
        patch = geom.box(x_min, y_min, x_max, y_max)

        supported_layers = self.get_available_map_objects()
        unsupported_layers = [layer for layer in layers if layer not in supported_layers]

        assert len(unsupported_layers) == 0, f"Object representation for layer(s): {unsupported_layers} is unavailable"

        object_map: Dict[SemanticMapLayer, List[MapObject]] = defaultdict(list)

        for layer in layers:
            object_map[layer] = self._get_proximity_map_object(patch, layer)

        return object_map
def get_distance_between_dtpp_list2d_and_point(
    point: Point2D, 
    map_object: DtppRoutLane
) -> float: 
    # print(f"--- map_object: {map_object}")   
    # polygon = geom.Polygon([(pt.x, pt.y) for pt in map_object.route_lanes_line]).buffer(0)
    # return float(geom.Point(point.x, point.y).distance(polygon))
    mid_pt = Point2D(sum([pt.x for pt in map_object.route_lanes_line]) / len(map_object.route_lanes_line), sum([pt.y for pt in map_object.route_lanes_line]) / len(map_object.route_lanes_line))
    return np.linalg.norm(np.array([point.x, point.y]) - np.array([mid_pt.x, mid_pt.y]))

def get_distance_between_dtpp_lane_and_point(point: Point2D, map_object: Union[DtppLane, DtppCrossWalk, DtppRoutLane]) -> float:
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
        raise ValueError(f"Map object type {type(map_object)} is not supported for distance calculation.")

def get_lane_polylines(
    map_api: DtppMap, point: Point2D, radius: float
) -> Tuple[MapObjectPolylines, MapObjectPolylines, MapObjectPolylines, LaneSegmentLaneIDs]:
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
    lanes_mid: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_left: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lanes_right: List[List[Point2D]] = []  # shape: [num_lanes, num_points_per_lane (variable), 2]
    lane_ids: List[str] = []  # shape: [num_lanes]
    layer_names = [SemanticMapLayer.LANE, SemanticMapLayer.LANE_CONNECTOR]
    layers = map_api.get_proximal_map_objects(point, radius, layer_names)

    map_objects: List[DtppLane] = []

    for layer_name in layer_names:
        map_objects += layers[layer_name]
    # sort by distance to query point
    map_objects.sort(key=lambda map_obj: float(get_distance_between_dtpp_lane_and_point(point, map_obj)))

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

def prune_route_by_connectivity(route_roadblock_ids: List[str], roadblock_ids: Set[str]) -> List[str]:
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
    map_api: DtppMap, point: Point2D, radius: float) -> MapObjectPolylines:
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
    rout_lane : DtppRoutLane = None
    last_id = None

    for wp in map_api._routing:
        if last_id and wp[0].lane_id != last_id:
            route_lane_polylines.append(rout_lane)
            rout_lane = DtppRoutLane(id = wp[0].lane_id, route_lanes_line = [Point2D(wp[0].transform.location.x, wp[0].transform.location.y)])
        else:
            wpt = Point2D(wp[0].transform.location.x, wp[0].transform.location.y)
            distance_to_point = np.linalg.norm(
                np.array([wpt.x, wpt.y]) - np.array([point.x, point.y])
            )
            if distance_to_point < radius:
                if rout_lane:                
                    rout_lane.route_lanes_line.append(wpt)
                else:
                    rout_lane = DtppRoutLane(id = wp[0].lane_id, route_lanes_line = [wpt])
        last_id = wp[0].lane_id
        # print(f"--- last_id: {last_id}, rout_lane: {rout_lane}")

    if rout_lane.route_lanes_line:
        route_lane_polylines.append(rout_lane)

    route_lane_polylines.sort(key=lambda lane: float(get_distance_between_dtpp_list2d_and_point(point, lane)))
    poly_lines = [poly.route_lanes_line for poly in route_lane_polylines]        

    return MapObjectPolylines(poly_lines)


def get_crosswalk_polygons(map_api: DtppMap, point: Point2D, radius: float) -> MapObjectPolylines:
    corsswalks = map_api._map.get_crosswalks()
    corsswalks_info: List[DtppCrossWalk] = []
    crosswalk = DtppCrossWalk(id = 0, cross_walk_line = [])
    last_loc = None

    for loc in corsswalks:
        if last_loc and loc.distance(last_loc) < 30 and len(crosswalk.cross_walk_line) < 5:
            crosswalk.cross_walk_line.append(Point2D(loc.x, loc.y))
        else:
            def distance(p1: Point2D, cw: List[Point2D]) -> float:
                # 计算平均距离
                return sum([np.linalg.norm(np.array([p1.x, p1.y]) - np.array([p2.x, p2.y])) for p2 in cw]) / len(cw)
            
            if crosswalk.cross_walk_line and distance(point, crosswalk.cross_walk_line) < radius:
                crosswalk.id = len(corsswalks_info) + 1
                corsswalks_info.append(crosswalk)
            crosswalk.cross_walk_line = [Point2D(loc.x, loc.y)]
            last_loc = loc

    if crosswalk.cross_walk_line:
        crosswalk.id = len(corsswalks_info) + 1
        corsswalks_info.append(crosswalk)
        
    corsswalks_info.sort(key=lambda crosswalk: float(get_distance_between_dtpp_lane_and_point(point, crosswalk)))
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
            raise ValueError(f"Object representation for layer: {feature_name} is unavailable")

    # extract lanes
    if VectorFeatureLayer.LANE in feature_layers:
        lanes_mid, lanes_left, lanes_right, lane_ids = get_lane_polylines(map_api, point, radius)

        # lane baseline paths
        coords[VectorFeatureLayer.LANE.name] = lanes_mid

        # lane traffic light data
        traffic_light_data[VectorFeatureLayer.LANE.name] = get_traffic_light_encoding(lane_ids, traffic_light_status_data)

        # lane boundaries
        if VectorFeatureLayer.LEFT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.LEFT_BOUNDARY.name] = MapObjectPolylines(lanes_left.polylines)
        if VectorFeatureLayer.RIGHT_BOUNDARY in feature_layers:
            coords[VectorFeatureLayer.RIGHT_BOUNDARY.name] = MapObjectPolylines(lanes_right.polylines)

    # extract route
    if VectorFeatureLayer.ROUTE_LANES in feature_layers:
        route_polylines = get_route_lane_polylines_from_roadblock_ids(map_api, point, radius)
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
