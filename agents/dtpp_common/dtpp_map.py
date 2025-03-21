
import carla

from typing import Any, List, Dict, Callable, Union
from collections import defaultdict
from scipy.spatial.distance import cdist
import shapely.geometry as geom

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
# 用bokeh画出 trimmed paths
from bokeh.plotting import figure, show
from bokeh.models import Arrow, OpenHead, Text, ColumnDataSource
# from bokeh.io import output_notebook  # 如果在 Jupyter Notebook 中使用

from nuplan.planning.training.preprocessing.feature_builders.vector_builder_utils import *
from nuplan.common.actor_state.state_representation import Point2D
from nuplan.common.maps.maps_datatypes import (
    SemanticMapLayer,
    TrafficLightStatusData,
    TrafficLightStatuses,
    TrafficLightStatusType,
)

from agents.dtpp_common.dtpp_planner_utils import get_rear_axle_world_coordinates
# from agents.dtpp_common.dtpp_data_inputs import get_distance_between_dtpp_lane_and_point
from custom_format import *

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

    def draw_dtpp_map(self, actor, trajectory=None, bokeh: bool = True):
        if not bokeh:
            self._draw_map_top(self._routing)
        else:
            self._draw_map_top_bokeh(routing=self._routing, actor = actor, trajectory=trajectory)

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
    def _draw_trajectory(self, p,  trajectory, color="pink"):
        trj_x = [tsf.location.x for tsf in trajectory]
        trj_y = [tsf.location.y for tsf in trajectory]
        p.scatter(trj_x, trj_y, size=5, color=color, alpha=0.5, marker="o", line_width=20)
        # p.line(trj_x, trj_y, line_width=2, color=color)
        
    def _draw_vehicle(self, actor, p):
        bbox = actor.bounding_box
        transform = actor.get_transform()

        # 计算 Bounding Box 的顶点坐标
        vertices = [
            transform.transform(bbox.location + carla.Location(x=bbox.extent.x, y=bbox.extent.y, z=bbox.extent.z)),
            transform.transform(bbox.location + carla.Location(x=-bbox.extent.x, y=bbox.extent.y, z=bbox.extent.z)),
            transform.transform(bbox.location + carla.Location(x=-bbox.extent.x, y=-bbox.extent.y, z=bbox.extent.z)),
            transform.transform(bbox.location + carla.Location(x=bbox.extent.x, y=-bbox.extent.y, z=bbox.extent.z)),
            transform.transform(bbox.location + carla.Location(x=bbox.extent.x, y=bbox.extent.y, z=-bbox.extent.z)),
            transform.transform(bbox.location + carla.Location(x=-bbox.extent.x, y=bbox.extent.y, z=-bbox.extent.z)),
            transform.transform(bbox.location + carla.Location(x=-bbox.extent.x, y=-bbox.extent.y, z=-bbox.extent.z)),
            transform.transform(bbox.location + carla.Location(x=bbox.extent.x, y=-bbox.extent.y, z=-bbox.extent.z))
        ]
        # 提取 Bounding Box 的顶点坐标
        x = [v.x for v in vertices]
        y = [v.y for v in vertices]
        z = [v.z for v in vertices]

        # 定义 Bounding Box 的边（连接顶点的线段）
        edges = [
            [0, 1], [1, 2], [2, 3], [3, 0],  # 底面
            [4, 5], [5, 6], [6, 7], [7, 4],  # 顶面
            [0, 4], [1, 5], [2, 6], [3, 7]   # 侧面
        ]

        # 创建 Bokeh 数据源
        source = ColumnDataSource(data={
            'x': x,
            'y': y,
            'z': z
        })

        # 创建 Bokeh 图形
        # p = figure(title="Vehicle Bounding Box", x_axis_label='X', y_axis_label='Y', width=800, height=600)

        # 绘制 Bounding Box 的边
        for edge in edges:
            p.line(
                x=[x[edge[0]], x[edge[1]]],
                y=[y[edge[0]], y[edge[1]]],
                line_width=2,
                line_color="blue"
            )

    def _draw_map_top_bokeh(self, routing, actor: carla.Actor=None, trajectory=None):


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
        # 绘制车辆
        self._draw_vehicle(actor=actor, p=p)
        # 绘制候选车道线
        if actor:
            self.plot_candiate_lanes_bokeh(actor, p, colors)
        if trajectory:
            self._draw_trajectory(p, trajectory)
        show(p)

    def plot_candiate_lanes_bokeh(self, actor, p, colors):
        trim_lanes = self.get_candidate_paths(actor)
        # 遍历 trimmed_paths 并绘制每条路径
        for idx, path in enumerate(trim_lanes):
            x = path[2][:, 0].tolist()
            y = path[2][:, 1].tolist()
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
        logger.debug(f"--- function: {self._get_lane.__name__}")
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
                interval = 0.5
                while len(lane_line) < 3:
                    logger.warning(f"lane_line length is less than 2: {len(lane_line)}, stretch it!")
                    # TODO(fanyu): 这里结合 routing 信息查找下一个点应该更合理, 而且要去掉已经加入的lane
                    lane_line.append(Point2D(lane["exit"].transform.location.x, lane["exit"].transform.location.y))
                    next_wp = lane["exit"].next(interval)[-1] # lane["path"][-1].next(interval)[-1]
                    # while len(next_wps) * interval < max_length:
                    #     next_lane_spt = next_wps[-1].next(interval)[-1] # TODO(fanyu): next找到的是一个list，暂时取最后一个元素
                    #     next_wps_i = [next_lane_spt]+next_lane_spt.next_until_lane_end(interval)
                    #     next_wps += next_wps_i
                    next_lane_spt = next_wp.next_until_lane_end(interval)
                    lane_line += [Point2D(wp.transform.location.x, wp.transform.location.y) for wp in next_lane_spt]
                    
                    
                    
                dtpp_lanes.append(DtppLane(lane["entry"].lane_id, lane_line))  # TODO(fanyu): 是否替换为 DtppMapObject
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
        self.draw_dtpp_map(actor)
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
        # 1. 获取自车当前位置的 Waypoint，几何中心
        # vehicle_location = vehicle.get_location()
        vehicle_location = get_rear_axle_world_coordinates(vehicle)
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
            # logger.debug(f'--- shape:{distances.shape}, distances: {distances}')
            closest_idx = np.argmin(distances)
            closest_dis = distances[0, closest_idx]
            logger.debug(f'--- closest_dis:{closest_dis}, closest_idx:{closest_idx}')
            trimmed = path[closest_idx:]
            size = trimmed.shape[0]
            if size >= 3:
                trimmed_paths.append((size*interval, closest_dis, trimmed))

        return trimmed_paths
    

    


