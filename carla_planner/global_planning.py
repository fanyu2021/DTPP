#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : global_planning.py

import math
import carla
import networkx as nx
import numpy as np
from enum import Enum
from carla_planner import planning_utils



class RoadOption(Enum):
    """
    RoadOption represents the possible topological configurations when moving from a segment of lane to others.
    RoadOption表示从一段车道移动到其他车道时可能的拓扑配置
    """
    VOID = -1
    LEFT = 1
    RIGHT = 2
    STRAIGHT = 3
    LANE_FOLLOW = 4
    CHANGE_LANE_LEFT = 5
    CHANGE_LANE_RIGHT = 6


class global_path_planner(object):
    def __init__(self, world_map, sampling_resolution):
        self._map = world_map  # type: carla.Map
        self._sampling_resolution = sampling_resolution
        self._topology = None
        self._graph = nx.DiGraph()  # type: nx.DiGraph
        self._id_map = None
        self._road_to_edge = None
        self._start_loc = None
        self._end_loc = None
        self._has_routing = False
        self._path_way = []

        # initiate the planner
        self._build_topology()
        self._build_graph()

    def get_topology_and_graph_info(self):
        return self._topology, self._graph, self._id_map, self._road_to_edge
    
    def get_start_end_waypoint_by_node_id(self, start_node_id: int, end_node_id: int):
        """
        获取起点和终点的路点
        param:  start_node_id: 起点所在边的左端点id
                end_node_id:  终点所在边的左端点id
        return: 起点和终点的路点

        """
        start_node_loc = carla.Location(x=self._graph.nodes[start_node_id]['vertex'][0], y=self._graph.nodes[start_node_id]['vertex'][1], z=self._graph.nodes[start_node_id]['vertex'][2])
        destination_node_loc = carla.Location(x=self._graph.nodes[end_node_id]['vertex'][0], y=self._graph.nodes[end_node_id]['vertex'][1], z=self._graph.nodes[end_node_id]['vertex'][2])
        return start_node_loc, destination_node_loc
        
        

    def _build_topology(self):
        """
        the output of carla.Map.get_topology() could look like this: [(w0, w1), (w0, w2), (w1, w3), (w2, w3), (w0, w4)].
        由于carla.Map.get_topology()只能函数获取起点和终点信息构成的边信息，这些信息不能够为全局路径规划提供细节信息，因此需要重新构建拓扑
        新拓扑用字典类型存储每个路段，具有以下结构：
        {
        entry (carla.Waypoint): waypoint of entry point of road segment
        exit (carla.Waypoint): waypoint of exit point of road segment
        path (list of carla.Waypoint):  list of waypoints between entry to exit, separated by the resolution
        }
        return: None
        注：carla自带的_build_topology中还有entryxyz,exitxyz属性，在这里不做使用
        """
        self._topology = []
        for seg in self._map.get_topology():
            w1 = seg[0]  # type: carla.Waypoint
            w2 = seg[1]  # type: carla.Waypoint
            new_seg = dict()
            new_seg["entry"] = w1
            new_seg["exit"] = w2
            new_seg["path"] = []
            # 按照采样分辨率将w1和w2之间的路径点采样出来
            w1_loc = w1.transform.location  # type: carla.Location
            if w1_loc.distance(w2.transform.location) > self._sampling_resolution:
                # 如果起始路点和结束路点之间存在其他路点，则根据采样分辨率将中间点全部存储在new_seg["path"]中
                new_waypoint = w1.next(self._sampling_resolution)[0]  # 这里从起始路点的下一个开始，
                while new_waypoint.transform.location.distance(w2.transform.location) > self._sampling_resolution:
                    # 结束路点不会记录到new_seg["path"]中
                    new_seg["path"].append(new_waypoint)
                    new_waypoint = new_waypoint.next(self._sampling_resolution)[0]
            else:  # 如果起始路点和结束路点之间的距离小于或等于采样分辨率，则仍然让new_seg["path"]保持空列表
                # new_seg["path"].append(w1.next(self._sampling_resolution)[0])
                pass
            self._topology.append(new_seg)

    def _build_graph(self):
        """"
        构建图，方便可视化和运用图论的知识进行全局路径规划
        self._graph是一个二向图，属性如下：
            Node properties:
                vertex: (x,y,z) position in world map， 在DiGraph类型下数据结构为{id: {'vertex': (x, y, z)}}
            Edge properties:
                entry_vector: 入口点沿切线方向的单位向量（unit vector along tangent at entry point）
                exit_vector: 出口点沿切线方向的单位向量（unit vector along tangent at exit point）
                net_vector:  入口指向出口的方向的单位向量（unit vector of the chord from entry to exit）
                intersection: 布尔类型，是否属于交叉路口boolean indicating if the edge belongs to an  intersection
        self._id_map  # 字典类型，建立节点id和位置的对应{(x, y, z): id}
        self._road_to_edge  # 字典类型，建立road_id,section_id,lane_id 和边的对应关系
        """
        # self._graph = nx.DiGraph()  # it is initializes in the
        self._id_map = dict()  # 字典类型，建立节点id和位置的对应{(x, y, z): id}
        self._road_to_edge = dict()  # 字典类型，建立road_id,section_id,lane_id 和边的对应关系

        for seg in self._topology:
            entry_waypoint = seg["entry"]  # type: carla.Waypoint
            exit_waypoint = seg["exit"]  # type: carla.Waypoint
            path = seg["path"]  # 不包含端点
            intersection = entry_waypoint.is_intersection
            road_id, section_id, lane_id = entry_waypoint.road_id, entry_waypoint.section_id, entry_waypoint.lane_id
            entry_xyz = entry_waypoint.transform.location
            entry_xyz = (np.round(entry_xyz.x, 2), np.round(entry_xyz.y, 2), np.round(entry_xyz.z, 2))  # 对小数长度进行限制
            exit_xyz = exit_waypoint.transform.location
            exit_xyz = (np.round(exit_xyz.x, 2), np.round(exit_xyz.y, 2), np.round(exit_xyz.z, 2))
            for xyz in entry_xyz, exit_xyz:
                if xyz not in self._id_map:
                    New_ID = len(self._id_map)
                    self._id_map[xyz] = New_ID
                    # 将新的节点加入graph
                    self._graph.add_node(New_ID, vertex=xyz)

            n1 = self._id_map[entry_xyz]
            n2 = self._id_map[exit_xyz]

            if road_id not in self._road_to_edge:
                self._road_to_edge[road_id] = dict()
            if section_id not in self._road_to_edge[road_id]:
                self._road_to_edge[road_id][section_id] = dict()
            # 会有左右车道和多车道的情况 举例 13: {0: {-1: (34, 46), 1: (47, 31)}}，
            # 即id为13的道路，包含一个section,这个section是双向单车道
            self._road_to_edge[road_id][section_id][lane_id] = (n1, n2)

            entry_forward_vector = entry_waypoint.transform.rotation.get_forward_vector()  # 这里是入口节点的方向信息
            exit_forward_vector = exit_waypoint.transform.rotation.get_forward_vector()  # 这里是出口节点的方向信息，用于车辆规划路径时的转向

            # 将新的边加入graph
            self._graph.add_edge(u_of_edge=n1, v_of_edge=n2,
                                 length=len(path) + 1, path=path,
                                 entry_waypoint=entry_waypoint, exit_waypoint=exit_waypoint,
                                 entry_vector=entry_forward_vector, exit_vector=exit_forward_vector,
                                 net_vector=planning_utils.Vector_fun(entry_waypoint.transform.location,
                                                                      exit_waypoint.transform.location),
                                 intersection=intersection, type=RoadOption.LANE_FOLLOW)

    def _find_location_edge(self, loc: carla.Location):
        """
        用于查找给定位置所属的路段，并返回其所属的edge
        param:  loc: 给定的一个位置
        return: 返回graph的一条边(n1, n2)
        """
        nearest_wp = self._map.get_waypoint(loc)  # type: carla.Waypoint
        # 现在面临一个问题，对于两个路段相接处的节点，定位的是前一个路段还是后一个路段,在路径规划中二者本质上没有区别，但是自己没有搞明白这个方法的原理
        # 测试的结果是在交叉路口或者弯道情况下，返回的是后一个路段； 在直线道路中返回的是前一个路段
        edge = None
        try:
            # 用最近的路点所在的road_id,section_id和lane_id来定位其所在的边
            edge = self._road_to_edge[nearest_wp.road_id][nearest_wp.section_id][nearest_wp.lane_id]
        except KeyError:
            pass
        return edge

    def _route_search(self, origin, destination):
        """
        使用A*确定从起点到终点的最优距离
        param:  origin: carla.Location 类型
                destination:
        return: list类型，成员是图中节点id
        """
        start_edge = self._find_location_edge(origin)  # 获取起点所在边
        end_edge = self._find_location_edge(destination)  # 获取终点所在边
        route = self._A_star(start_edge[0], end_edge[0])
        if route is None:  # 如果不可达就报错
            raise nx.NetworkXNoPath(f"Node {start_edge[0]} not reachable from {end_edge[0]}")
        route.append(end_edge[1])  # 可达的话就将终点所在边的右端点加入路径
        return route

    def _A_star(self, n_begin, n_end):
        """
        采用A*算法计算两点之间的最短路径
        param:  n_begin: 起点所在边的左端点id
                n_end:  终点所在边的左端点id
        return: 路径list， 每个元素是图中节点id
        """
        route = []
        open_set = dict()  # 字典， 记录每个节点的父节点和最短路径
        closed_set = dict()
        open_set[n_begin] = (0, -1)  # 每个节点对应一个元组，第一个元素是节点到起点的最短路径，第二个元素是父节点的id

        def cal_heuristic(n):
            # hypot返回原点到一点的多维欧几里得距离
            return math.hypot(self._graph.nodes[n]['vertex'][0] - self._graph.nodes[n_end]['vertex'][0],
                              self._graph.nodes[n]['vertex'][1] - self._graph.nodes[n_end]['vertex'][1],
                              self._graph.nodes[n]['vertex'][2] - self._graph.nodes[n_end]['vertex'][2])

        while 1:
            if len(open_set) == 0:  # 终点不可达
                return None
            # find the node with minimum distance between n_begin in open_set
            c_node = min(open_set, key=lambda n: open_set[n][0] + cal_heuristic(n))
            # print(c_node)
            if c_node == n_end:
                closed_set[c_node] = open_set[c_node]
                del open_set[c_node]  # 如果当前节点是终点，则把该节点从open_set中移除，加入到close_set.
                break
            for suc in self._graph.successors(c_node):  # 处理当前所有节点的后继
                new_cost = self._graph.get_edge_data(c_node, suc)["length"]     # 当前节点到后继节点的cost
                if suc in closed_set:  # 如果访问过就不再访问
                    continue
                elif suc in open_set:  # 如果在即将访问的集合中，判断是否需要更新路径
                    if open_set[c_node][0] + new_cost < open_set[suc][0]:
                        open_set[suc] = (open_set[c_node][0] + new_cost, c_node)
                else:  # 如果是新节点，直接加入open_set中
                    open_set[suc] = (open_set[c_node][0] + new_cost, c_node)
            closed_set[c_node] = open_set[c_node]
            del open_set[c_node]  # 遍历过该节点，则把该节点从open_set中移除，加入到close_set.

        route.append(n_end)
        while 1:
            if closed_set[route[-1]][1] != -1:
                route.append(closed_set[route[-1]][1])  # 通过不断回溯找到最短路径
            else:
                break
        return list(reversed(route))

    @staticmethod
    def _closest_index(current_waypoint, waypoint_list):
        """
        确定waypoint_list中距离当前路点最近的路点的索引值
        param:  current_waypoint:
                waypoint_list:
        return: 整数， 索引值
        """
        min_distance = float('inf')  # 初始情况下设置为最大值
        closest_index = -1
        for i, waypoint in enumerate(waypoint_list):
            distance = waypoint.transform.location.distance(current_waypoint.transform.location)
            if distance < min_distance:
                min_distance = distance
                closest_index = i

        return closest_index

    def routing_search(self, origin, destination)->list:
        """
        得到完整的由waypoint构成的完整路径
        param:  origin: 起点，carla.Location类型
                destination: 终点
        return: list类型，元素是(carla.Waypoint类型, edge["type"]),这里多加了一个边的类型进行输出，
                 是为了后面的更全面考虑某些道路规定的跟车或者超车行为
        """
        route = self._route_search(origin, destination)  # 获取A*的初步规划结果->list
        origin_wp = self._map.get_waypoint(origin)  # type: carla.Waypoint
        destination_wp = self._map.get_waypoint(destination)  # type: carla.Waypoint
        # path_way = []

        # 第一段路径
        edge = self._graph.get_edge_data(route[0], route[1])
        path = [edge["entry_waypoint"]] + edge["path"] + [edge["exit_waypoint"]]
        clos_index = self._closest_index(origin_wp, path)
        for wp in path[clos_index:]:
            self._path_way.append((wp, edge["type"]))

        # 中间路径
        if len(route) > 3:  # 先判断是否有中间路径
            for index in range(1, len(route) - 2):
                edge = self._graph.get_edge_data(route[index], route[index + 1])
                path = edge["path"] + [edge["exit_waypoint"]]  # 每一段路段的终点是下一个路段的起点，所以这里不加起点
                for wp in path:
                    self._path_way.append((wp, edge["type"]))

        # 最后一段路径
        edge = self._graph.get_edge_data(route[-2], route[-1])
        # print(edge)
        path = edge["path"] + [edge["exit_waypoint"]]
        clos_index = self._closest_index(destination_wp, path)
        if clos_index != 0:  # 判断终点是否是当前路段的起点，如果不是，将后续的路点加入path_way;
            for wp in path[:clos_index + 1]:
                self._path_way.append((wp, edge["type"]))
        else:  # 如果是，后面的路段终点则在上个路段已经添加进path_way中，这里不进行重复操作
            pass
        self._start_loc = origin
        self._end_loc = destination
        self._has_routing = True
        return self._path_way
    
    def draw_routing_result(self, plt):
        if not self._has_routing:
            raise RuntimeError("Please call routing first.")
        vertexs = nx.get_node_attributes(self._graph, "vertex")
        # print("vertex", vertexs)
        # 将位置信息提取出来，字典类型{node: (x, y)}
        pos = {}
        for node in vertexs.keys():
            pos.update({node: vertexs[node][0:2]})
        # print(pos)
        # path_way = self.search_path_way(origin, destination)
        way_points = {'x': [], 'y': []} # 用于存储路径点的x和y坐标
        for waypoint in self._path_way:
            # print(f'---68---waypoint = {waypoint}')
            way_points['x'].append(waypoint[0].transform.location.x)
            way_points['y'].append(waypoint[0].transform.location.y)
            
        plt.figure(1)
        plt.title("DTPP_Routing", fontsize=20, color='b', backgroundcolor='white', fontweight='bold', style='italic', family='Times New Roman', loc='center', pad=20, rotation=0)
        nx.draw_networkx(self._graph, pos=pos, arrows=True, with_labels=True, node_size=50, width=1, font_size=12)
        nx.draw_networkx_nodes(self._graph, pos=pos, nodelist=self._route_search(self._start_loc, self._end_loc), node_size=40, label="test_nodes", node_color='r')
        
        plt.scatter(way_points['x'], way_points['y'], c='b', marker='o', s=20)
        # 着重画出起点和终点
        plt.plot(self._start_loc.x, self._start_loc.y, 'ro', label="start", markersize=20)
        plt.plot(self._end_loc.x, self._end_loc.y, 'yo', label="end", markersize=20)
        # plt.show()
        plt.savefig('DTPP_Routing.png')
        
    def draw_all_spawn_points(self, plt, All_spawn_points):
        debug_all_spawn_points = {'x': [], 'y': []}
        for i, tsf in enumerate(All_spawn_points):
            # world.debug.draw_string(i.location, 'X', color=carla.Color(r=255, g=0, b=0), life_time=100)
            debug_all_spawn_points['x'].append(tsf.location.x)
            debug_all_spawn_points['y'].append(tsf.location.y)
            plt.text(tsf.location.x+0.5, tsf.location.y+0.5, str(i), fontsize=8, color='r')
        plt.scatter(debug_all_spawn_points['x'], debug_all_spawn_points['y'], s=40, c='r', marker='x', label='All spawn points')