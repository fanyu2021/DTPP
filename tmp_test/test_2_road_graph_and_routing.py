#   -*- coding: utf-8 -*-
# @Author  : Xiaoya Liu
# @File    : test_2.py

"""
完成基于graph进行节点级别的初步全局规划，以及路点级别的路径规划
"""

import carla
from carla_planner.global_planning import global_path_planner
import networkx as nx
import matplotlib.pyplot as plt

client = carla.Client("localhost", 2000)
client.set_timeout(20)
vailable_maps = client.get_available_maps()
print(f"Available maps in CARLA:{vailable_maps}")
# 对象创建好了之后，在对象中添加需要的环境中的地图
world = client.load_world('/Game/Carla/Maps/Town05')
amap = world.get_map()  # type: carla.Map

# topo = amap.get_topology()
# print(len(topo))
# print(topo)
def draw_map(world, amap):
    global_route_plan = global_path_planner(world_map=amap, sampling_resolution=1)
    topology, graph, id_map, road_id_to_edge = global_route_plan.get_topology_and_graph_info()
# print(len(topology))
# print(topology.nodes)

# print("graph.edges", len(graph.nodes))
# print(graph.nodes)
# print("graph.edges", len(graph.edges))
# print(graph.edges)

# print("id_map", len(id_map))
# print("id_map", id_map)
# print("road_id_to_edge", len(road_id_to_edge))
# print("road_id_to_edge", road_id_to_edge)
# print(nx.get_node_attributes(graph, ""))

#     start_id = 20
#     end_id = 15
#     print(f'start_location = {graph.nodes[start_id]["vertex"]}')
#     print(f'end_location = {graph.nodes[end_id]["vertex"]}')
#     route = global_route_plan._route_search(origin=carla.Location(graph.nodes[start_id]['vertex'][0],
#                                                               graph.nodes[start_id]['vertex'][1],
#                                                               graph.nodes[start_id]['vertex'][2]),
#                                         destination=carla.Location(graph.nodes[end_id]['vertex'][0],
#                                                                    graph.nodes[end_id]['vertex'][1],
#                                                                    graph.nodes[end_id]['vertex'][2])
#                                         )
# # print(f'---route---:{route}')

#     pathway = global_route_plan.search_path_way(origin=carla.Location(graph.nodes[start_id]['vertex'][0],
#                                                                   graph.nodes[start_id]['vertex'][1],
#                                                                   graph.nodes[start_id]['vertex'][2]),
#                                             destination=carla.Location(graph.nodes[end_id]['vertex'][0],
#                                                                        graph.nodes[end_id]['vertex'][1],
#                                                                        graph.nodes[end_id]['vertex'][2])
                                            # )
    # debug = world.debug  # type: carla.DebugHelper
    # way_points = {'x': [], 'y': []} # 用于存储路径点的x和y坐标
    # for waypoint in pathway:
    # # print(f'---68---waypoint = {waypoint}')
    #     way_points['x'].append(waypoint[0].transform.location.x)
    #     way_points['y'].append(waypoint[0].transform.location.y)
    #     debug.draw_point(waypoint[0].transform.location + carla.Location(0, 0, 2),
    #                  size=0.05, color=carla.Color(0, 255, 0), life_time=0)
# 获取图中指定属性的节点信息，得到一个字典，key是节点id，value是节点指定的属性
    vertexs = nx.get_node_attributes(graph, "vertex")
# print("vertex", vertexs)
# 将位置信息提取出来，字典类型{node: (x, y)}
    pos = {}
    for node in vertexs.keys():
        pos.update({node: vertexs[node][0:2]})
# print(pos)

    # plt.figure(1)
    # plt.title("test_graph")
    nx.draw_networkx(graph, pos=pos, arrows=True, with_labels=True, node_size=50, width=1, font_size=12)
    # nx.draw_networkx_nodes(graph, pos=pos, nodelist=route, node_size=40, label="test_nodes", node_color='r')

#     plt.scatter(way_points['x'], way_points['y'], c='b', marker='.', s=10)
# # 着重画出起点和终点
#     plt.plot(graph.nodes[start_id]['vertex'][0], graph.nodes[start_id]['vertex'][1], 'ro', label="start", markersize=20)
#     plt.plot(graph.nodes[end_id]['vertex'][0], graph.nodes[end_id]['vertex'][1], 'yo', label="end", markersize=20)

# 画出所有可以生成 actor 的位置
    # spawn_points = amap.get_spawn_points()
    # for i in range(len(spawn_points)):
    #     x = spawn_points[i].location.x
    #     y = spawn_points[i].location.y
    #     plt.scatter(x, y, s=10, c='r')
    #     plt.text(x, y, i)

#     plt.figure(2)
#     plt.title("test_nodes")
# # This draws only the nodes of the graph G.
#     nx.draw_networkx_nodes(graph, pos=pos, node_size=20, label="test_nodes")

#     plt.figure(3)
#     plt.title("test_edges")
# # This draws only the nodes of the graph G.
#     nx.draw_networkx_edges(graph, pos=pos, node_size=20, label="test_edges")

draw_map(world, amap)
plt.show()

# topology中的数据结构
# {'entry': <carla.libcarla.Waypoint object at 0x000001D05B7D9450>,
# 'exit': <carla.libcarla.Waypoint object at 0x000001D05B8E1270>,
# 'entryxyz': (53.0, 188.0, 0.0),
# 'exitxyz': (35.0, 188.0, 0.0),
# 'path': [<carla.libcarla.Waypoint object at 0x000001D05B8F9870>,
# <carla.libcarla.Waypoint object at 0x000001D05B8F98D0>, <carla.libcarla.Waypoint object at 0x000001D05B8F9930>,
# <carla.libcarla.Waypoint object at 0x000001D05B8F9990>, <carla.libcarla.Waypoint object at 0x000001D05B8F99F0>,
# <carla.libcarla.Waypoint object at 0x000001D05B8F9A50>, <carla.libcarla.Waypoint object at 0x000001D05B8F9AB0>]},

# {'entry': <carla.libcarla.Waypoint object at 0x000001D05B8E1330>,
# 'exit': <carla.libcarla.Waypoint object at 0x000001D05B8E1390>,
# 'entryxyz': (46.0, 228.0, 0.0),
# 'exitxyz': (46.0, 199.0, 0.0),
# 'path': [<carla.libcarla.Waypoint object at 0x000001D05B8F9810>,
# <carla.libcarla.Waypoint object at 0x000001D05B8F9B10>, <carla.libcarla.Waypoint object at 0x000001D05B8F9750>,
# <carla.libcarla.Waypoint object at 0x000001D05B8F9C30>, <carla.libcarla.Waypoint object at 0x000001D05B8F9C90>,
# <carla.libcarla.Waypoint object at 0x000001D05B8F9CF0>, <carla.libcarla.Waypoint object at 0x000001D05B8F9D50>,
# <carla.libcarla.Waypoint object at 0x000001D05B8F9DB0>, <carla.libcarla.Waypoint object at 0x000001D05B8F9E10>,
# <carla.libcarla.Waypoint object at 0x000001D05B8F9E70>, <carla.libcarla.Waypoint object at 0x000001D05B8F9ED0>,
# <carla.libcarla.Waypoint object at 0x000001D05B8F9F30>, <carla.libcarla.Waypoint object at 0x000001D05B8F9F90>]}

#  graph中节点中的属性
# {0： {'vertex': (53.0, 188.0, 0.0)}}

# graph中边中的属性
# (0, 1,
# {'length': 8,
# 'path': [<carla.libcarla.Waypoint object at 0x0000010DD4C2F930>, <carla.libcarla.Waypoint object at 0x0000010DD4C2F990>,
# <carla.libcarla.Waypoint object at 0x0000010DD4C2F9F0>, <carla.libcarla.Waypoint object at 0x0000010DD4C2FA50>,
# <carla.libcarla.Waypoint object at 0x0000010DD4C2FAB0>, <carla.libcarla.Waypoint object at 0x0000010DD4C2FB10>,
# <carla.libcarla.Waypoint object at 0x0000010DD4C2FB70>],
# 'entry_waypoint': <carla.libcarla.Waypoint object at 0x0000010DD4C0F330>,
# 'exit_waypoint': <carla.libcarla.Waypoint object at 0x0000010DD4C0F3F0>,
# 'entry_vector': array([-9.99999881e-01, -4.59106755e-04,  1.74845553e-07]),
# 'exit_vector': array([-9.99999881e-01, -4.58327908e-04,  1.74845553e-07]),
# 'net_vector': [-0.9999998948377721, -0.00045861143134230935, 0.0],
# 'intersection': True,
# 'type': <RoadOption.LANEFOLLOW: 4>})
