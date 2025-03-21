import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import matplotlib as mpl
# 用bokeh画出 trimmed paths
# from bokeh.plotting import figure
import bokeh.plotting as bplt
from bokeh.models import Arrow, OpenHead, Text, ColumnDataSource, Label
# from bokeh.io import output_notebook  # 如果在 Jupyter Notebook 中使用

import numpy as np
from typing import List
import random

from nuplan.common.actor_state.state_representation import Point2D


import carla

from agents.dtpp_common.dtpp_map import DtppMap
from agents.dtpp_common.dtpp_planner_utils import get_vehicle_params_from_actor
from planner_utils import trajectory_smoothing

colors = ["blue", "green", "red", "orange", "purple", "brown"]

class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DtppDebuger(metaclass=SingletonMeta):
    def __init__(self, bokeh: bool = True):
        self._use_bokeh = bokeh
        if self._use_bokeh:
            self._set_bokeh_p()
        else:
            self._set_plt()

    def _set_bokeh_p(self):
        self._p = bplt.figure(
            title="Dtpp Map",
            width=1280,
            height=1080,
            tools="pan,wheel_zoom,box_zoom,reset,save",
            match_aspect=True,
        )
        self._p.grid.visible = True
        self._p.xaxis.axis_label = "X"
        self._p.yaxis.axis_label = "Y"

    def _set_plt(self):
        plt.figure(figsize=(10, 10))
        plt.axis("equal")
        plt.grid()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.yaxis.set_major_locator(MultipleLocator(10))

    def draw_dtpp_map(self, actor, dtpp_map, trajectory=None):
        if not self._use_bokeh:
            self._draw_map_top_matplotlib(dtpp_map=dtpp_map)
        else:
            self._draw_map_top_bokeh(dtpp_map=dtpp_map, actor = actor, trajectory=trajectory)

    def _draw_map_top_matplotlib(self, dtpp_map, vehicle:carla.Actor=None):
        # from tmp_test.test_2_road_graph_and_routing import draw_map
        routing = dtpp_map._routing
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
        self._draw_topology(dtpp_map._topology)
        # import time
        # plt.savefig(f'./routing_{time.time()}.png')
        # plt.show()

    def show(self):
        if self._use_bokeh:
            bplt.show(self._p)
        else:
            plt.show()


    def _draw_topology(self, topology):
        for lane in topology:
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
    def _draw_trajectory(self, trajectory: List[carla.Transform], color="pink"):
        trj_x = [tsf.location.x for tsf in trajectory]
        trj_y = [tsf.location.y for tsf in trajectory]
        self._p.scatter(trj_x, trj_y, size=5, color=color, alpha=0.5, marker="o", line_width=20)
        # self._p.line(trj_x, trj_y, line_width=2, color=color)
        
    def _draw_vehicle(self, actor: carla.Actor):
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
        # p = bplt.figure(title="Vehicle Bounding Box", x_axis_label='X', y_axis_label='Y', width=800, height=600)

        # 绘制 Bounding Box 的边
        for edge in edges:
            self._p.line(
                x=[x[edge[0]], x[edge[1]]],
                y=[y[edge[0]], y[edge[1]]],
                line_width=2,
                line_color="blue"
            )

    def _draw_map_top_bokeh(self, dtpp_map: DtppMap, actor: carla.Actor=None, trajectory=None):

        routing = dtpp_map._routing
        # 绘制路由点
        route_x = [wp_road_opt[0].transform.location.x for wp_road_opt in routing]
        route_y = [wp_road_opt[0].transform.location.y for wp_road_opt in routing]
        self._p.scatter(route_x, route_y, size=5, color="yellow", alpha=0.5)


        # 定义颜色列表，用于区分不同的路径
        
        
        # 绘制起点和终点
        self._p.scatter(
            [routing[0][0].transform.location.x],
            [routing[0][0].transform.location.y],
            size=15,
            color="red",
            marker="circle",
        )
        self._p.scatter(
            [routing[-1][0].transform.location.x],
            [routing[-1][0].transform.location.y],
            size=15,
            color="green",
            marker="circle",
        )

        # 绘制拓扑结构
        self._draw_topology_bokeh(topology=dtpp_map._topology)
        # 绘制车辆
        self._draw_vehicle(actor=actor)
        # 绘制候选车道线
        if actor:
            self.plot_candiate_lanes_bokeh(actor, dtpp_map, colors)
        if trajectory:
            self._draw_trajectory(trajectory)
        # show(self._p)

    def plot_candiate_lanes_bokeh(self, actor, dtpp_map, colors):
        trim_lanes = dtpp_map.get_candidate_lanes(actor)
        # 遍历 trimmed_paths 并绘制每条路径
        for idx, path in enumerate(trim_lanes):
            x = path[2][:, 0].tolist()
            y = path[2][:, 1].tolist()
            # self._p.line(
            #     x,
            #     y,
            #     line_width=2,
            #     color=colors[idx % len(colors)],
            #     legend_label=f"Path {idx+1}",
            # )
            self._p.scatter(x, y, size=5, color=colors[idx % len(colors)], alpha=0.5)

        # 添加图例位置
        self._p.legend.location = "top_left"

    def _draw_topology_bokeh(self, topology):


        for lane in topology:
            lane_line = [
                (wp.transform.location.x, wp.transform.location.y)
                for wp in lane["path"]
            ]
            color = "red" if lane["entry"].is_junction else "blue"
            line_type = "dashed" if lane["entry"].lane_id < 0 else "solid"

            # 绘制车道线
            self._p.multi_line(
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

            self._p.add_layout(
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
            self._p.add_layout(
                Label(
                    x=start_x + 0.5,
                    y=start_y + 0.5,
                    text=f"s{lane['entry'].section_id}.r{lane['entry'].road_id}\nl{lane['entry'].lane_id}.j{lane['entry'].junction_id}",
                    text_font_size="14px",
                    text_color=color,
                    text_font_style="bold",
                )
            )

    def plot_generated_paths(self, g_paths):
        print(f"g_path.shape:{g_paths}")
        xs, ys = [], []
        for i, path in enumerate(g_paths):
            x_list = [pt[0] for pt in path]
            y_list = [pt[1] for pt in path]
            xs.append(x_list)
            ys.append(y_list)

            color = colors[i % len(colors)]
            self._p.add_layout(
                Label(
                    x=x_list[-1] + 0.5*random.random(),
                    y=y_list[-1] + 0.5*random.random(),
                    text=str(i),
                    text_font_size="24px",
                    text_color=color,
                    text_font_style="bold",
                )
            )

            
            line_type = 'dashed'
            self._p.multi_line(
                xs,
                ys,
                line_color=color,
                line_dash="dotdash" if line_type == "dashed" else "solid",
                line_width=1,
            )

        # color = 'red'
        # line_type = 'dashed'
        # self._p.multi_line(
        #     xs,
        #     ys,
        #     line_color=color,
        #     line_dash="dotdash" if line_type == "dashed" else "solid",
        #     line_width=1,
        # )

    
    @staticmethod
    def plot(self, iteration, env_inputs, ego_future, agents_future, vehicle_param):
        fig = plt.gcf()
        dpi = 100
        size_inches = 800 / dpi
        fig.set_size_inches([size_inches, size_inches])
        fig.set_dpi(dpi)
        fig.set_tight_layout(True)

        # plot map
        map_lanes = env_inputs['map_lanes'][0]
        for i in range(map_lanes.shape[0]):
            lane = map_lanes[i].cpu().numpy()
            if lane[0, 0] != 0:
                plt.plot(lane[:, 0], lane[:, 1], color="gray", linewidth=20, zorder=1)
                plt.plot(lane[:, 0], lane[:, 1], "k--", linewidth=1, zorder=2)

        map_crosswalks = env_inputs['map_crosswalks'][0]
        for crosswalk in map_crosswalks:
            pts = crosswalk.cpu().numpy()
            plt.plot(pts[:, 0], pts[:, 1], 'b:', linewidth=2)

        ## plot ego vehicle_param
        # front_length = get_vehicle_params_from_actor().front_length
        # rear_length = get_vehicle_params_from_actor().rear_length
        # width = get_vehicle_params_from_actor().width
        front_length = vehicle_param.front_length
        rear_length = vehicle_param.rear_length
        width = vehicle_param.width
        rect = plt.Rectangle((0 - rear_length, 0 - width/2), front_length + rear_length, width, 
                             linewidth=2, color='r', alpha=0.9, zorder=3)
        plt.gca().add_patch(rect)

        # plot agents
        agents = env_inputs['neighbor_agents_past'][0]
        for agent in agents:
            agent = agent[-1].cpu().numpy()
            if agent[0] != 0:
                rect = plt.Rectangle((agent[0] - agent[6]/2, agent[1] - agent[7]/2), agent[6], agent[7],
                                      linewidth=2, color='m', alpha=0.9, zorder=3,
                                      transform=mpl.transforms.Affine2D().rotate_around(*(agent[0], agent[1]), agent[2]) + plt.gca().transData)
                plt.gca().add_patch(rect)
                                    

        # plot ego and agents future trajectories
        ego = ego_future.cpu().numpy()
        agents = agents_future.cpu().numpy()
        plt.plot(ego[:, 0], ego[:, 1], color="r", linewidth=3)
        plt.gca().add_patch(plt.Circle((ego[29, 0], ego[29, 1]), 0.5, color="r", zorder=4))
        plt.gca().add_patch(plt.Circle((ego[79, 0], ego[79, 1]), 0.5, color="r", zorder=4))

        for agent in agents:
            if np.abs(agent[0, 0]) > 1:
                agent = trajectory_smoothing(agent)
                plt.plot(agent[:, 0], agent[:, 1], color="m", linewidth=3)
                plt.gca().add_patch(plt.Circle((agent[29, 0], agent[29, 1]), 0.5, color="m", zorder=4))
                plt.gca().add_patch(plt.Circle((agent[79, 0], agent[79, 1]), 0.5, color="m", zorder=4))

        # plot
        plt.gca().margins(0)  
        plt.gca().set_aspect('equal')
        plt.gca().axes.get_yaxis().set_visible(False)
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axis([-50, 50, -50, 50])
        plt.show()

#     from bokeh.plotting import figure, show
# from bokeh.models import ColumnDataSource, Rect, Circle, MultiLine
# from bokeh.io import output_notebook
# import numpy as np
    @staticmethod
    def plot_bokeh(self, iteration, env_inputs, ego_futures, agents_future, vehicle_param):
        # 输出到 Notebook（可选）
        # output_notebook()

        # 创建 Bokeh 图形
        p = bplt.figure(
            title=f"Iteration {iteration}",
            width=800,
            height=800,
            x_range=(-50, 50),
            y_range=(-50, 50),
            tools="pan,wheel_zoom,box_zoom,reset",
            match_aspect=True
        )

        # 隐藏坐标轴
        p.axis.visible = False
        p.xgrid.visible = False
        p.ygrid.visible = False


        for i, traj in enumerate(ego_futures):
            self.plot_single_traj_bokeh(env_inputs, traj, agents_future, vehicle_param, p)

        # 显示图形
        bplt.show(p)

    @staticmethod
    def plot_single_traj_bokeh(env_inputs, ego_future, agents_future, vehicle_param, p):
        # ------------------------------
        # 绘制地图车道
        # ------------------------------
        map_lanes = env_inputs['map_lanes'][0]
        for i in range(map_lanes.shape[0]):
            lane = map_lanes[i].cpu().numpy()
            if lane[0, 0] != 0:
                # 绘制车道中心线
                p.line(
                    lane[:, 0], lane[:, 1],
                    line_width=1,
                    line_dash="dashed",
                    line_color="black",
                    line_alpha=0.5
                )
                # 绘制车道区域
                p.multi_line(
                    xs=[lane[:, 0]],
                    ys=[lane[:, 1]],
                    line_width=20,
                    line_color="gray",
                    line_alpha=0.3
                )

        # ------------------------------
        # 绘制人行横道
        # ------------------------------
        map_crosswalks = env_inputs['map_crosswalks'][0]
        for crosswalk in map_crosswalks:
            pts = crosswalk.cpu().numpy()
            p.line(
                pts[:, 0], pts[:, 1],
                line_width=2,
                line_dash="dotted",
                line_color="blue",
                line_alpha=0.5
            )

        # ------------------------------
        # 绘制自车
        # ------------------------------
        front_length = vehicle_param.front_length
        rear_length = vehicle_param.rear_length
        width = vehicle_param.width

        # 自车矩形
        p.rect(
            x=0 - rear_length + (front_length + rear_length) / 2,
            y=0,
            width=front_length + rear_length,
            height=width,
            angle=0,
            fill_color="red",
            fill_alpha=0.9,
            line_color="black",
            line_width=2
        )

        # ------------------------------
        # 绘制其他车辆
        # ------------------------------
        agents = env_inputs['neighbor_agents_past'][0]
        for agent in agents:
            agent = agent[-1].cpu().numpy()
            if agent[0] != 0:
                p.rect(
                    x=agent[0],
                    y=agent[1],
                    width=agent[6],
                    height=agent[7],
                    angle=agent[2],
                    fill_color="magenta",
                    fill_alpha=0.9,
                    line_color="black",
                    line_width=2
                )

        # ------------------------------
        # 绘制自车和其他车辆的未来轨迹
        # ------------------------------
        ego = ego_future.cpu().numpy()
        agents = agents_future.cpu().numpy()

        # 自车轨迹
        p.line(
            ego[:, 0], ego[:, 1],
            line_width=3,
            line_color="red"
        )
        p.circle(
            x=[ego[29, 0], ego[79, 0]],
            y=[ego[29, 1], ego[79, 1]],
            size=10,
            fill_color="red",
            line_color="black"
        )

        # 其他车辆轨迹
        for agent in agents:
            if np.abs(agent[0, 0]) > 1:
                agent = trajectory_smoothing(agent)
                p.line(
                    agent[:, 0], agent[:, 1],
                    line_width=3,
                    line_color="magenta"
                )
                p.circle(
                    x=[agent[29, 0], agent[79, 0]],
                    y=[agent[29, 1], agent[79, 1]],
                    size=10,
                    fill_color="magenta",
                    line_color="black"
                )






# # 创建单例实例
# s1 = MySingleton()
# s2 = MySingleton()

# # 验证是否为同一个实例
# print(s1 is s2)  # 输出: True

# # 调用单例实例的方法
# s1.increment()
# print(s2.value)  # 输出: 1
