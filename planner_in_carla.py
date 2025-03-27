import torch
import scipy
import random
import itertools
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
from shapely.geometry.base import CAP_STYLE
from path_planner import calc_spline_course
from bezier_path import calc_4points_bezier_path
from collections import defaultdict
from spline_planner import SplinePlanner
from torch.nn.utils.rnn import pad_sequence
from scenario_tree_prediction import *
from planner_utils import *
# from nuplan.planning.simulation.observation.idm.utils import path_to_linestring
from typing import List

import carla

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType, STATIC_OBJECT_TYPES

from agents.dtpp_common.features_adapter import get_ego_state_list_from_actor
from agents.dtpp_common.dtpp_planner_utils import get_vehicle_params_from_actor
from custom_format import *
logger = create_colored_logger(name=__name__)

from debug.dtpp_debug import DtppDebuger
from debug.world_debuger import WorldDebuger


def path_to_linestring(path: List[EgoState]) -> LineString:
    """
    Converts a List of StateSE2 into a LineString
    :param path: path to be converted
    :return: LineString.
    """
    return LineString([(point.x, point.y) for point in path])


class TrajTree:
    def __init__(self, traj, parent, depth):
        self.traj = traj
        self.state = traj[-1, :5]
        self.children = list()
        self.parent = parent
        self.depth = depth
        self.attribute = dict()
        if parent is not None:
            self.total_traj = torch.cat((parent.total_traj, traj), 0)
        else:
            self.total_traj = traj

    def expand(self, child):
        self.children.append(child)

    def expand_set(self, children):
        self.children += children

    def expand_children(self, paths, horizon, speed_limit, planner):
        trajs = planner.gen_trajectories(self.state, horizon, paths, speed_limit, self.isroot())
        if trajs is None:
            return False
        children = [TrajTree(traj, self, self.depth + 1) for traj in trajs]
        self.expand_set(children)
        return True

    def isroot(self):
        return self.parent is None

    def isleaf(self):
        return len(self.children) == 0

    def get_subseq_trajs(self):
        return [child.traj for child in self.children]
    
    def get_all_leaves(self, leaf_set=[]):
        if self.isleaf():
            print(self.state)
            leaf_set.append(self)
        else:
            for child in self.children:
                leaf_set = child.get_all_leaves(leaf_set)

        return leaf_set

    @staticmethod
    def get_children(obj):
        if isinstance(obj, TrajTree):
            return obj.children
        
        elif isinstance(obj, list):
            children = [node.children for node in obj]
            children = list(itertools.chain.from_iterable(children))
            return children
        
        else:
            raise TypeError("obj must be a TrajTree or a list")

    def plot_tree(self, ax=None, msize=12):
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 10))
        state = self.state.cpu().detach().numpy()
        
        ax.plot(state[0], state[1], marker="o", color="b", markersize=msize)

        if self.traj.shape[0] > 1:
            if self.parent is not None:
                traj_l = torch.cat((self.parent.traj[-1:],self.traj),0)
                traj = traj_l.cpu().detach().numpy()
            else:
                traj = self.traj.cpu().detach().numpy()

            ax.plot(traj[:, 0], traj[:, 1], color="k")

        for child in self.children:
            child.plot_tree(ax)

        return ax

    @staticmethod
    def get_children_index_torch(nodes):
        indices = dict()
        for depth, nodes_d in nodes.items():
            if depth+1 in nodes:
                childs_d = nodes[depth+1]
                indices_d = list()
                for node in nodes_d:
                    indices_d.append(torch.tensor([childs_d.index(child) for child in node.children]))
                indices[depth] = pad_sequence(indices_d, batch_first=True, padding_value=-1)

        return indices
    
    @staticmethod
    def get_nodes_by_level(obj, depth, nodes=None, trim_short_branch=True):
        assert obj.depth <= depth
        if nodes is None:
            nodes = defaultdict(lambda: list())

        if obj.depth == depth:
            nodes[depth].append(obj)

            return nodes, True
        else:
            if obj.isleaf():
                return nodes, False
            else:
                flag = False
                children_flags = dict()
                for child in obj.children:
                    nodes, child_flag = TrajTree.get_nodes_by_level(child, depth, nodes)
                    children_flags[child] = child_flag
                    flag = flag or child_flag

                if trim_short_branch:
                    obj.children = [child for child in obj.children if children_flags[child]]
                if flag:
                    nodes[obj.depth].append(obj)

                return nodes, flag


class CarlaTreePlanner:
    def __init__(self, device, encoder, decoder, n_candidates_expand=5, n_candidates_max=30):
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.max_path_len = 120 # [m]
        self.target_depth = MAX_LEN # [m]
        self.target_speed = 15 # [m/s]
        self.horizon = 8 # [s]
        self.first_stage_horizon = 3 # [s]
        self.n_candidates_expand = n_candidates_expand # second stage
        self.n_candidates_max = n_candidates_max # max number of candidates
        self.planner = SplinePlanner(self.first_stage_horizon, self.horizon) 
        self.dtpp_debuger = DtppDebuger()


    def generate_paths(self, routes):
        ego_state = self.ego_state.rear_axle.x, self.ego_state.rear_axle.y, self.ego_state.rear_axle.heading
        
        # generate paths
        new_paths = []
        path_distance = []
        for (path_len, dist, path_polyline) in routes:
            if len(path_polyline) > 81:
                sampled_index = np.array([5, 10, 15, 20]) * 4
            elif len(path_polyline) > 61:
                sampled_index = np.array([5, 10, 15]) * 4
            elif len(path_polyline) > 41:
                sampled_index = np.array([5, 10]) * 4
            elif len(path_polyline) > 21:
                sampled_index = [20]
            else:
                sampled_index = [1]
     
            target_states = path_polyline[sampled_index].tolist()
            for j, state in enumerate(target_states):
                first_stage_path = calc_4points_bezier_path(ego_state[0], ego_state[1], ego_state[2], 
                                                            state[0], state[1], state[2], 3, sampled_index[j])[0]
                second_stage_path = path_polyline[sampled_index[j]+1:, :2]
                path_polyline = np.concatenate([first_stage_path, second_stage_path], axis=0)
                new_paths.append(path_polyline)  
                path_distance.append(dist)   
        logger.debug(f'--- new_paths size:{len(new_paths)}, len(dist):{len(path_distance)}')
        # evaluate paths
        candiate_paths = {}
        for path, dist in zip(new_paths, path_distance):
            cost = self.calculate_cost(path, dist)
            candiate_paths[cost] = path

        # sort paths by cost
        candidate_paths = []
        nums = len(candiate_paths)
        # nums = 3
        for cost in sorted(candiate_paths.keys())[:nums]:
            path = candiate_paths[cost]
            path = self.post_process(path)
            candidate_paths.append(path)
        logger.debug(f'--- candidate_paths size:{len(candidate_paths)}')
        return candidate_paths
    
    def calculate_cost(self, path, dist):
        # path curvature
        curvature = self.calculate_path_curvature(path[0:100])
        curvature = np.max(curvature)

        # lane change
        lane_change = dist

        # check obstacles
        obstacles = self.check_obstacles(path[0:100:10], self.obstacles)
        
        # final cost
        cost = 10 * obstacles + 1 * lane_change  + 0.1 * curvature

        return cost

    def post_process(self, path):
        path = self.transform_to_ego_frame(path)
        index = np.arange(0, len(path), 10)
        x = path[:, 0][index]
        y = path[:, 1][index]

        # spline interpolation
        rx, ry, ryaw, rk = calc_spline_course(x, y)
        spline_path = np.stack([rx, ry, ryaw, rk], axis=1)
        ref_path = spline_path[:self.max_path_len*10]

        return ref_path

    def depth_first_search(self, starting_edge, depth=0):
        if depth >= self.target_depth:
            return [[starting_edge]]
        else:
            traversed_edges = []
            child_edges = [edge for edge in starting_edge.outgoing_edges if edge.id in self.candidate_lane_edge_ids]

            if child_edges:
                for child in child_edges:
                    edge_len = len(child.baseline_path.discrete_path) * 0.25
                    traversed_edges.extend(self.depth_first_search(child, depth+edge_len))

            if len(traversed_edges) == 0:
                return [[starting_edge]]

            edges_to_return = []

            for edge_seq in traversed_edges:
                edges_to_return.append([starting_edge] + edge_seq)
                    
            return edges_to_return

    @staticmethod
    def calculate_path_curvature(path):
        dx = np.gradient(path[:, 0])
        dy = np.gradient(path[:, 1])
        d2x = np.gradient(dx)
        d2y = np.gradient(dy)
        curvature = np.abs(dx * d2y - d2x * dy) / (dx**2 + dy**2)**(3/2)

        return curvature

    @staticmethod
    def check_obstacles(path, obstacles):
        expanded_path = LineString(path).buffer((WIDTH/2), cap_style=CAP_STYLE.square)

        for obstacle in obstacles:
            obstacle_polygon = obstacle.geometry
            if expanded_path.intersects(obstacle_polygon):
                return 1

        return 0

    def predict(self, encoder_outputs, traj_inputs, agent_states, timesteps):
        ego_trajs = torch.zeros((self.n_candidates_max, self.horizon*10, 6)).to(self.device)
        for i, traj in enumerate(traj_inputs):
            ego_trajs[i, :len(traj)] = traj[..., :6].float()

        ego_trajs = ego_trajs.unsqueeze(0)
        agent_trajs, scores, _, _ = self.decoder(encoder_outputs, ego_trajs, agent_states, timesteps)

        return agent_trajs, scores
    
    def transform_to_ego_frame(self, path):
        x = path[:, 0] - self.ego_state.rear_axle.x
        y = path[:, 1] - self.ego_state.rear_axle.y
        x_e = x * np.cos(-self.ego_state.rear_axle.heading) - y * np.sin(-self.ego_state.rear_axle.heading)
        y_e = x * np.sin(-self.ego_state.rear_axle.heading) + y * np.cos(-self.ego_state.rear_axle.heading)
        path = np.column_stack([x_e, y_e])
        return path
    

    def plan(self, iteration, dtpp_map, vehicle: carla.Actor, env_inputs, candidate_lanes, traffic_light, observation, debug=False):
        start_time_ms = time.time()*1e3 # ms
        self._world_debuger = WorldDebuger(actor=vehicle)
        time_usedict_ms = {"r1": 0, "p1": 0, "r2": 0, "p2": 0, "cs": 0}
        # get environment information
        self.ego_state = get_ego_state_list_from_actor(0, ego=vehicle)
        # self.candidate_lane_edge_ids = [lane.id for lane in candidate_lanes]
        # self.route_roadblocks = route_roadblocks
        self.traffic_light = traffic_light
        object_types = [TrackedObjectType.VEHICLE, TrackedObjectType.BARRIER,
                        TrackedObjectType.CZONE_SIGN, TrackedObjectType.TRAFFIC_CONE,
                        TrackedObjectType.GENERIC_OBJECT]
        objects = observation.tracked_objects.get_tracked_objects_of_types(object_types)
        self.obstacles = []
        for obj in objects:
            if obj.tracked_object_type == TrackedObjectType.VEHICLE:
                if obj.velocity.magnitude() < 0.1:
                    self.obstacles.append(obj.box)
            else:
                self.obstacles.append(obj.box)

        # initial tree (root node)
        # x, y, heading, velocity, acceleration, curvature, time
        state = torch.tensor([[0, 0, 0, # x, y, heading 
                               self.ego_state.dynamic_car_state.rear_axle_velocity_2d.x,
                               self.ego_state.dynamic_car_state.rear_axle_acceleration_2d.x, 0, 0]], dtype=torch.float32)
        tree = TrajTree(state, None, 0)

        # environment encoding
        encoder_outputs = self.encoder(env_inputs)
        agent_states = env_inputs['neighbor_agents_past']

        # logger.debug(f'encoder_outputs: {encoder_outputs}')

        near_lanes = dtpp_map.get_candidate_lanes(vehicle)
        # logger.debug(f'candidate_paths: {candidate_paths}')
        candidate_paths = self.generate_paths(near_lanes)
        self._world_debuger.plot_generated_paths(candidate_paths, actor=vehicle)
        candidate_paths = candidate_paths[:3]
        if len(candidate_paths) == 0:
            logger.error('No candidate paths!!!')

        
        
        # self.dtpp_debuger.draw_dtpp_map(actor=vehicle, dtpp_map=dtpp_map)
        # self.dtpp_debuger.plot_generated_paths(candidate_paths, actor=vehicle)
        # self.dtpp_debuger.show()

        self._world_debuger.plot_candiate_lanes(dtpp_map=dtpp_map)
        

        # self.speed_limit = edges[0].speed_limit_mps or self.target_speed # TODO(fanyu): 道路限速
        self.speed_limit = self.target_speed # TODO(fanyu): 道路限速
        
        # expand tree
        tree.expand_children(candidate_paths, self.first_stage_horizon, self.speed_limit, self.planner)
        leaves = TrajTree.get_children(tree)
        # self.dtpp_debuger.plot_tree_bokeh(tree, actor=vehicle)
        

        # query the model
        parent_scores = {}
        trajs = [leaf.total_traj[1:] for leaf in leaves]
        time_usedict_ms["r1"] = time.time()*1e3 - start_time_ms
        agent_trajectories, scores = self.predict(encoder_outputs, trajs, agent_states, self.first_stage_horizon*10)
        time_usedict_ms["p1"] = time.time()*1e3 - (start_time_ms + time_usedict_ms["r1"])
        indices = torch.topk(scores, self.n_candidates_expand)[1][0]
        pruned_leaves = []
        for i in indices:
            if i.item() < len(leaves):
                # res = leaves[i].expand_children(candidate_paths, self.horizon-self.first_stage_horizon, self.speed_limit, self.planner)
                # if not res:
                #     continue
                pruned_leaves.append(leaves[i])
                parent_scores[leaves[i]] = scores[0, i].item()

        # expand leaves with higher scores
        for leaf in pruned_leaves:
            logger.debug(f'leaf.state:{leaf.state[:2]}')
            res = leaf.expand_children(candidate_paths, self.horizon-self.first_stage_horizon, self.speed_limit, self.planner)
            

        # get all leaves
        leaves = TrajTree.get_children(leaves)
        if len(leaves) > self.n_candidates_max:
           leaves = random.sample(leaves, self.n_candidates_max)

        # query the model      
        trajs = [leaf.total_traj[1:] for leaf in leaves]
        time_usedict_ms["r2"] = time.time()*1e3 - (start_time_ms + time_usedict_ms["r1"] + time_usedict_ms["p1"])
        agent_trajectories, scores = self.predict(encoder_outputs, trajs, agent_states, self.horizon*10)
        time_usedict_ms["p2"] = time.time()*1e3 - (start_time_ms + time_usedict_ms["r1"] + time_usedict_ms["p1"]
                                                + time_usedict_ms["r2"])
        # calculate scores
        children_scores = {}
        for i, leaf in enumerate(leaves):
            if leaf.parent in children_scores:
                children_scores[leaf.parent].append(scores[0, i].item())
            else:
                children_scores[leaf.parent] = [scores[0, i].item()]

        # get the best parent
        best_parent = None
        best_child_index = None
        best_score = -np.inf
        for parent in parent_scores.keys():
            score = parent_scores[parent] + np.max(children_scores[parent])
            if score > best_score:
                best_parent = parent
                best_score = score
                best_child_index = np.argmax(children_scores[parent])

        # get the best trajectory
        best_traj = best_parent.children[best_child_index].total_traj[1:, :3]
        time_usedict_ms["cs"] = time.time()*1e3 - (start_time_ms + time_usedict_ms["r1"] + time_usedict_ms["p1"]
                                             + time_usedict_ms["r2"] + time_usedict_ms["p2"])
        # self._world_debuger.record_times(time_usedict_ms)

        
        # plot 
        # if debug:
        # if True:
            # self.dtpp_debuger.plot_bokeh(iteration=iteration, env_inputs=env_inputs,
            #                              ego_futures=trajs, agents_future=agent_trajectories[0, i], 
            #                              vehicle_param=get_vehicle_params_from_actor(vehicle))
            # self.dtpp_debuger.show()
            # for i, traj in enumerate(trajs):
            #     self.plot(iteration, env_inputs, traj, agent_trajectories[0, i])
                
        # logger.debug(f'best_traj: {best_traj}')

        return best_traj
    
    def plot(self, iteration, env_inputs, ego_future, agents_future):
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

        # plot ego
        front_length = get_pacifica_parameters().front_length
        rear_length = get_pacifica_parameters().rear_length
        width = get_pacifica_parameters().width
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
