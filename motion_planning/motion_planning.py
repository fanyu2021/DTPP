from carla_planner import planning_utils, path_planning
from carla2inputs import *
from nuplan_adapter.nuplan_data_process import create_model_input_from_carla

def motion_planning_rule(static_obs_info, dynamic_obs_info,
             ego_loc, pred_loc, global_frenet_path, match_point_list):
    while 1:
        # 接收主进程发送的用于局部路径规划的数据，如果没有收到数据子进程会阻塞
        # possible_static_obs, possible_dynamic_obs, \
        #     ego_loc, pred_loc, local_frenet_path_opt, global_frenet_path, match_point_list, road_ids = conn.recv()
        start_time = time.time()
        # 1.确定预测点在全局路径上的投影点索引
        match_point_list, _ = planning_utils.find_match_points(xy_list=[pred_loc],
                                                                frenet_path_node_list=global_frenet_path,
                                                                is_first_run=False,
                                                                pre_match_index=match_point_list[0])
        # 2.根据匹配点的索引在全局路径上采样一定数量的点
        local_frenet_path_ = planning_utils.sampling(match_point_list[0], global_frenet_path,
                                                     back_length=10, forward_length=50)
        # 由于我们的道路采样精度最少是2（1的情况不考虑，太小的采样精度在实际中不现实），所以确定参考线的时候向后取50个点可以保证最少以百米的未来参考
        # 后面进行动态规划的时候我们搜索的范围就是一百米，所以要保证动态规划的过程中参考线是存在的

        # 3.对采样点进行平滑，作为后续规划的参考线
        local_frenet_path_opt = planning_utils.smooth_reference_line(local_frenet_path_)

        # 计算以车辆当前位置为原点的s_map
        s_map = planning_utils.cal_s_map_fun(local_frenet_path_opt, origin_xy=ego_loc[0:2])
        # path_s, path_l = planning_utils.cal_s_l_fun(local_frenet_path_opt, local_frenet_path_opt, s_map)
        # 提取障碍物的位置信息
        if len(static_obs_info) != 0 and static_obs_info[0][-1] <= 30:
            static_obs_xy = []
            for x, y, dis in static_obs_info:
                static_obs_xy.append((x, y))

            # 计算障碍物的s,l
            obs_s_list, obs_l_list = planning_utils.cal_s_l_fun(static_obs_xy, local_frenet_path_opt, s_map)
        else:
            obs_s_list, obs_l_list = [], []

        dynamic_obs_xy = []
        obs_dis_speed_list = []
        print("--- 42 --- possible_dynamic_obs:", dynamic_obs_info)
        if len(dynamic_obs_info) != 0:
            for x, y, dis_, speed_ in dynamic_obs_info:
                dynamic_obs_xy.append((x, y))
                obs_dis_speed_list.append((dis_, speed_))

        # 计算规划起点的s, l
        begin_s_list, begin_l_list = planning_utils.cal_s_l_fun([pred_loc], local_frenet_path_opt, s_map)

        "自车从规划起点预测后面在不同时刻的位置， 同时预测障碍物在不同时刻的位置，确定二者交汇位置和时间，记录这些信息"
        if len(dynamic_obs_xy) != 0:
            Len_vehicle = 2.910  # 自车长度
            Len_obs = 3  # 障碍物车辆长度
            V_obs = obs_dis_speed_list[0][1]  # 障碍物的速度
            Dis = obs_dis_speed_list[0][0]  # 障碍物距离自车的距离
            V_ego = math.sqrt(ego_loc[3] ** 2 + ego_loc[4] ** 2)
            delta_v = V_ego - V_obs
            # print("V_ego, V_obs", V_ego, V_obs)
            # 相遇开始的时间和相遇结束的时间
            meet_t = (Dis - Len_vehicle / 2 - Len_obs / 2) / delta_v
            delta_t = (Len_vehicle + Len_obs) / delta_v
            leave_t = meet_t + delta_t
            # print("meet_t, leave_t", meet_t, leave_t)
            """
            meet_s 是障碍物在相遇时尾部的s值
            leave_s 是障碍物在与自车分离时头部的s值
            -------------00(meet_s)^^-------------------------
            -------------------------^^(leave_s)00------------
            """
            meet_s = begin_s_list[0] + Dis + V_obs * meet_t - Len_obs / 2
            leave_s = begin_s_list[0] + Dis + V_obs * leave_t + Len_obs / 2
            delta_s = leave_s - meet_s
            obs_pos = meet_s + delta_s / 2
            # print("meet_s and leave_s", begin_s_list[0], Dis, meet_s, leave_s)
            # print("障碍物位置和长度", obs_pos, delta_s)
            # print(obs_s_list)
            if leave_s < 80:
                obs_s_list.append(meet_s - 10)
                obs_s_list.append(obs_pos)
                obs_s_list.append(leave_s)
                obs_l_list.append(0)
                obs_l_list.append(0)
                obs_l_list.append(0)
        """从规划起点进行动态规划"""
        # 计算规划起点的l对s的导数和偏导数
        l_list, _, _, _, l_ds_list, _, l_dds_list = \
            planning_utils.cal_s_l_deri_fun(xy_list=[pred_loc],
                                            V_xy_list=[ego_loc[3:5]],
                                            a_xy_list=[ego_loc[5:]],
                                            local_path_xy_opt=local_frenet_path_opt,
                                            origin_xy=pred_loc)
        # 从起点开始沿着s进行横向和纵向采样，然后动态规划,相邻点之间依据五次多项式进一步采样，间隔一米
        # print("*motion planning time cost:", time.time() - start_time)
        dp_path_s, dp_path_l = path_planning.DP_algorithm(obs_s_list, obs_l_list,
                                                          plan_start_s=begin_s_list[0],
                                                          plan_start_l=l_list[0],
                                                          plan_start_dl=l_ds_list[0],
                                                          plan_start_ddl=l_dds_list[0])
        # print("**dp planning time cost:", time.time() - start_time)
        # 对动态规划得到的路径进行降采样，减少二次规划的计算量，然后二次规划完成后再插值填充恢复
        dp_path_l = dp_path_l[::2]
        dp_path_s = dp_path_s[::2]
        l_min, l_max = \
            path_planning.cal_lmin_lmax(dp_path_s=dp_path_s, dp_path_l=dp_path_l,
                                        obs_s_list=obs_s_list, obs_l_list=obs_l_list,
                                        obs_length=5, obs_width=5)  # 这一步的延迟很低，忽略不计

        # 二次规划变量过多会导致计算延迟比较高，需要平衡二者之间的关系
        # print("l_min_max_length", len(l_min))
        """二次规划"""
        qp_path_l, qp_path_dl, qp_path_ddl = \
            path_planning.Quadratic_planning(l_min, l_max,
                                             plan_start_l=l_list[0],
                                             plan_start_dl=l_ds_list[0],
                                             plan_start_ddl=l_dds_list[0])
        # print(qp_path_l)
        # print("**qp planning time cost:", time.time() - start_time)
        path_s = [dp_path_s[0]]
        path_l = [qp_path_l[0]]
        for i in range(1, len(qp_path_l)):
            path_s.append((dp_path_s[i] + dp_path_s[i - 1]) / 2)
            path_l.append((qp_path_l[i] + qp_path_l[i - 1]) / 2)
        path_s.append(dp_path_s[-1])
        path_l.append(qp_path_l[-1])

        current_local_frenet_path_opt = \
            path_planning.frenet_2_x_y_theta_kappa(plan_start_s=begin_s_list[0],
                                                   plan_start_l=begin_l_list[0],
                                                   enriched_s_list=path_s,
                                                   enriched_l_list=path_l,
                                                   frenet_path_opt=local_frenet_path_opt,
                                                   s_map=s_map)
        # 将重新规划得到的路径信息发送给主进程，让控制器进行轨迹跟踪
        # conn.send((current_local_frenet_path_opt, match_point_list, path_s, path_l))
        # print("***motion planning time cost:", time.time() - start_time)
        return (current_local_frenet_path_opt, match_point_list, path_s, path_l)
        
def motion_planning_e2e(planner, possible_static_obs, possible_dynamic_obs,
            ego_loc, pred_loc, local_frenet_path_opt, global_frenet_path, match_point_list, road_ids):
    """
    端到端规划
    """
    carla_scenario_input : CarlaScenarioInput = CarlaScenarioInput()
    while True:
        # 接收主进程发送的用于局部路径规划的数据，如果没有收到数据子进程会阻塞
        # possible_static_obs, possible_dynamic_obs, \
        #     ego_loc, pred_loc, local_frenet_path_opt, global_frenet_path, match_point_list, road_ids = conn.recv()
        carla_scenario = CarlaScenario(possible_dynamic_obs = possible_static_obs, 
                                       possible_static_obs = possible_dynamic_obs, 
                                       vehicle_loc = ego_loc,
                                       pred_loc = pred_loc, 
                                       local_frenet_path_opt = local_frenet_path_opt,
                                       global_frenet_path = global_frenet_path, 
                                       match_point_list = match_point_list,
                                       road_ids = road_ids)
        
        carla_scenario_input = create_model_input_from_carla(carla_scenario=carla_scenario, carla_scenario_input=carla_scenario_input)

        trajectory = planner.compute_planner_trajectory(carla_scenario_input=carla_scenario_input)
        # conn.send(trajectory)  # Todo(fanyu):发送规划结果给主进程
        return trajectory