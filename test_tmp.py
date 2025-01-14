import yaml
import datetime
import torch
import argparse
import warnings
from tqdm import tqdm
from planner import Planner
from common_utils import *
warnings.filterwarnings("ignore") 

# from nuplan.planning.simulation.planner.idm_planner import IDMPlanner
# from nuplan.planning.simulation.planner.simple_planner import SimplePlanner
from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_utils import ScenarioMapping
# from nuplan.planning.simulation.callback.simulation_log_callback import SimulationLogCallback
# from nuplan.planning.simulation.callback.metric_callback import MetricCallback
# from nuplan.planning.simulation.callback.multi_callback import MultiCallback
# from nuplan.planning.simulation.main_callback.metric_aggregator_callback import MetricAggregatorCallback
# from nuplan.planning.simulation.main_callback.metric_file_callback import MetricFileCallback
# from nuplan.planning.simulation.main_callback.multi_main_callback import MultiMainCallback
# from nuplan.planning.simulation.main_callback.metric_summary_callback import MetricSummaryCallback
# from nuplan.planning.simulation.observation.tracks_observation import TracksObservation
from nuplan.planning.simulation.observation.idm_agents import IDMAgents
# from nuplan.planning.simulation.controller.perfect_tracking import PerfectTrackingController
# from nuplan.planning.simulation.controller.log_playback import LogPlaybackController
from nuplan.planning.simulation.controller.two_stage_controller import TwoStageController
from nuplan.planning.simulation.controller.tracker.lqr import LQRTracker
from nuplan.planning.simulation.controller.motion_model.kinematic_bicycle import KinematicBicycleModel
from nuplan.planning.simulation.simulation_time_controller.step_simulation_time_controller import StepSimulationTimeController
from nuplan.planning.simulation.runner.simulations_runner import SimulationRunner
from nuplan.planning.simulation.simulation import Simulation
from nuplan.planning.simulation.simulation_setup import SimulationSetup
# from nuplan.planning.nuboard.nuboard import NuBoard
# from nuplan.planning.nuboard.base.data_class import NuBoardFile

from carla2inputs import *

# import carla
# from agents.navigation.behavior_agent import BehaviorAgent

# import logging
# # 配置日志记录
# logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


def main_simulation(planner, scenario):
    print('Building metric engines...DONE\n')
    # Ego Controller and Perception
    
    try:
        tracker = LQRTracker(q_longitudinal=[10.0], r_longitudinal=[1.0], q_lateral=[1.0, 10.0, 0.0], 
                             r_lateral=[1.0], discretization_time=0.1, tracking_horizon=10, 
                             jerk_penalty=1e-4, curvature_rate_penalty=1e-2, 
                             stopping_proportional_gain=0.5, stopping_velocity=0.2)
        motion_model = KinematicBicycleModel(get_pacifica_parameters())
        ego_controller = TwoStageController(scenario, tracker, motion_model) 
        observations = IDMAgents(target_velocity=10, min_gap_to_lead_agent=1.0, headway_time=1.5,
                                 accel_max=1.0, decel_max=2.0, scenario=scenario,
                                 open_loop_detections_types=["PEDESTRIAN", "BARRIER", "CZONE_SIGN", "TRAFFIC_CONE", "GENERIC_OBJECT"])
    except Exception as err:
        raise Exception(f"Failed to initialize controller and perception: {err}")
    
    # Simulation Manager
    simulation_time_controller = StepSimulationTimeController(scenario)
    simulation_setup = SimulationSetup(
        time_controller=simulation_time_controller,
        observations=observations,
        ego_controller=ego_controller,
        scenario=scenario,
    )
    simulation = Simulation(
        simulation_setup=simulation_setup,
        # callback=MultiCallback([metric_callback, sim_log_callback])
    )
    # Begin simulation
    simulation_runner = SimulationRunner(simulation, planner)
    report, trajectory = simulation_runner.fy_run()

    if report.succeeded:
        print("--- Successfully ran simulation.")
        simulation_runner.print_trajectory(trajectory)
    else:
        print("--- Failed Simulation.\n '%s'", report.error_message)

    print('Finished running simulations!')




def main(args):

    # initialize planner
    torch.set_grad_enabled(False)
    planner = Planner(model_path=args.model_path, device=args.device)

    print('Extracting scenarios...')
    use_nuplan_scenario = True
    scenario = None
    if use_nuplan_scenario:
        scenario = creat_nuplan_scenario(args)
    else:
        scenario = create_scenario_from_carla()

    # begin testing
    print('Running simulations...')
    main_simulation(planner, scenario)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--data_path', type=str)
    parser.add_argument('--data_path', type=str, default='/media/xph123/DATA/f_tmp/DTPP_datasets/nuplan-v1.1_test/data/cache/test')
    parser.add_argument('--map_path', type=str, default='/media/xph123/DATA/f_tmp/DTPP_datasets/nuplan-maps-v1.0/maps')
    parser.add_argument('--model_path', type=str, default='/home/xph123/fanyu/E2E/DTPP/base_model/base_model.pth')
    parser.add_argument('--test_type', type=str, default='closed_loop_reactive_agents')
    parser.add_argument('--load_test_set', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--scenarios_per_type', type=int, default=1)
    args = parser.parse_args()

    main(args)
