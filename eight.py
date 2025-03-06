"""Script to run a eight trajectory generation and tracking with single quacopter environment"""

import argparse

from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Eight trajectory generation and tracking for single quadcopter environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task != "FAST-Quadcopter-Direct-v0":
    raise ValueError("Only the 'FAST-Quadcopter-Direct-v0' environment is supported for eight trajectory generation and tracking.")

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
from loguru import logger
import rclpy
import sys
import time
import torch

import quadcopter_env
from isaaclab_tasks.utils import parse_env_cfg
from minco import MinJerkOpt


def generate_eight_trajectory(p_odom, v_odom, a_odom, p_init):
    num_pieces = 6

    head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)
    tail_pva = torch.stack([p_init, torch.zeros_like(p_init), torch.zeros_like(p_init)], dim=2)

    inner_pts = torch.zeros((p_odom.shape[0], 3, num_pieces - 1), device=p_odom.device)
    inner_pts[:, :, 0] = p_init + torch.tensor([2.0, -2.0, 0.0], device=p_odom.device)
    inner_pts[:, :, 1] = p_init + torch.tensor([2.0, 2.0, 0.0], device=p_odom.device)
    inner_pts[:, :, 2] = p_init + torch.tensor([0.0, 0.0, 0.0], device=p_odom.device)
    inner_pts[:, :, 3] = p_init + torch.tensor([-2.0, -2.0, 0.0], device=p_odom.device)
    inner_pts[:, :, 4] = p_init + torch.tensor([-2.0, 2.0, 0.0], device=p_odom.device)

    durations = torch.full((p_odom.shape[0], num_pieces), 2.0, device=p_odom.device)

    MJO = MinJerkOpt(head_pva, tail_pva, num_pieces)
    start = time.perf_counter()
    MJO.generate(inner_pts, durations)
    end = time.perf_counter()
    logger.info(f"Trajectory generation takes {end - start:.6f}s")

    return MJO.get_traj()


def main():
    # Create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Reset environment
    env.reset()

    traj, traj_dur, env_reset, replan_required, traj_update_required, execution_time = None, None, None, None, None, None
    p_init, p_odom, v_odom = None, None, None

    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            if traj is None:
                obs, _, _, _, _ = env.step(torch.zeros((env.unwrapped.num_envs, 14), device=env.unwrapped.device))
                p_init = obs["policy"][:, :3].clone()
                p_odom = obs["policy"][:, :3]
                v_odom = obs["policy"][:, 7:10]
                a_odom = torch.zeros_like(v_odom)

                traj = generate_eight_trajectory(p_odom, v_odom, a_odom, p_init)
                traj_dur = traj.get_total_duration()
                execution_time = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
                traj_update_required = torch.tensor([False] * env.unwrapped.num_envs, device=env.unwrapped.device)

            if traj_update_required.any():
                a_odom = torch.where(env_reset, torch.zeros_like(v_odom), traj.get_acc(execution_time))
                update_traj = generate_eight_trajectory(
                    p_odom[traj_update_required], v_odom[traj_update_required], a_odom[traj_update_required], p_init[traj_update_required]
                )
                traj[traj_update_required] = update_traj
                traj_dur[traj_update_required] = update_traj.get_total_duration()
                execution_time[traj_update_required] = 0.0
                replan_required.fill_(False)

            actions = torch.cat(
                (
                    traj.get_pos(execution_time) / env_cfg.p_max,
                    traj.get_vel(execution_time) / env_cfg.v_max,
                    traj.get_acc(execution_time) / env_cfg.a_max,
                    traj.get_jer(execution_time) / env_cfg.j_max,
                    torch.zeros_like(execution_time).unsqueeze(1),
                    torch.zeros_like(execution_time).unsqueeze(1),
                ),
                dim=1,
            )

            # Apply actions
            obs, _, reset_terminated, reset_time_outs, _ = env.step(actions)
            execution_time += env.unwrapped.step_dt

            env_reset = reset_terminated | reset_time_outs
            replan_required = execution_time > 0.9 * traj_dur
            traj_update_required = env_reset | replan_required
            p_odom[traj_update_required] = obs["policy"][traj_update_required, :3]
            v_odom[traj_update_required] = obs["policy"][traj_update_required, 7:10]

    # Close the simulator
    env.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="INFO")

    rclpy.init()
    # Run the main function
    main()
    rclpy.shutdown()

    # Close sim app
    simulation_app.close()
