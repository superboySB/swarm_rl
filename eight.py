"""Script to run a eight trajectory generation and tracking with single quacopter environment"""

import argparse

from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Eight trajectory generation and tracking for single quadcopter environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1024, help="Number of environments to simulate.")
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
import math
import rclpy
import sys
import torch

import quadcopter_env
from isaaclab_tasks.utils import parse_env_cfg

from minco import MinJerkOpt, Trajectory


def main():
    # Create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Reset environment
    env.reset()

    traj, traj_dur, traj_update, p_odom, v_odom, a_odom = None, None, None, None, None, None
    t = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            if traj is None:
                obs, _, _, _, _ = env.step(torch.zeros((env.unwrapped.num_envs, 14), device=env.unwrapped.device))
                p_odom = obs["policy"][:, :3]
                v_odom = obs["policy"][:, 7:10]
                a_odom = torch.zeros_like(v_odom)

                num_pieces = 6
                head_state = torch.stack([p_odom, v_odom, a_odom], dim=2)
                tail_state = torch.stack([p_odom, torch.zeros_like(p_odom), torch.zeros_like(p_odom)], dim=2)
                inner_pts = torch.zeros((env.unwrapped.num_envs, 3, num_pieces - 1), device=env.unwrapped.device)
                inner_pts[:, :, 0] = p_odom + torch.tensor([5.0, -5.0, 0.0], device=env.unwrapped.device)
                inner_pts[:, :, 1] = p_odom + torch.tensor([5.0, 5.0, 0.0], device=env.unwrapped.device)
                inner_pts[:, :, 2] = p_odom + torch.tensor([0.0, 0.0, 0.0], device=env.unwrapped.device)
                inner_pts[:, :, 3] = p_odom + torch.tensor([-5.0, -5.0, 0.0], device=env.unwrapped.device)
                inner_pts[:, :, 4] = p_odom + torch.tensor([-5.0, 5.0, 0.0], device=env.unwrapped.device)
                durations = torch.full((env.unwrapped.num_envs, num_pieces), 5.0, device=env.unwrapped.device)

                MJO = MinJerkOpt(head_state, tail_state, num_pieces)
                MJO.generate(inner_pts, durations)
                traj = MJO.getTraj()
                traj_dur = traj.getTotalDuration()
                traj_update = torch.tensor([False] * env.unwrapped.num_envs, device=env.unwrapped.device)

            actions = torch.cat(
                (
                    traj.getPos(t) / env_cfg.p_max,
                    traj.getVel(t) / env_cfg.v_max,
                    traj.getAcc(t) / env_cfg.a_max,
                    traj.getJer(t) / env_cfg.j_max,
                    torch.zeros_like(t).unsqueeze(1),
                    torch.zeros_like(t).unsqueeze(1),
                ),
                dim=1,
            )

            # Apply actions
            obs, _, reset_terminated, reset_time_outs, _ = env.step(actions)
            t += env.unwrapped.step_dt

            reset_env_ids = (reset_terminated | reset_time_outs).nonzero(as_tuple=False).squeeze(-1)
            p_odom[reset_env_ids] = obs["policy"][reset_env_ids, :3]
            v_odom[reset_env_ids] = obs["policy"][reset_env_ids, 7:10]
            a_odom[reset_env_ids] = torch.zeros_like(v_odom[reset_env_ids])

    # Close the simulator
    env.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level="DEBUG")

    rclpy.init()
    # Run the main function
    main()
    rclpy.shutdown()

    # Close sim app
    simulation_app.close()
