"""Script to run a eight trajectory generation and tracking with single quacopter environment"""

import argparse
import os
import sys
from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Eight trajectory generation and tracking for single quadcopter environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Name of the task. Optional includes: FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Verbosity level of the custom logger."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in ["FAST-Swarm-Bodyrate", "FAST-Swarm-Waypoint"]:
    raise ValueError("Swarm envs are not supported for eight trajectory generation and tracking #^#")
elif args_cli.task in ["FAST-RGB-Waypoint", "FAST-Depth-Waypoint"]:
    args_cli.enable_cameras = True
elif args_cli.task != "FAST-Quadcopter-Waypoint":
    raise ValueError("Invalid task name #^# Please select from: FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint.")

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# TODO: Improve import modality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import gymnasium as gym
from loguru import logger
import math
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import time
import torch

from envs import camera_waypoint_env, quadcopter_bodyrate_env, quadcopter_waypoint_env, swarm_bodyrate_env, swarm_waypoint_env
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_inv, quat_rotate
from utils.minco import MinJerkOpt


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
    logger.trace(f"Eight trajectory generation takes {end - start:.5f}s")

    return MJO.get_traj()


def visualize_images_live(images):
    # Images shape can be (N, height, width) or (N, height, width, channels)
    N = images.shape[0]

    channels = images.shape[-1]
    if channels == 1:
        # Convert grayscale images to 3 channels by repeating the single channel
        images = np.repeat(images, 3, axis=-1)
        images = np.where(np.isinf(images), np.nan, images)
    elif channels == 4:
        # Use only the first 3 channels as RGB, ignore the 4th channel (perhaps alpha)
        images = images[..., :3]

    # Get the height and width from the first image (all images have the same size)
    height, width = images.shape[1], images.shape[2]

    # Calculate the grid size
    cols = int(math.ceil(math.sqrt(N)))
    rows = int(math.ceil(N / cols))

    # Create an empty canvas to hold the images
    canvas = np.zeros((rows * height, cols * width, 3))

    for idx in range(N):
        row = idx // cols
        col = idx % cols
        # Place the image in the grid cell
        canvas[row * height : (row * height) + height, col * width : (col * width) + width, :] = images[idx]

    # Display the canvas
    if not hasattr(visualize_images_live, "img_plot"):
        # Create the plot for the first time
        visualize_images_live.fig = plt.figure()
        visualize_images_live.fig.canvas.manager.set_window_title("Images")
        visualize_images_live.img_plot = plt.imshow(canvas)
        plt.axis("off")
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    else:
        # Update the existing plot
        visualize_images_live.img_plot.set_data(canvas)

    plt.draw()
    plt.pause(0.001)  # Pause to allow the figure to update


def main():
    # Create environment configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.debug_vis_goal = False
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    # Reset environment
    env.reset()

    traj, traj_dur, execution_time, env_reset, replan_required, traj_update_required = None, None, None, None, None, None
    p_init, p_odom, q_odom, v_odom = None, None, None, None

    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            if traj is None:
                obs, _, _, _, _ = env.step(torch.zeros((env.unwrapped.num_envs, env_cfg.action_space), device=env.unwrapped.device))
                p_init = obs["odom"][:, :3].clone()
                p_odom = obs["odom"][:, :3]
                q_odom = obs["odom"][:, 3:7]
                v_odom = obs["odom"][:, 7:10]
                a_odom = torch.zeros_like(v_odom)

                traj = generate_eight_trajectory(p_odom, v_odom, a_odom, p_init)
                traj_dur = traj.get_total_duration()
                execution_time = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device)
                traj_update_required = torch.tensor([False] * env.unwrapped.num_envs, device=env.unwrapped.device)

            if traj_update_required.any():
                a_odom = torch.where(env_reset.unsqueeze(1), torch.zeros_like(v_odom), traj.get_acc(execution_time))
                update_traj = generate_eight_trajectory(
                    p_odom[traj_update_required], v_odom[traj_update_required], a_odom[traj_update_required], p_init[traj_update_required]
                )
                traj[traj_update_required] = update_traj
                traj_dur[traj_update_required] = update_traj.get_total_duration()
                execution_time[traj_update_required] = 0.0

            waypoints = [
                traj.get_pos(execution_time + i * env_cfg.duration) - traj.get_pos(execution_time + (i - 1) * env_cfg.duration)
                / env_cfg.p_max
                * env_cfg.clip_action
                for i in range(1, env_cfg.num_pieces + 1)
            ]
            actions = torch.cat(waypoints, dim=1)
            actions = torch.cat(
                (
                    actions,
                    traj.get_vel(execution_time + env_cfg.num_pieces * env_cfg.duration) / env_cfg.v_max * env_cfg.clip_action,
                    traj.get_acc(execution_time + env_cfg.num_pieces * env_cfg.duration) / env_cfg.a_max * env_cfg.clip_action,
                ),
                dim=1,
            )

            # Apply actions
            obs, _, reset_terminated, reset_time_outs, _ = env.step(actions)
            execution_time += env.unwrapped.step_dt
            p_odom = obs["odom"][:, :3]
            q_odom = obs["odom"][:, 3:7]
            v_odom = obs["odom"][:, 7:10]

            env_reset = reset_terminated | reset_time_outs
            replan_required = execution_time > 0.77 * traj_dur  # Magical Doncic
            traj_update_required = env_reset | replan_required

            if args_cli.task in ["FAST-RGB-Waypoint", "FAST-Depth-Waypoint"]:
                visualize_images_live(obs["image"].cpu().numpy())

    # Close the simulator
    env.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level=args_cli.verbosity)

    rclpy.init()
    main()
    rclpy.shutdown()

    simulation_app.close()
