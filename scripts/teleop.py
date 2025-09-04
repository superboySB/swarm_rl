"""Script to run a keyboard teleoperation with quadcopter environments"""

import argparse
import os
import sys
from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for quadcopter environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Name of the task. Optional includes: FAST-Quadcopter-Waypoint; FAST-Quadcopter-Vel; FAST-Quadcopter-Acc; FAST-RGB-Waypoint; FAST-Depth-Waypoint.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--acceleration", type=float, default=4.0, help="Acceleration of teleoperation.")
parser.add_argument("--velocity", type=float, default=4.0, help="Velocity of teleoperation.")
parser.add_argument(
    "--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Verbosity level of the custom logger."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in ["FAST-Swarm-Bodyrate", "FAST-Swarm-Vel", "FAST-Swarm-Waypoint"]:
    raise ValueError("Swarm envs are not supported for keyboard teleoperation due to the observation space limitation of Isaaclab 'multi_agent_to_single_agent' API #^#")
elif args_cli.task in ["FAST-RGB-Waypoint", "FAST-Depth-Waypoint"]:
    args_cli.enable_cameras = True
elif args_cli.task not in ["FAST-Quadcopter-Waypoint", "FAST-Quadcopter-Vel", "FAST-Quadcopter-Acc"]:
    raise ValueError(
        "Invalid task name #^# Please select from: FAST-Quadcopter-Waypoint; FAST-Quadcopter-Vel; FAST-Quadcopter-Acc; FAST-RGB-Waypoint; FAST-Depth-Waypoint."
    )

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
import torch

from envs import camera_waypoint_env, quadcopter_waypoint_env, quadcopter_vel_env, quadcopter_acc_env
from isaaclab.devices import Se3Keyboard
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab.utils.math import quat_inv, quat_apply


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

    # Create controller
    teleop_interface = Se3Keyboard()

    # Reset environment
    env.reset()
    teleop_interface.reset()

    p_desired, p_odom, q_odom = None, None, None
    dt = env.unwrapped.step_dt

    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Get keyboard command
            delta_pose, _ = teleop_interface.advance()

            actions = None
            if args_cli.task.endswith("Waypoint"):
                actions = torch.zeros(env_cfg.action_space, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                if p_desired is not None:
                    if delta_pose[0] > 0:
                        p_desired[:, 0] += args_cli.velocity * dt
                    elif delta_pose[0] < 0:
                        p_desired[:, 0] -= args_cli.velocity * dt
                    if delta_pose[1] > 0:
                        p_desired[:, 1] += args_cli.velocity * dt
                    elif delta_pose[1] < 0:
                        p_desired[:, 1] -= args_cli.velocity * dt
                    if delta_pose[4] > 0:
                        p_desired[:, 2] += args_cli.velocity * dt
                    elif delta_pose[4] < 0:
                        p_desired[:, 2] -= args_cli.velocity * dt

                    goal_in_body_frame = quat_apply(quat_inv(q_odom), p_desired - p_odom)
                    norm = goal_in_body_frame.norm(p=2, dim=1, keepdim=True)
                    clip_scale = torch.where(norm > env_cfg.num_pieces * env_cfg.p_max, env_cfg.num_pieces * env_cfg.p_max / (norm + 1e-6), torch.ones_like(norm))
                    goal_in_body_frame *= clip_scale
                    for i in range(env_cfg.num_pieces):
                        actions[:, 3 * i : 3 * (i + 1)] = goal_in_body_frame / env_cfg.num_pieces / env_cfg.p_max * env_cfg.clip_action

            elif args_cli.task.endswith("Vel"):
                actions = torch.zeros(env_cfg.action_space, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                if delta_pose[0] > 0:
                    actions[:, 0] = args_cli.velocity
                elif delta_pose[0] < 0:
                    actions[:, 0] = -args_cli.velocity
                if delta_pose[1] > 0:
                    actions[:, 1] = args_cli.velocity
                elif delta_pose[1] < 0:
                    actions[:, 1] = -args_cli.velocity

                speed = torch.norm(actions, dim=1, keepdim=True)
                clip_scale = torch.where(speed > env_cfg.v_max, env_cfg.v_max / (speed + 1e-6), torch.ones_like(speed))
                actions *= clip_scale

            elif args_cli.task.endswith("Acc"):
                actions = torch.zeros(env_cfg.action_space, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                if delta_pose[0] > 0:
                    actions[:, 0] = args_cli.acceleration
                elif delta_pose[0] < 0:
                    actions[:, 0] = -args_cli.acceleration
                if delta_pose[1] > 0:
                    actions[:, 1] = args_cli.acceleration
                elif delta_pose[1] < 0:
                    actions[:, 1] = -args_cli.acceleration

                norm = torch.norm(actions, dim=1, keepdim=True)
                clip_scale = torch.where(norm > env_cfg.v_max, env_cfg.v_max / (norm + 1e-6), torch.ones_like(norm))
                actions *= clip_scale

            else:
                actions = {
                    drone: torch.zeros(env_cfg.action_spaces[drone], device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1) for drone in env_cfg.possible_agents
                }
                if p_desired is not None:
                    for drone in env_cfg.possible_agents:
                        if delta_pose[0] > 0:
                            p_desired[drone][:, 0] += args_cli.velocity * dt
                        elif delta_pose[0] < 0:
                            p_desired[drone][:, 0] -= args_cli.velocity * dt
                        if delta_pose[1] > 0:
                            p_desired[drone][:, 1] += args_cli.velocity * dt
                        elif delta_pose[1] < 0:
                            p_desired[drone][:, 1] -= args_cli.velocity * dt
                        if delta_pose[4] > 0:
                            p_desired[drone][:, 2] += args_cli.velocity * dt
                        elif delta_pose[4] < 0:
                            p_desired[drone][:, 2] -= args_cli.velocity * dt

                        goal_in_body_frame = quat_apply(quat_inv(q_odom[drone]), p_desired[drone] - p_odom[drone])
                        norm = goal_in_body_frame.norm(p=2, dim=1, keepdim=True)
                        clip_scale = torch.where(
                            norm > env_cfg.num_pieces * env_cfg.p_max[drone], env_cfg.num_pieces * env_cfg.p_max[drone] / (norm + 1e-6), torch.ones_like(norm)
                        )
                        goal_in_body_frame *= clip_scale
                        for i in range(env_cfg.num_pieces):
                            actions[drone][:, 3 * i : 3 * (i + 1)] = goal_in_body_frame / env_cfg.num_pieces / env_cfg.p_max[drone] * env_cfg.clip_action

            # Apply actions
            obs, _, reset_terminated, reset_time_outs, _ = env.step(actions)

            if args_cli.task.endswith("Waypoint"):
                p_odom = obs["odom"][:, :3]
                q_odom = obs["odom"][:, 3:7]
                if p_desired is None:
                    p_desired = obs["odom"][:, :3].clone()
                reset_env_ids = (reset_terminated | reset_time_outs).nonzero(as_tuple=False).squeeze(-1)
                p_desired[reset_env_ids] = p_odom[reset_env_ids].clone()
            elif args_cli.task.endswith("Vel") or args_cli.task.endswith("Acc"):
                pass
            else:
                p_odom = {drone: obs[drone][:, :3] for drone in env_cfg.possible_agents}
                q_odom = {drone: obs[drone][:, 3:7] for drone in env_cfg.possible_agents}
                if p_desired is None:
                    p_desired = {drone: obs[drone][:, :3].clone() for drone in env_cfg.possible_agents}
                reset_env_ids = (math.prod(reset_terminated.values()) | math.prod(reset_time_outs.values())).nonzero(as_tuple=False).squeeze(-1)
                for drone in env_cfg.possible_agents:
                    p_desired[drone][reset_env_ids] = p_odom[drone][reset_env_ids].clone()

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
