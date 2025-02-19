"""Script to run a keyboard teleoperation with quacopter environments"""

import argparse

from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for quadcopter environments.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--sensitivity", type=float, default=0.5, help="Sensitivity factor.")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in ["FAST-Quadcopter-RGB-Camera-Direct-v0", "FAST-Quadcopter-Depth-Camera-Direct-v0"]:
    args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import gymnasium as gym
from loguru import logger
import math
import matplotlib.pyplot as plt
import numpy as np
import rclpy
import sys
import torch

import quadcopter_env, camera_env, swarm_env
from isaaclab.devices import Se3Keyboard
from isaaclab_tasks.utils import parse_env_cfg

from utils import quat_to_yaw


def delta_pose_to_action(delta_pose: np.ndarray) -> np.ndarray:
    action = np.zeros(14)

    if delta_pose[4] > 0:
        action[5] = args_cli.sensitivity
    elif delta_pose[4] < 0:
        action[5] = -args_cli.sensitivity

    if delta_pose[0] > 0:
        action[3] = args_cli.sensitivity
    elif delta_pose[0] < 0:
        action[3] = -args_cli.sensitivity

    if delta_pose[1] > 0:
        action[4] = args_cli.sensitivity
    elif delta_pose[1] < 0:
        action[4] = -args_cli.sensitivity

    if delta_pose[2] > 0:
        action[13] = args_cli.sensitivity
    elif delta_pose[2] < 0:
        action[13] = -args_cli.sensitivity

    return action


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
    # Create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # Create controller
    teleop_interface = Se3Keyboard()
    # Add teleoperation key for env reset
    teleop_interface.add_callback("R", env.reset)

    # Reset environment
    env.reset()
    teleop_interface.reset()

    p_desired, yaw_desired = None, None
    v_max, yaw_dot_max, dt = env.unwrapped.cfg.v_max, env.unwrapped.cfg.yaw_dot_max, env.unwrapped.step_dt
    # Simulate environment
    while simulation_app.is_running():
        # Run everything in inference mode
        with torch.inference_mode():
            # Get keyboard command
            delta_pose, _ = teleop_interface.advance()
            action = delta_pose_to_action(delta_pose)
            action = action.astype("float32")

            # Convert to torch
            if args_cli.task == "FAST-Quadcopter-Swarm-Direct-v0":
                actions = {drone: torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1) for drone in env_cfg.possible_agents}
                if p_desired is not None and yaw_desired is not None:
                    for drone in env_cfg.possible_agents:
                        p_desired[drone] += actions[drone][:, 3:6] * v_max[drone] * dt
                        yaw_desired[drone] += actions[drone][:, 13] * yaw_dot_max[drone] * dt

                        actions[drone][:, :3] = p_desired[drone] / env_cfg.p_max[drone]
                        actions[drone][:, 12] = yaw_desired[drone] / math.pi
            else:
                actions = torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)
                if p_desired is not None and yaw_desired is not None:
                    p_desired += actions[:, 3:6] * v_max * dt
                    yaw_desired += actions[:, 13] * yaw_dot_max * dt

                    actions[:, :3] = p_desired / env_cfg.p_max
                    actions[:, 12] = yaw_desired / math.pi

            # Apply actions
            obs, _, reset_terminated, reset_time_outs, _ = env.step(actions)

            if args_cli.task == "FAST-Quadcopter-Swarm-Direct-v0":
                if p_desired is None and yaw_desired is None:
                    p_desired = {drone: obs[drone][:, :3] for drone in env_cfg.possible_agents}
                    yaw_desired = {drone: quat_to_yaw(obs[drone][:, 3:7]) for drone in env_cfg.possible_agents}

                reset_env_ids = (math.prod(reset_terminated.values()) | math.prod(reset_time_outs.values())).nonzero(as_tuple=False).squeeze(-1)
                for drone in env_cfg.possible_agents:
                    p_desired[drone][reset_env_ids] = obs[drone][reset_env_ids, :3]
                    yaw_desired[drone][reset_env_ids] = quat_to_yaw(obs[drone][reset_env_ids, 3:7])
            else:
                if p_desired is None and yaw_desired is None:
                    p_desired = obs["policy"][:, :3]
                    yaw_desired = quat_to_yaw(obs["policy"][:, 3:7])

                reset_env_ids = (reset_terminated | reset_time_outs).nonzero(as_tuple=False).squeeze(-1)
                p_desired[reset_env_ids] = obs["policy"][reset_env_ids, :3]
                yaw_desired[reset_env_ids] = quat_to_yaw(obs["policy"][reset_env_ids, 3:7])

            if args_cli.task in ["FAST-Quadcopter-RGB-Camera-Direct-v0", "FAST-Quadcopter-Depth-Camera-Direct-v0"]:
                visualize_images_live(obs["image"].cpu().numpy())

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
