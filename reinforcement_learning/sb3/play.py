"""Script to play a checkpoint if an RL agent from Stable-Baselines3."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from Stable-Baselines3.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Name of the task. Optional includes: FAST-Quadcopter-Bodyrate; FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint; FAST-Swarm-Bodyrate; FAST-Swarm-Waypoint.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=300, help="Length of the recorded video (in frames).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--real_time", action="store_true", default=True, help="Run in real-time, if possible.")
parser.add_argument(
    "--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Verbosity level of the custom logger."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in ["FAST-RGB-Waypoint", "FAST-Depth-Waypoint"]:
    args_cli.enable_cameras = True
elif args_cli.task not in ["FAST-Quadcopter-Bodyrate", "FAST-Quadcopter-Waypoint", "FAST-Swarm-Bodyrate", "FAST-Swarm-Waypoint"]:
    raise ValueError(
        "Invalid task name #^# Please select from: FAST-Quadcopter-Bodyrate; FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint; FAST-Swarm-Bodyrate; FAST-Swarm-Waypoint."
    )
if args_cli.video:
    args_cli.enable_cameras = True

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

# TODO: Improve import modality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import gymnasium as gym
from loguru import logger
import rclpy
import time
import torch

from stable_baselines3 import PPO

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg

from envs import camera_waypoint_env, quadcopter_bodyrate_env, quadcopter_waypoint_env, swarm_bodyrate_env, swarm_waypoint_env


def main():
    """Play with stable-baselines agent."""

    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    log_root_path = os.path.join("outputs", "sb3", args_cli.task, "flowline")
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint is None:
        checkpoint_path = get_checkpoint_path(log_root_path, other_dirs=["models"])
        logger.info(f"No checkpoint specified, using auto-detected checkpoint from {checkpoint_path} ~(^v^)~")
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(checkpoint_path)

    # Post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "play",
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Sb3VecEnvWrapper(env)

    agent = PPO.load(checkpoint_path, env, device=agent_cfg["device"])

    dt = env.unwrapped.step_dt

    # Reset environment
    obs = env.reset()
    # Simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # Run everything in inference mode
        with torch.inference_mode():
            # Agent stepping
            actions, _ = agent.predict(obs, deterministic=True)
            # Env stepping
            obs, _, _, _ = env.step(actions)

        # Time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level=args_cli.verbosity)

    rclpy.init()
    main()
    rclpy.shutdown()

    simulation_app.close()
