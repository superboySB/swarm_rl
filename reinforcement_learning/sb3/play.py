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
    help="Name of the task. Optional Includes: FAST-Quadcopter-Direct-v0; FAST-Quadcopter-RGB-Camera-Direct-v0; FAST-Quadcopter-Depth-Camera-Direct-v0; FAST-Quadcopter-Swarm-Direct-v0.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playing.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in frames).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--real_time", action="store_true", default=False, help="Run in real-time, if possible.")
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in ["FAST-Quadcopter-RGB-Camera-Direct-v0", "FAST-Quadcopter-Depth-Camera-Direct-v0"]:
    args_cli.enable_cameras = True
elif args_cli.task not in ["FAST-Quadcopter-Direct-v0", "FAST-Quadcopter-Swarm-Direct-v0"]:
    raise ValueError(
        "Invalid task name #^# Please select from: FAST-Quadcopter-Direct-v0; FAST-Quadcopter-RGB-Camera-Direct-v0; FAST-Quadcopter-Depth-Camera-Direct-v0; FAST-Quadcopter-Swarm-Direct-v0."
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
import numpy as np
import rclpy
import time
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
import isaaclab_tasks  # noqa: F401
from envs import quadcopter_env, camera_env, swarm_env
from isaaclab_tasks.utils.parse_cfg import get_checkpoint_path, load_cfg_from_registry, parse_env_cfg


def main():
    """Play with stable-baselines agent."""

    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg = load_cfg_from_registry(args_cli.task, "sb3_cfg_entry_point")

    log_root_path = os.path.join("outputs", "sb3", args_cli.task, "flowline")
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint is None:
        checkpoint_path = get_checkpoint_path(log_root_path, ".*", "model_final.zip", ["models"])
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

    if "normalize_input" in agent_cfg:
        env = VecNormalize(
            env,
            training=True,
            norm_obs="normalize_input" in agent_cfg and agent_cfg.pop("normalize_input"),
            norm_reward="normalize_value" in agent_cfg and agent_cfg.pop("normalize_value"),
            clip_obs="clip_obs" in agent_cfg and agent_cfg.pop("clip_obs"),
            gamma=agent_cfg["gamma"],
            clip_reward=np.inf,
        )

    agent = PPO.load(checkpoint_path, env, device=agent_cfg["device"])

    dt = env.unwrapped.physics_dt

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
    logger.add(sys.stdout, level="INFO")

    rclpy.init()
    main()
    rclpy.shutdown()

    simulation_app.close()
