# Copyright (c) 2022-2025, 99.9% The Isaac Lab Project Developers, 0.1% LAJi.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from isaaclab.app import AppLauncher

# Local imports
import cli_args  # isort: skip

# Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Name of the task. Optional includes: FAST-Quadcopter-Bodyrate; FAST-Quadcopter-Vel; FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint; FAST-Swarm-Bodyrate; FAST-Swarm-Acc; FAST-Swarm-Vel; FAST-Swarm-Waypoint.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Verbosity level of the custom logger."
)

# Append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in ["FAST-RGB-Waypoint", "FAST-Depth-Waypoint"]:
    args_cli.enable_cameras = True
elif args_cli.task not in ["FAST-Quadcopter-Bodyrate", "FAST-Quadcopter-Vel", "FAST-Quadcopter-Waypoint", "FAST-Swarm-Bodyrate", "FAST-Swarm-Acc", "FAST-Swarm-Vel", "FAST-Swarm-Waypoint"]:
    raise ValueError(
        "Invalid task name #^# Please select from: FAST-Quadcopter-Bodyrate; FAST-Quadcopter-Vel; FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint; FAST-Swarm-Bodyrate; FAST-Swarm-Acc; FAST-Swarm-Vel; FAST-Swarm-Waypoint."
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

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path, parse_env_cfg

from envs import camera_waypoint_env, quadcopter_bodyrate_env, quadcopter_waypoint_env, swarm_bodyrate_env, swarm_acc_env, swarm_vel_env, swarm_waypoint_env

# PLACEHOLDER: Extension template (do not remove this comment)


def main():
    """Play with RSL-RL agent."""
    
    # Parse configuration
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env_cfg.fix_range = True

    # Specify directory for logging experiments
    log_root_path = os.path.join("outputs", "rsl_rl", args_cli.task, "flowline")
    log_root_path = os.path.abspath(log_root_path)
    if args_cli.checkpoint is None:
        checkpoint_path = get_checkpoint_path(log_root_path, other_dirs=["checkpoints"])
        logger.info(f"No checkpoint specified, using auto-detected checkpoint from {checkpoint_path} ~(^v^)~")
    else:
        checkpoint_path = args_cli.checkpoint
    log_dir = os.path.dirname(os.path.dirname(checkpoint_path))

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

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # Load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(checkpoint_path)

    # Obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # Extract the neural network module
    # We do this in a try-except to maintain backwards compatibility.
    try:
        # Version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # Version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # Export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(checkpoint_path), "exported")
    export_policy_as_jit(policy_nn, ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=ppo_runner.obs_normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # Reset environment
    # obs, _ = env.get_observations()
    obs, _ = env.reset()
    # Simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # Run everything in inference mode
        with torch.inference_mode():
            # Agent stepping
            actions = policy(obs)
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
