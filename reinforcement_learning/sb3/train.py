# Copyright (c) 2022-2025, 99.9% The Isaac Lab Project Developers, 0.1% LAJi.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with Stable Baselines3.

Since Stable-Baselines3 does not support buffers living on GPU directly,
we recommend using smaller number of environments. Otherwise,
there will be significant overhead in GPU->CPU transfer.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Name of the task. Optional includes: FAST-Quadcopter-Bodyrate; FAST-Quadcopter-Vel; FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint; FAST-Swarm-Bodyrate; FAST-Swarm-Waypoint.",
)
parser.add_argument("--num_envs", type=int, default=1000, help="Number of environments to simulate.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--save_interval", type=int, default=1e7, help="Interval between checkpoints (in steps).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in frames).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument(
    "--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Verbosity level of the custom logger."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in ["FAST-RGB-Waypoint", "FAST-Depth-Waypoint"]:
    args_cli.enable_cameras = True
elif args_cli.task not in ["FAST-Quadcopter-Bodyrate", "FAST-Quadcopter-Vel", "FAST-Quadcopter-Waypoint", "FAST-Swarm-Bodyrate", "FAST-Swarm-Waypoint"]:
    raise ValueError(
        "Invalid task name #^# Please select from: FAST-Quadcopter-Bodyrate; FAST-Quadcopter-Vel; FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint; FAST-Swarm-Bodyrate; FAST-Swarm-Waypoint."
    )
if args_cli.video:
    args_cli.enable_cameras = True
# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

# TODO: Improve import modality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datetime import datetime
import gymnasium as gym
from loguru import logger
import random
import rclpy
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from isaaclab_tasks.utils.hydra import hydra_task_config

from policy import DistributedActorCentralizedCriticPolicy
from envs import camera_waypoint_env, quadcopter_bodyrate_env, quadcopter_waypoint_env, swarm_bodyrate_env, swarm_waypoint_env


@hydra_task_config(args_cli.task, "sb3_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with stable-baselines agent."""

    # Override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Set the environment seed
    # Note: certain randomizations occur in the environment initialization so we set the seed here
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    if args_cli.max_iterations is not None:
        agent_cfg["n_timesteps"] = args_cli.max_iterations * agent_cfg["n_steps"] * env_cfg.scene.num_envs

    log_root_path = os.path.abspath(os.path.join("outputs", "sb3", args_cli.task, "flowline"))
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(log_root_path, run_info)
    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    env_dir = os.path.join(os.path.dirname(__file__), "../../", "envs")
    dump_env_src_dir = os.path.join(log_dir, "src")
    os.makedirs(dump_env_src_dir, exist_ok=True)
    if args_cli.task == "FAST-Quadcopter-Bodyrate":
        env_src_file = "quadcopter_bodyrate_env.py"
    elif args_cli.task == "FAST-Quadcopter-Vel":
        env_src_file = "quadcopter_vel_env.py"
    elif args_cli.task == "FAST-Quadcopter-Waypoint":
        env_src_file = "quadcopter_waypoint_env.py"
    elif args_cli.task in ["FAST-RGB-Waypoint", "FAST-Depth-Waypoint"]:
        env_src_file = "camera_waypoint_env.py"
    elif args_cli.task == "FAST-Swarm-Bodyrate":
        env_src_file = "swarm_bodyrate_env.py"
    elif args_cli.task == "FAST-Swarm-Waypoint":
        env_src_file = "swarm_waypoint_env.py"
    shutil.copy2(os.path.join(env_dir, env_src_file), os.path.join(dump_env_src_dir, env_src_file))

    # Post-process agent configuration
    agent_cfg = process_sb3_cfg(agent_cfg)
    n_timesteps = agent_cfg.pop("n_timesteps")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    save_interval = max(args_cli.save_interval // env_cfg.scene.num_envs, 1)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: (step + 1) % save_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "training-fragment",
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = Sb3VecEnvWrapper(env)

    if args_cli.checkpoint:
        agent = PPO.load(args_cli.checkpoint, env=env, device=agent_cfg["device"])
    else:
        if args_cli.task in ["FAST-Swarm-Bodyrate", "FAST-Swarm-Waypoint"]:
            agent_cfg["policy_kwargs"]["num_agents"] = env_cfg.num_drones
            agent = PPO(DistributedActorCentralizedCriticPolicy, env, verbose=2, **agent_cfg)
        else:
            policy_arch = agent_cfg.pop("policy")
            agent = PPO(policy_arch, env, verbose=2, **agent_cfg)

    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    agent.set_logger(new_logger)

    checkpoint_callback = CheckpointCallback(save_freq=save_interval, save_path=os.path.join(log_dir, "models"), name_prefix="model", verbose=2)

    agent.learn(total_timesteps=n_timesteps, callback=checkpoint_callback, progress_bar=True)

    agent.save(os.path.join(log_dir, "models", "model_final"))

    env.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level=args_cli.verbosity)

    rclpy.init()
    main()
    rclpy.shutdown()

    simulation_app.close()
