# Copyright (c) 2022-2025, 99.9% The Isaac Lab Project Developers, 0.1% LAJi.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from isaaclab.app import AppLauncher

# Local imports
import cli_args  # isort: skip


# Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Name of the task. Optional includes: FAST-Quadcopter-Bodyrate; FAST-Quadcopter-Vel; FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint; FAST-Swarm-Bodyrate; FAST-Swarm-Vel; FAST-Swarm-Waypoint.",
)
parser.add_argument("--num_envs", type=int, default=1000, help="Number of environments to simulate.")
parser.add_argument("--sim_device", type=str, default="cuda:0", help="Device to run the simulation on.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument("--save_interval", type=int, default=None, help="Interval between checkpoints (in iters).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument("--init_log_std", type=float, default=None, help="Initial log standard deviation of the Gaussian model.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument(
    "--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Verbosity level of the custom logger."
)

# Append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in ["FAST-RGB-Waypoint", "FAST-Depth-Waypoint"]:
    args_cli.enable_cameras = True
elif args_cli.task not in ["FAST-Quadcopter-Bodyrate", "FAST-Quadcopter-Vel", "FAST-Quadcopter-Waypoint", "FAST-Swarm-Bodyrate", "FAST-Swarm-Vel", "FAST-Swarm-Waypoint"]:
    raise ValueError(
        "Invalid task name #^# Please select from: FAST-Quadcopter-Bodyrate; FAST-Quadcopter-Vel; FAST-Quadcopter-Waypoint; FAST-RGB-Waypoint; FAST-Depth-Waypoint; FAST-Swarm-Bodyrate; FAST-Swarm-Vel; FAST-Swarm-Waypoint."
    )
if args_cli.video:
    args_cli.enable_cameras = True
# Clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# For distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)


"""Rest everything follows."""

# TODO: Improve import modality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from datetime import datetime
import gymnasium as gym
from loguru import logger
import rclpy
import shutil
import torch

from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from envs import camera_waypoint_env, quadcopter_bodyrate_env, quadcopter_waypoint_env, swarm_bodyrate_env, swarm_vel_env, swarm_waypoint_env

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg: DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""

    # Override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    # Set the environment seed
    # Note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    agent_cfg.max_iterations = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    # Multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # Set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    log_root_path = os.path.abspath(os.path.join("outputs", "rsl_rl", args_cli.task, "flowline"))
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_info += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, run_info)

    # FIXME: Not robust 并非鲁棒 :(
    if args_cli.save_interval:
        agent_cfg.save_interval = max(args_cli.save_interval, 1)
    video_interval = agent_cfg.save_interval * agent_cfg.num_steps_per_env

    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    env_dir = os.path.join(os.path.dirname(__file__), "../../", "envs")
    dump_env_src_dir = os.path.join(log_dir, "src")
    os.makedirs(dump_env_src_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
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
    elif args_cli.task == "FAST-Swarm-Vel":
        env_src_file = "swarm_vel_env.py"
    elif args_cli.task == "FAST-Swarm-Waypoint":
        env_src_file = "swarm_waypoint_env.py"
    shutil.copy2(os.path.join(env_dir, env_src_file), os.path.join(dump_env_src_dir, env_src_file))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: (step + 1) % video_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "training-fragment",
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions[0])

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # Write git state to logs
    runner.add_git_repo_to_log(__file__)
    if args_cli.checkpoint:
        runner.load(args_cli.checkpoint)

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    env.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level=args_cli.verbosity)

    rclpy.init()
    main()
    rclpy.shutdown()

    simulation_app.close()
