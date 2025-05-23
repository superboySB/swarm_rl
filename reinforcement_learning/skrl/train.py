"""Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys
from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Name of the task. Optional includes: FAST-Quadcopter-Waypoint-v0; FAST-Quadcopter-Bodyrate-v0; FAST-Quadcopter-RGB-Camera-v0; FAST-Quadcopter-Depth-Camera-v0; FAST-Quadcopter-Swarm-Direct-v0.",
)
parser.add_argument("--num_envs", type=int, default=10000, help="Number of environments to simulate.")
parser.add_argument("--sim_device", type=str, default="cuda:0", help="Device to run the simulation on.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL policy training iterations.")
parser.add_argument("--save_interval", type=int, default=None, help="Interval between checkpoints (in steps).")
parser.add_argument("--video", action="store_true", default=True, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=100, help="Length of the recorded video (in steps).")
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--initial_log_std", type=float, default=None, help="Initial log standard deviation of the Gaussian model.")
parser.add_argument("--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes.")
parser.add_argument("--ml_framework", type=str, default="torch", choices=["torch", "jax", "jax-numpy"], help="The ML framework used for training the skrl agent.")
parser.add_argument("--algorithm", type=str, default="PPO", choices=["AMP", "PPO", "IPPO", "MAPPO"], help="The RL algorithm used for training the skrl agent.")
parser.add_argument(
    "--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Verbosity level of the custom logger."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.task is None:
    raise ValueError("The task argument is required and cannot be None.")
elif args_cli.task in ["FAST-Quadcopter-RGB-Camera-v0", "FAST-Quadcopter-Depth-Camera-v0"]:
    args_cli.enable_cameras = True
elif args_cli.task not in ["FAST-Quadcopter-Waypoint-v0", "FAST-Quadcopter-Bodyrate-v0", "FAST-Quadcopter-Swarm-Direct-v0"]:
    raise ValueError(
        "Invalid task name #^# Please select from: FAST-Quadcopter-Waypoint-v0; FAST-Quadcopter-Bodyrate-v0; FAST-Quadcopter-RGB-Camera-v0; FAST-Quadcopter-Depth-Camera-v0; FAST-Quadcopter-Swarm-Direct-v0."
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
import os
import random
import rclpy
import shutil
import torch

import skrl
from packaging import version

# Check for minimum supported skrl version
SKRL_VERSION = "1.4.2"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(f"Unsupported skrl version: {skrl.__version__}. " f"Install supported version using 'pip install skrl>={SKRL_VERSION}'")
    exit()
if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config

from envs import camera_waypoint_env, quadcopter_bodyrate_env, quadcopter_waypoint_env, swarm_env

# Config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""

    # Override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.sim_device if args_cli.sim_device is not None else env_cfg.sim.device
    # Set the agent and environment seed from command line
    # Note: certain randomization occur in the environment initialization so we set the seed here
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]

    # Multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # Configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    log_root_path = os.path.abspath(os.path.join("outputs", "skrl", args_cli.task, "flowline"))
    run_info = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    # Set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_info
    log_dir = os.path.join(log_root_path, run_info)

    # FIXME: Not robust 并非鲁棒 :(
    if args_cli.save_interval:
        save_interval = max(args_cli.save_interval, 1)
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = save_interval
    else:
        save_interval = agent_cfg["agent"]["experiment"]["checkpoint_interval"]
    agent_cfg["agent"]["experiment"]["write_interval"] = int(save_interval / 100)

    # Dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    env_dir = os.path.join(os.path.dirname(__file__), "../../", "envs")
    dump_env_src_dir = os.path.join(log_dir, "src")
    os.makedirs(dump_env_src_dir, exist_ok=True)
    if args_cli.task in ["FAST-Quadcopter-Waypoint-v0", "FAST-Quadcopter-Bodyrate-v0"]:
        env_src_file = "quadcopter_env.py"
    elif args_cli.task in ["FAST-Quadcopter-RGB-Camera-v0", "FAST-Quadcopter-Depth-Camera-v0"]:
        env_src_file = "camera_env.py"
    elif args_cli.task == "FAST-Quadcopter-Swarm-Direct-v0":
        env_src_file = "swarm_env.py"
    shutil.copy2(os.path.join(env_dir, env_src_file), os.path.join(dump_env_src_dir, env_src_file))

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: (step + 1) % save_interval == 0,
            "video_length": args_cli.video_length,
            "name_prefix": "training-fragment",
            "disable_logger": True,
        }
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # Same as: wrap_env(env, wrapper="auto")

    # Configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    if resume_path:
        runner.agent.load(resume_path)

        # FIXME: Improve to be more rational
        if args_cli.initial_log_std is not None:
            with torch.no_grad():
                runner.agent.checkpoint_modules["policy"].log_std_parameter.data.fill_(args_cli.initial_log_std)

    runner.run()

    env.close()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level=args_cli.verbosity)

    rclpy.init()
    main()
    rclpy.shutdown()

    simulation_app.close()
