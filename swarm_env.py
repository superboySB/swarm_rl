from __future__ import annotations

import gymnasium as gym
from loguru import logger
import math
import time
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils import quat_to_ang_between_z_body_and_z_world
from controller import Controller


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment"""

    def __init__(self, env: QuadcopterSwarmEnv, window_name: str = "IsaacLab"):
        """Initialize the window

        Args:
            env: The environment object
            window_name: The name of the window. Defaults to "IsaacLab"
        """
        # Initialize base window
        super().__init__(env, window_name)
        # Add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # Add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class QuadcopterSwarmEnvCfg(DirectMARLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(-5.0, -5.0, 2.5))

    # Env
    episode_length_s = 60.0
    decimation = 2
    num_drones = 9  # Number of drones per environment
    possible_agents = [f"drone_{i}" for i in range(num_drones)]
    action_spaces = {agent: 14 for agent in possible_agents}
    observation_spaces = {agent: 13 for agent in possible_agents}
    state_space = 0
    debug_vis = False

    ui_window_class_type = QuadcopterEnvWindow

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation * 5,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=6, replicate_physics=True)

    # Robot
    drone_cfg: ArticulationCfg = DJI_FPV_CFG.copy()
    p_max = {agent: 500.0 for agent in possible_agents}
    v_max = {agent: 5.0 for agent in possible_agents}
    yaw_dot_max = {agent: math.pi / 2 for agent in possible_agents}
    init_gap = 0.8


class QuadcopterSwarmEnv(DirectMARLEnv):
    cfg: QuadcopterSwarmEnvCfg

    def __init__(self, cfg: QuadcopterSwarmEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.actions = {}
        self.a_desired_total, self.thrust_desired, self._thrust_desired, self.q_desired, self.w_desired, self.m_desired = {}, {}, {}, {}, {}, {}

        # Get specific body indices for each drone
        self.body_ids = {agent: self.robots[agent].find_bodies("body")[0] for agent in self.cfg.possible_agents}

        self.robot_masses = {agent: self.robots[agent].root_physx_view.get_masses()[0, 0] for agent in self.cfg.possible_agents}
        self.robot_inertias = {agent: self.robots[agent].root_physx_view.get_inertias()[0, 0] for agent in self.cfg.possible_agents}
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # Controllers
        self.controllers = {
            agent: Controller(
                self.step_dt,
                self.gravity,
                self.robot_masses[agent].to(self.device),
                self.robot_inertias[agent].to(self.device),
                self.num_envs,
                self.device,
            )
            for agent in self.cfg.possible_agents
        }

    def _setup_scene(self):
        self.robots = {}
        points_per_side = math.ceil(math.sqrt(self.cfg.num_drones))
        side_length = (points_per_side - 1) * self.cfg.init_gap
        for i, agent in enumerate(self.cfg.possible_agents):
            row = i // points_per_side
            col = i % points_per_side
            init_pos = (col * self.cfg.init_gap - side_length / 2, row * self.cfg.init_gap - side_length / 2, 1.0)

            drone = Articulation(
                self.cfg.drone_cfg.replace(
                    prim_path=f"/World/envs/env_.*/Robot_{i}",
                    init_state=self.cfg.drone_cfg.init_state.replace(pos=init_pos),
                )
            )
            self.robots[agent] = drone
            self.scene.articulations[agent] = drone

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # TODO: to be deleted
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        start = time.perf_counter()
        for agent in self.possible_agents:
            self.actions[agent] = actions[agent].clone().clamp(-1.0, 1.0)
            self.actions[agent][:, :3] *= self.cfg.p_max[agent]
            self.actions[agent][:, :3] += self.terrain.env_origins
            self.actions[agent][:, 3:6] *= self.cfg.v_max[agent]
            self.actions[agent][:, 12] *= math.pi
            self.actions[agent][:, 13] *= self.cfg.yaw_dot_max[agent]

            self.a_desired_total[agent], self.thrust_desired[agent], self.q_desired[agent], self.w_desired[agent], self.m_desired[agent] = (
                self.controllers[agent].get_control(self.robots[agent].data.root_link_state_w, self.actions[agent])
            )
            self._thrust_desired[agent] = torch.cat((torch.zeros(self.num_envs, 2, device=self.device), self.thrust_desired[agent].unsqueeze(1)), dim=1)
        end = time.perf_counter()
        logger.debug(f"get_controls takes {end - start:.6f}s")

    def _apply_action(self) -> None:
        for agent in self.possible_agents:
            self.robots[agent].set_external_force_and_torque(
                self._thrust_desired[agent].unsqueeze(1), self.m_desired[agent].unsqueeze(1), body_ids=self.body_ids[agent]
            )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {}
        for agent in self.possible_agents:
            odom = self.robots[agent].data.root_link_state_w.clone()
            odom[:, :3] -= self.terrain.env_origins
            observations[agent] = odom
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards = {agent: 0 for agent in self.possible_agents}
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:
            z_exceed_bounds = torch.logical_or(
                self.robots[agent].data.root_link_pos_w[:, 2] < -0.1, self.robots[agent].data.root_link_pos_w[:, 2] > 10.0
            )
            ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robots[agent].data.root_link_quat_w))
            _died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 60.0)
            died = torch.logical_or(died, _died)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return {agent: died for agent in self.cfg.possible_agents}, {agent: time_out for agent in self.cfg.possible_agents}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["drone_0"]._ALL_INDICES

        for agent in self.possible_agents:
            self.robots[agent].reset(env_ids)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Sample new commands for each drone
        for agent in self.possible_agents:
            # Reset robot state
            joint_pos = self.robots[agent].data.default_joint_pos[env_ids]
            joint_vel = self.robots[agent].data.default_joint_vel[env_ids]
            default_root_state = self.robots[agent].data.default_root_state[env_ids]
            default_root_state[:, :3] += self.terrain.env_origins[env_ids]
            self.robots[agent].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[agent].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[agent].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)


gym.register(
    id="FAST-Quadcopter-Swarm-Direct-v0",
    entry_point=QuadcopterSwarmEnv,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": QuadcopterSwarmEnvCfg},
)
