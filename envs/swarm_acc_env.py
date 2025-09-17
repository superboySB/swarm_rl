from __future__ import annotations

import gymnasium as gym
import math
import random
import time
import torch
from collections import deque
from collections.abc import Sequence
from loguru import logger

from rclpy.node import Node
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import AccelStamped, Vector3Stamped

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.utils.math import quat_inv, quat_apply

from envs.quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils.utils import quat_to_ang_between_z_body_and_z_world
from utils.controller import Controller


@configclass
class SwarmAccEnvCfg(DirectMARLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 10.0))

    # Reward weights
    to_live_reward_weight = 1.0  # 《活着》
    death_penalty_weight = 0.0
    approaching_goal_reward_weight = 1.0
    success_reward_weight = 10.0
    mutual_collision_penalty_weight = 10.0
    mutual_collision_avoidance_soft_penalty_weight = 0.1
    ang_vel_penalty_weight = 0.02
    action_norm_penalty_weight = 0.02
    action_norm_near_goal_penalty_weight = 1.0
    action_diff_penalty_weight = 0.02
    # Exponential decay factors and tolerances
    mutual_collision_avoidance_reward_scale = 1.0

    # Mission settings
    flight_range = 4.5
    flight_altitude = 1.0  # Desired flight altitude
    safe_dist = 1.3
    collide_dist = 0.8
    goal_reset_delay = 1.0  # Delay for resetting goal after reaching it
    mission_names = ["migration", "crossover", "chaotic"]
    # mission_prob = [0.0, 0.2, 0.8]
    # mission_prob = [1.0, 0.0, 0.0]
    mission_prob = [0.0, 1.0, 0.0]
    # mission_prob = [0.0, 0.0, 1.0]
    success_distance_threshold = 0.25  # Distance threshold for considering goal reached
    max_sampling_tries = 100  # Maximum number of attempts to sample a valid initial state or goal
    lowpass_filter_cutoff_freq = 10000.0
    torque_ctrl_delay_s = 0.0
    realistic_ctrl = True
    # Params for mission crossover
    fix_range = False
    uniformly_distributed_prob = 0.7

    # Observation parameters
    max_visible_distance = 5.0
    max_angle_of_view = 40.0  # Maximum field of view of camera in tilt direction
    # Domain randomization
    enable_domain_randomization = True
    max_dist_noise_std = 0.5
    max_bearing_noise_std = 0.05
    drop_prob = 0.1

    # Parameters for environment and agents
    episode_length_s = 30.0
    physics_freq = 200.0
    control_freq = 50.0
    action_freq = 15.0
    gui_render_freq = 50.0
    control_decimation = physics_freq // control_freq
    num_drones = 6  # Number of drones per environment
    decimation = math.ceil(physics_freq / action_freq)  # Environment decimation
    render_decimation = physics_freq // gui_render_freq
    clip_action = 1.0
    possible_agents = None
    action_spaces = None
    history_length = 5
    history_buffer_interval = 0.1
    history_buffer_scroll_decimation = action_freq // (1 / history_buffer_interval)
    self_observation_dim = 6
    relative_observation_dim = 4
    transient_observasion_dim = self_observation_dim + relative_observation_dim * (num_drones - 1)
    # transient_observasion_dim = 8
    observation_spaces = None
    transient_state_dim = 18 * num_drones
    state_space = transient_state_dim
    possible_agents = [f"drone_{i}" for i in range(num_drones)]
    action_spaces = {agent: 2 for agent in possible_agents}
    a_max = {agent: 10.0 for agent in possible_agents}
    v_max = {agent: 5.0 for agent in possible_agents}
    def __post_init__(self):
        self.observation_spaces = {agent: self.history_length * self.transient_observasion_dim for agent in self.possible_agents}

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / physics_freq,
        render_interval=render_decimation,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1000, env_spacing=15, replicate_physics=True)

    # Robot
    drone_cfg: ArticulationCfg = DJI_FPV_CFG.copy()
    init_gap = 2.0  # TODO: Redundant feature, to be removed o_0

    # Debug visualization
    debug_vis = True
    debug_vis_goal = True
    debug_vis_collide_dist = False
    debug_vis_rel_pos = False


class SwarmAccEnv(DirectMARLEnv):
    cfg: SwarmAccEnvCfg

    def __init__(self, cfg: SwarmAccEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Action and control decimation must be greater than or equal to 1 #^#")

        self.goals = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.env_mission_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_goal_timer = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.cfg.possible_agents}
        self.success_dist_thr = torch.zeros(self.num_envs, device=self.device)

        self.mission_prob = torch.tensor(self.cfg.mission_prob, device=self.device)
        # Mission migration params
        self.unified_goal_xy = torch.zeros(self.num_envs, 2, device=self.device)
        # Mission crossover params
        self.rand_r = torch.zeros(self.num_envs, device=self.device)
        self.ang = torch.zeros(self.num_envs, self.cfg.num_drones, device=self.device)

        self.body_ids = {agent: self.robots[agent].find_bodies("body")[0] for agent in self.cfg.possible_agents}  # Get specific body indices for each drone
        self.robot_masses = {agent: self.robots[agent].root_physx_view.get_masses()[0, 0].to(self.device) for agent in self.cfg.possible_agents}
        self.robot_inertias = {agent: self.robots[agent].root_physx_view.get_inertias()[0, 0].to(self.device) for agent in self.cfg.possible_agents}
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)
        self.robot_weights = {agent: (self.robot_masses[agent] * self.gravity.norm()).item() for agent in self.cfg.possible_agents}

        # Normalized actions
        self.actions = {agent: torch.zeros(self.num_envs, self.cfg.action_spaces[agent], device=self.device) for agent in self.cfg.possible_agents}
        self.prev_actions = {agent: torch.zeros(self.num_envs, self.cfg.action_spaces[agent], device=self.device) for agent in self.cfg.possible_agents}

        # Denormalized actions
        self.p_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.v_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.a_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.j_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.yaw_desired = {agent: torch.zeros(self.num_envs, 1, device=self.device) for agent in self.cfg.possible_agents}
        self.yaw_dot_desired = {agent: torch.zeros(self.num_envs, 1, device=self.device) for agent in self.cfg.possible_agents}

        # Controller
        self.a_desired_total = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.thrust_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.q_desired = {agent: torch.zeros(self.num_envs, 4, device=self.device) for agent in self.cfg.possible_agents}
        self.w_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.m_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}

        self.controller = Controller(
            1 / self.cfg.control_freq,
            self.gravity,
            self.robot_masses["drone_0"].to(self.device),  # Assuming homogeneous swarm
            self.robot_inertias["drone_0"].to(self.device),
            self.num_envs * self.cfg.num_drones,
        )
        self.control_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Low-pass filter for smoothing input signal
        self.lowpass_filter_alpha = (2 * math.pi * self.cfg.lowpass_filter_cutoff_freq * self.physics_dt) / (
            2 * math.pi * self.cfg.lowpass_filter_cutoff_freq * self.physics_dt + 1
        )
        self.a_desired_smoothed = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.prev_a_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}

        # Artificial delay for torque control
        self.delay_steps = max(math.ceil(cfg.torque_ctrl_delay_s / self.physics_dt), 1)
        self.thrust_buffer = {agent: deque([torch.zeros(self.num_envs, 3, device=self.device) for _ in range(self.delay_steps)]) for agent in self.cfg.possible_agents}
        self.m_buffer = {agent: deque([torch.zeros(self.num_envs, 3, device=self.device) for _ in range(self.delay_steps)]) for agent in self.cfg.possible_agents}

        self.prev_dist_to_goals = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.cfg.possible_agents}

        self.relative_positions_w = {
            i: {j: torch.zeros(self.num_envs, 3, device=self.device) for j in range(self.cfg.num_drones) if j != i} for i in range(self.cfg.num_drones)
        }
        self.relative_positions_with_observability = {}
        self.last_observable_relative_positions = {
            agent: torch.zeros(self.num_envs, self.cfg.relative_observation_dim * (self.cfg.num_drones - 1), device=self.device) for agent in self.cfg.possible_agents
        }

        self.died = {agent: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for agent in self.cfg.possible_agents}
        self.reset_history_buffer = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.observation_buffer = {
            agent: torch.zeros(self.cfg.history_length, self.num_envs, self.cfg.transient_observasion_dim, device=self.device) for agent in self.cfg.possible_agents
        }
        self.scroll_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # Logging
        self.episode_sums = {}

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # ROS2
        self.node = Node("swarm_acc_env", namespace="swarm_acc_env")
        self.odom_pub = self.node.create_publisher(Odometry, "odom", 10)
        self.action_pub = self.node.create_publisher(Odometry, "action", 10)
        if self.cfg.realistic_ctrl:
            self.a_desired_pub = self.node.create_publisher(AccelStamped, "a_desired", 10)
            self.a_desired_total_pub = self.node.create_publisher(AccelStamped, "a_desired_total", 10)
            self.m_desired_pub = self.node.create_publisher(Vector3Stamped, "m_desired", 10)

    def _setup_scene(self):
        self.robots = {}
        points_per_side = math.ceil(math.sqrt(self.cfg.num_drones))
        side_length = (points_per_side - 1) * self.cfg.init_gap
        for i, agent in enumerate(self.cfg.possible_agents):
            row = i // points_per_side
            col = i % points_per_side
            init_pos = (col * self.cfg.init_gap - side_length / 2, row * self.cfg.init_gap - side_length / 2, self.cfg.flight_altitude)

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

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # TODO: Where would it make more sense to place ⬇️?
        self.reset_history_buffer[:] = False

        for agent in self.possible_agents:
            # Denormalize and clip the input signal
            self.actions[agent] = actions[agent].clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action
            a_xy_desired = self.actions[agent] * self.cfg.a_max[agent]
            norm_xy = torch.norm(a_xy_desired, dim=1, keepdim=True)
            clip_scale = torch.clamp(norm_xy / self.cfg.a_max[agent], min=1.0)
            self.a_desired[agent][:, :2] = a_xy_desired / clip_scale

    def _apply_action(self) -> None:
        prev_v_desired, a_after_v_clip = {}, {}
        for agent in self.possible_agents:
            # Low-pass filtering acceleration cmd
            self.a_desired_smoothed[agent] = self.lowpass_filter_alpha * self.a_desired[agent] + (1.0 - self.lowpass_filter_alpha) * self.prev_a_desired[agent]
            self.prev_a_desired[agent] = self.a_desired_smoothed[agent].clone()

            # Clip velocity cmd
            prev_v_desired[agent] = self.v_desired[agent].clone()
            self.v_desired[agent][:, :2] += self.a_desired_smoothed[agent][:, :2] * self.physics_dt
            speed_xy = torch.norm(self.v_desired[agent][:, :2], dim=1, keepdim=True)
            clip_scale = torch.clamp(speed_xy / self.cfg.v_max[agent], min=1.0)
            self.v_desired[agent][:, :2] /= clip_scale

            # Update acceleration cmd after velocity clipping
            a_after_v_clip[agent] = (self.v_desired[agent] - prev_v_desired[agent]) / self.physics_dt

            self.p_desired[agent][:, :2] += prev_v_desired[agent][:, :2] * self.physics_dt + 0.5 * a_after_v_clip[agent][:, :2] * self.physics_dt**2

        ### ============= Realistic acceleration tracking ============= ###

        if self.cfg.realistic_ctrl:
            get_control_idx = (self.control_counter % self.cfg.control_decimation == 0).nonzero(as_tuple=False).squeeze(-1)
            if len(get_control_idx) > 0:
                start = time.perf_counter()

                # Parallel calculation of low-level control for all drones
                root_state_w_all = [self.robots[agent].data.root_state_w[get_control_idx] for agent in self.possible_agents]
                root_state_w_all = torch.cat(root_state_w_all, dim=0)
                state_desired_all = []
                for agent in self.possible_agents:
                    # Concatenate into full-state command
                    state_desired = torch.cat(
                        (
                            self.p_desired[agent][get_control_idx],
                            self.v_desired[agent][get_control_idx],
                            # self.a_desired[agent][get_control_idx],
                            a_after_v_clip[agent][get_control_idx],
                            self.j_desired[agent][get_control_idx],
                            self.yaw_desired[agent][get_control_idx],
                            self.yaw_dot_desired[agent][get_control_idx],
                        ),
                        dim=1,
                    )
                    state_desired_all.append(state_desired)
                state_desired_all = torch.cat(state_desired_all, dim=0)

                # Compute low-level control
                a_desired_total_all, _thrust_desired_all, q_desired_all, w_desired_all, m_desired_all = self.controller.get_control(
                    root_state_w_all, state_desired_all, self.env_ids_to_ctrl_ids(get_control_idx)
                )

                # Converting 1-dim thrust cmd to force cmd in 3-dim body frame
                num_get_ctrl_envs = len(get_control_idx)
                thrust_desired_all = torch.cat((torch.zeros(self.cfg.num_drones * num_get_ctrl_envs, 2, device=self.device), _thrust_desired_all.unsqueeze(1)), dim=1)

                # Split the parallel computing result among each drone
                a_chunks = torch.split(a_desired_total_all, num_get_ctrl_envs, dim=0)
                thrust_chunks = torch.split(thrust_desired_all, num_get_ctrl_envs, dim=0)
                q_chunks = torch.split(q_desired_all, num_get_ctrl_envs, dim=0)
                w_chunks = torch.split(w_desired_all, num_get_ctrl_envs, dim=0)
                m_chunks = torch.split(m_desired_all, num_get_ctrl_envs, dim=0)

                for i, agent in enumerate(self.possible_agents):
                    self.a_desired_total[agent][get_control_idx] = a_chunks[i]
                    self.thrust_desired[agent][get_control_idx] = thrust_chunks[i]
                    self.q_desired[agent][get_control_idx] = q_chunks[i]
                    self.w_desired[agent][get_control_idx] = w_chunks[i]
                    self.m_desired[agent][get_control_idx] = m_chunks[i]

                end = time.perf_counter()
                logger.debug(f"get_control for all drones takes {end - start:.5f}s")

                self.control_counter[get_control_idx] = 0
            self.control_counter += 1

            self._publish_debug_signals()

            for agent in self.possible_agents:
                # Artificial delay for ideal force and torque control
                delayed_thrust = self.thrust_buffer[agent].popleft()
                delayed_m = self.m_buffer[agent].popleft()
                self.thrust_buffer[agent].append(self.thrust_desired[agent].clone())
                self.m_buffer[agent].append(self.m_desired[agent].clone())

                self.robots[agent].set_external_force_and_torque(delayed_thrust.unsqueeze(1), delayed_m.unsqueeze(1), body_ids=self.body_ids[agent])

        ### ============= Ideal acceleration tracking ============= ###

        else:
            self._publish_debug_signals()

            for agent in self.possible_agents:
                v_desired = self.v_desired[agent].clone()
                v_desired[:, 2] += 100.0 * (self.p_desired[agent][:, 2] - self.robots[agent].data.root_pos_w[:, 2])
                # Set angular velocity to zero, treat the rigid body as a particle
                self.robots[agent].write_root_velocity_to_sim(torch.cat((v_desired, torch.zeros_like(v_desired)), dim=1))

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        died_unified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:

            z_exceed_bounds = torch.logical_or(self.robots[agent].data.root_link_pos_w[:, 2] < 0.9, self.robots[agent].data.root_link_pos_w[:, 2] > 1.1)
            ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robots[agent].data.root_link_quat_w))
            self.died[agent] = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 80.0)

            x_exceed_bounds = torch.logical_or(
                self.robots[agent].data.root_link_pos_w[:, 0] - self.terrain.env_origins[:, 0] < -self.cfg.flight_range,
                self.robots[agent].data.root_link_pos_w[:, 0] - self.terrain.env_origins[:, 0] > self.cfg.flight_range,
            )
            y_exceed_bounds = torch.logical_or(
                self.robots[agent].data.root_link_pos_w[:, 1] - self.terrain.env_origins[:, 1] < -self.cfg.flight_range,
                self.robots[agent].data.root_link_pos_w[:, 1] - self.terrain.env_origins[:, 1] > self.cfg.flight_range,
            )
            self.died[agent] = torch.logical_or(self.died[agent], torch.logical_or(x_exceed_bounds, y_exceed_bounds))

            died_unified = torch.logical_or(died_unified, self.died[agent])

        # Update relative positions, detecting collisions along the way
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j] = self.robots[agent_j].data.root_pos_w - self.robots[agent_i].data.root_pos_w

                # collision = torch.linalg.norm(self.relative_positions_w[i][j], dim=1) < self.cfg.collide_dist
                # self.died[agent_i] = torch.logical_or(self.died[agent_i], collision)
            # died_unified = torch.logical_or(died_unified, self.died[agent_i])

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return {agent: died_unified for agent in self.cfg.possible_agents}, {agent: time_out for agent in self.cfg.possible_agents}

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards = {}

        for i, agent in enumerate(self.possible_agents):
            # Reward for avoiding collisions with other drones
            collide = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            mutual_collision_avoidance_soft_reward = torch.zeros(self.num_envs, device=self.device)
            for j, _ in enumerate(self.possible_agents):
                if i == j:
                    continue

                dist_btw_drones = torch.linalg.norm(self.relative_positions_w[i][j], dim=1)
                collide = torch.logical_or(collide, dist_btw_drones < self.cfg.collide_dist)

                collision_soft_penalty = torch.where(
                    dist_btw_drones < self.cfg.safe_dist,
                    torch.exp(self.cfg.mutual_collision_avoidance_reward_scale * (self.cfg.safe_dist - dist_btw_drones)) - 1.0,
                    torch.zeros(self.num_envs, device=self.device),
                )
                mutual_collision_avoidance_soft_reward -= collision_soft_penalty

            mutual_collision_reward = -torch.where(
                collide,
                torch.ones(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )

            # Reward for encouraging drones to approach the goal
            dist_to_goal = torch.linalg.norm(self.goals[agent] - self.robots[agent].data.root_pos_w, dim=1)
            approaching_goal_reward = self.prev_dist_to_goals[agent] - dist_to_goal
            self.prev_dist_to_goals[agent] = dist_to_goal

            # Additional reward when the drone is close to goal
            success_i = dist_to_goal < self.success_dist_thr
            success_reward = torch.where(
                success_i,
                torch.ones(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )

            death_reward = -torch.where(
                self.died[agent],
                torch.ones(self.num_envs, device=self.device),
                torch.zeros(self.num_envs, device=self.device),
            )

            ### ============= Smoothing ============= ###
            ang_vel_reward = -torch.linalg.norm(self.robots[agent].data.root_ang_vel_w, dim=1)
            action_norm_reward = -torch.linalg.norm(self.actions[agent], dim=1)
            action_norm_near_goal_reward = torch.where(
                success_i,
                -torch.linalg.norm(self.actions[agent], dim=1),
                torch.zeros(self.num_envs, device=self.device),
            )
            action_diff_reward = -torch.linalg.norm(self.actions[agent] - self.prev_actions[agent], dim=1)
            self.prev_actions[agent] = self.actions[agent].clone()

            reward = {
                "meaning_to_live": torch.ones(self.num_envs, device=self.device) * self.cfg.to_live_reward_weight * self.step_dt,
                "approaching_goal": approaching_goal_reward * self.cfg.approaching_goal_reward_weight * self.step_dt,
                "success": success_reward * self.cfg.success_reward_weight * self.step_dt,
                "death_penalty": death_reward * self.cfg.death_penalty_weight,
                "mutual_collision_penalty": mutual_collision_reward * self.cfg.mutual_collision_penalty_weight * self.step_dt,
                "mutual_collision_avoidance_soft_penalty": mutual_collision_avoidance_soft_reward
                * self.cfg.mutual_collision_avoidance_soft_penalty_weight
                * self.step_dt,
                ### ============= Smoothing ============= ###
                "ang_vel_penalty": ang_vel_reward * self.cfg.ang_vel_penalty_weight * self.step_dt,
                "action_norm_penalty": action_norm_reward * self.cfg.action_norm_penalty_weight * self.step_dt,
                "action_norm_near_goal_penalty": action_norm_near_goal_reward * self.cfg.action_norm_near_goal_penalty_weight * self.step_dt,
                "action_diff_penalty": action_diff_reward * self.cfg.action_diff_penalty_weight * self.step_dt,
            }

            # Logging
            for key, value in reward.items():
                if key in self.episode_sums:
                    self.episode_sums[key] += value / self.cfg.num_drones
                else:
                    self.episode_sums[key] = value / self.cfg.num_drones

            reward = torch.sum(torch.stack(list(reward.values())), dim=0)

            rewards[agent] = reward
        return rewards

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["drone_0"]._ALL_INDICES

        # Logging
        extras = dict()
        for key in self.episode_sums.keys():
            episodic_sum_avg = torch.mean(self.episode_sums[key][env_ids])
            extras["Mean_Epi_Reward_of_Reset_Envs/" + key] = episodic_sum_avg
            self.episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)

        for agent in self.possible_agents:
            self.robots[agent].reset(env_ids)

        super()._reset_idx(env_ids)
        if self.num_envs > 13 and len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Randomly assign missions to reset envs
        self.env_mission_ids[env_ids] = torch.multinomial(self.mission_prob, num_samples=len(env_ids), replacement=True)
        mission_0_ids = env_ids[self.env_mission_ids[env_ids] == 0]  # The migration mission
        mission_1_ids = env_ids[self.env_mission_ids[env_ids] == 1]  # The crossover mission
        mission_2_ids = env_ids[self.env_mission_ids[env_ids] == 2]  # The chaotic mission

        self.success_dist_thr[mission_0_ids] = self.cfg.success_distance_threshold * self.cfg.num_drones / 1.414
        self.success_dist_thr[mission_1_ids] = self.cfg.success_distance_threshold
        self.success_dist_thr[mission_2_ids] = self.cfg.success_distance_threshold

        ### ============= Reset robot state and specify goal ============= ###
        start = time.perf_counter()
        # The migration mission: huddled init states + unified random target
        if len(mission_0_ids) > 0:
            rg = self.cfg.flight_range - self.success_dist_thr[mission_0_ids][0]
            self.unified_goal_xy[mission_0_ids] = torch.zeros(len(mission_0_ids), 2, device=self.device).uniform_(-rg, rg)

            rand_init_p_mis0 = torch.zeros(len(mission_0_ids), self.cfg.num_drones, 2, device=self.device)
            done = torch.zeros(len(mission_0_ids), dtype=torch.bool, device=self.device)

            for attempt in range(5 * self.cfg.max_sampling_tries):
                active = ~done
                if not torch.any(active):
                    break

                active_ids = active.nonzero(as_tuple=False).squeeze(-1)
                init_p = (torch.rand(active_ids.numel(), self.cfg.num_drones, 2, device=self.device) * 2 - 1) * rg  # [num_active, num_drones, 2]
                rand_init_p_mis0[active_ids] = init_p
                dmat = torch.cdist(init_p, init_p)  # [num_active, num_drones, num_drones]
                eye = torch.eye(self.cfg.num_drones, dtype=torch.bool, device=self.device).expand(active_ids.numel(), -1, -1)
                dmat.masked_fill_(eye, float("inf"))
                dmin = dmat.amin(dim=(-2, -1))  # [num_active]

                ok = dmin > self.cfg.collide_dist
                if torch.any(ok):
                    done[active_ids[ok]] = True

            if torch.any(~done):
                failed_ids = mission_0_ids[~done].tolist()
                logger.warning(
                    f"The search for initial positions of the swarm meeting constraints within a side-length {2 * rg} box failed for envs {failed_ids}, using the final sample #_#"
                )

        # The crossover mission: init states on a circle + target on the opposite side
        if len(mission_1_ids) > 0:
            r_max = self.cfg.flight_range - self.success_dist_thr[mission_1_ids][0] - 0.25
            if self.cfg.fix_range:
                r_min = r_max
            else:
                r_min = r_max / 1.5

            rand_r = torch.rand(len(mission_1_ids), device=self.device) * (r_max - r_min) + r_min
            ang = torch.empty((len(mission_1_ids), self.cfg.num_drones), device=self.device)

            if random.random() < self.cfg.uniformly_distributed_prob:
                N = self.cfg.num_drones
                base = torch.arange(N, dtype=torch.float32, device=self.device) * (2 * math.pi / N)
                rot = torch.rand(len(mission_1_ids), 1, device=self.device) * (2 * math.pi)
                ang_ = (base.unsqueeze(0).expand(len(mission_1_ids), -1) + rot) % (2 * math.pi)
                perms = torch.argsort(torch.rand(len(mission_1_ids), N, device=self.device), dim=1)
                ang = torch.gather(ang_, dim=1, index=perms)

            else:
                done = torch.zeros(len(mission_1_ids), dtype=torch.bool, device=self.device)
                for attempt in range(5 * self.cfg.max_sampling_tries):
                    active = ~done
                    if not torch.any(active):
                        break

                    active_ids = active.nonzero(as_tuple=False).squeeze(-1)
                    ang_ = torch.rand((active_ids.numel(), self.cfg.num_drones), device=self.device) * 2 * math.pi
                    ang[active_ids] = ang_
                    r = rand_r[active_ids].unsqueeze(-1)

                    pts = torch.stack([torch.cos(ang_) * r, torch.sin(ang_) * r], dim=-1)  # [num_active, num_drones, 2]
                    dmat = torch.cdist(pts, pts)  # [num_active, num_drones, num_drones]
                    eye = torch.eye(self.cfg.num_drones, dtype=torch.bool, device=self.device).expand(active_ids.numel(), -1, -1)
                    dmat.masked_fill_(eye, float("inf"))
                    dmin = dmat.amin(dim=(-2, -1))  # [num_active]

                    ok = dmin > self.cfg.collide_dist
                    if torch.any(ok):
                        done[active_ids[ok]] = True

                if torch.any(~done):
                    failed_ids = mission_1_ids[~done].tolist()
                    logger.warning(
                        f"The search for initial positions of the swarm meeting constraints on a circle failed for envs {failed_ids}, using the final sample #_#"
                    )

            self.rand_r[mission_1_ids] = rand_r
            self.ang[mission_1_ids] = ang

        # The chaotic mission: random init states + respective random target
        if len(mission_2_ids) > 0:
            rg = self.cfg.flight_range - self.success_dist_thr[mission_2_ids][0] - 0.25

            rand_init_p_mis2 = torch.zeros(len(mission_2_ids), self.cfg.num_drones, 2, device=self.device)
            done = torch.zeros(len(mission_2_ids), dtype=torch.bool, device=self.device)

            for attempt in range(5 * self.cfg.max_sampling_tries):
                active = ~done
                if not torch.any(active):
                    break

                active_ids = active.nonzero(as_tuple=False).squeeze(-1)
                init_p = (torch.rand(active_ids.numel(), self.cfg.num_drones, 2, device=self.device) * 2 - 1) * rg  # [num_active, num_drones, 2]
                rand_init_p_mis2[active_ids] = init_p
                dmat = torch.cdist(init_p, init_p)  # [num_active, num_drones, num_drones]
                eye = torch.eye(self.cfg.num_drones, dtype=torch.bool, device=self.device).expand(active_ids.numel(), -1, -1)
                dmat.masked_fill_(eye, float("inf"))
                dmin = dmat.amin(dim=(-2, -1))  # [num_active]

                ok = dmin > self.cfg.collide_dist
                if torch.any(ok):
                    done[active_ids[ok]] = True

            if torch.any(~done):
                failed_ids = mission_2_ids[~done].tolist()
                logger.warning(
                    f"The search for initial positions of the swarm meeting constraints within a side-length {2 * rg} box failed for envs {failed_ids}, using the final sample #_#"
                )

            rand_goal_p = torch.zeros(len(mission_2_ids), self.cfg.num_drones, 2, device=self.device)
            done = torch.zeros(len(mission_2_ids), dtype=torch.bool, device=self.device)

            for attempt in range(self.cfg.max_sampling_tries):
                active = ~done
                if not torch.any(active):
                    break

                active_ids = active.nonzero(as_tuple=False).squeeze(-1)
                goal_p = (torch.rand(active_ids.numel(), self.cfg.num_drones, 2, device=self.device) * 2 - 1) * rg  # [num_active, num_drones, 2]
                rand_goal_p[active_ids] = goal_p
                dmat = torch.cdist(goal_p, goal_p)  # [num_active, num_drones, num_drones]
                eye = torch.eye(self.cfg.num_drones, dtype=torch.bool, device=self.device).expand(active_ids.numel(), -1, -1)
                dmat.masked_fill_(eye, float("inf"))
                dmin = dmat.amin(dim=(-2, -1))  # [num_active]

                ok = dmin > self.cfg.collide_dist
                if torch.any(ok):
                    done[active_ids[ok]] = True

            if torch.any(~done):
                failed_ids = mission_2_ids[~done].tolist()
                logger.warning(
                    f"The search for goal positions of the swarm meeting constraints within a side-length {2 * rg} box failed for envs {failed_ids}, using the final sample #_#"
                )

        end = time.perf_counter()
        logger.debug(f"Random search for initial and goal positions takes {end - start:.5f}s")

        for i, agent in enumerate(self.possible_agents):
            init_state = self.robots[agent].data.default_root_state.clone()

            if len(mission_0_ids) > 0:
                init_state[mission_0_ids, :2] = rand_init_p_mis0[:, i]
                self.goals[agent][mission_0_ids, :2] = self.unified_goal_xy[mission_0_ids].clone()

            if len(mission_1_ids) > 0:
                ang = self.ang[mission_1_ids, i]
                r = self.rand_r[mission_1_ids].unsqueeze(-1)

                init_state[mission_1_ids, :2] = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1) * r

                ang += math.pi  # Terminate angles
                self.goals[agent][mission_1_ids, :2] = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1) * r

            if len(mission_2_ids) > 0:
                init_state[mission_2_ids, :2] = rand_init_p_mis2[:, i]
                self.goals[agent][mission_2_ids, :2] = rand_goal_p[:, i]

            init_state[env_ids, 2] = float(self.cfg.flight_altitude)
            init_state[env_ids, :3] += self.terrain.env_origins[env_ids]

            self.robots[agent].write_root_pose_to_sim(init_state[env_ids, :7], env_ids)
            self.robots[agent].write_root_velocity_to_sim(init_state[env_ids, 7:], env_ids)
            self.robots[agent].write_joint_state_to_sim(
                self.robots[agent].data.default_joint_pos[env_ids], self.robots[agent].data.default_joint_vel[env_ids], None, env_ids
            )

            self.goals[agent][env_ids, 2] = float(self.cfg.flight_altitude)
            self.goals[agent][env_ids] += self.terrain.env_origins[env_ids]
            self.reset_goal_timer[agent][env_ids] = 0.0

            self.actions[agent][env_ids] = torch.zeros_like(self.actions[agent][env_ids])
            self.prev_actions[agent][env_ids] = torch.zeros_like(self.prev_actions[agent][env_ids])

            self.p_desired[agent][env_ids] = self.robots[agent].data.root_pos_w[env_ids].clone()
            self.v_desired[agent][env_ids] = torch.zeros_like(self.v_desired[agent][env_ids])
            self.prev_a_desired[agent][env_ids] = torch.zeros_like(self.prev_a_desired[agent][env_ids])

            if self.cfg.realistic_ctrl:
                self.a_desired_total[agent][env_ids] = torch.zeros_like(self.a_desired_total[agent][env_ids])
                self.thrust_desired[agent][env_ids] = torch.zeros_like(self.thrust_desired[agent][env_ids])
                self.q_desired[agent][env_ids] = torch.zeros_like(self.q_desired[agent][env_ids])
                self.w_desired[agent][env_ids] = torch.zeros_like(self.w_desired[agent][env_ids])
                self.m_desired[agent][env_ids] = torch.zeros_like(self.m_desired[agent][env_ids])

                for t in self.thrust_buffer[agent]:
                    t[env_ids] = torch.zeros_like(t[env_ids])
                for m in self.m_buffer[agent]:
                    m[env_ids] = torch.zeros_like(m[env_ids])

            self.prev_dist_to_goals[agent][env_ids] = torch.linalg.norm(self.goals[agent][env_ids] - self.robots[agent].data.root_pos_w[env_ids], dim=1)

        if self.cfg.realistic_ctrl:
            self.controller.reset(self.env_ids_to_ctrl_ids(env_ids))
            self.control_counter[env_ids] = 0

        # Most (but only most) of the time self.reset_history_buffer is equal to self.reset_buf
        self.reset_history_buffer[env_ids] = True
        self.scroll_counter[env_ids] = 0

        # Update relative positions
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j][env_ids] = self.robots[agent_j].data.root_pos_w[env_ids] - self.robots[agent_i].data.root_pos_w[env_ids]

            self.last_observable_relative_positions[agent_i][env_ids] = torch.zeros_like(self.last_observable_relative_positions[agent_i][env_ids])

    def env_ids_to_ctrl_ids(self, env_ids):
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        drone_offsets = torch.arange(self.cfg.num_drones, dtype=torch.long, device=self.device) * self.num_envs
        ctrl_ids = (drone_offsets[:, None] + env_ids[None, :]).reshape(-1)
        return ctrl_ids

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Reset goal after _get_rewards before _get_observations and _get_states
        # Asynchronous goal resetting in all missions except migration
        # (A mix of synchronous and asynchronous goal resetting may cause state to lose Markovianity :(
        start = time.perf_counter()
        for i, agent in enumerate(self.possible_agents):
            dist_to_goal = torch.linalg.norm(self.goals[agent] - self.robots[agent].data.root_pos_w, dim=1)
            success_i = dist_to_goal < self.success_dist_thr

            if success_i.any():
                self.reset_goal_timer[agent][success_i] += self.step_dt

            reset_goal_idx = (
                (
                    self.reset_goal_timer[agent]
                    > torch.rand(self.num_envs, device=self.device) * (5 * self.cfg.goal_reset_delay - self.cfg.goal_reset_delay) + self.cfg.goal_reset_delay
                )
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )

            if len(reset_goal_idx) > 0:
                mission_0_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 0]  # The migration mission
                mission_1_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 1]  # The crossover mission
                mission_2_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 2]  # The chaotic mission

                if len(mission_0_ids) > 0:
                    rg = self.cfg.flight_range - self.success_dist_thr[mission_0_ids][0]

                    unified_goal_xy = self.unified_goal_xy[mission_0_ids]
                    unified_new_goal_xy = torch.zeros(len(mission_0_ids), 2, device=self.device)
                    done = torch.zeros(len(mission_0_ids), dtype=torch.bool, device=self.device)

                    for attempt in range(self.cfg.max_sampling_tries):
                        active = ~done
                        if not torch.any(active):
                            break

                        active_ids = active.nonzero(as_tuple=False).squeeze(-1)
                        unified_new_goal_xy[active_ids] = torch.zeros_like(unified_new_goal_xy[active_ids]).uniform_(-rg, rg)
                        dist = torch.norm(unified_goal_xy[active_ids] - unified_new_goal_xy[active_ids])

                        ok = dist > 1.414 * rg
                        if torch.any(ok):
                            done[active_ids[ok]] = True

                    if torch.any(~done):
                        failed_ids = mission_0_ids[~done].tolist()
                        logger.warning(
                            f"The search for goal position of the swarm meeting constraints within a side-length {2 * rg} box failed for envs {failed_ids}, using the final sample #_#"
                        )

                    self.unified_goal_xy[mission_0_ids] = unified_new_goal_xy

                    # Synchronous goal resetting in mission migration
                    for i_, agent_ in enumerate(self.possible_agents):
                        self.goals[agent_][mission_0_ids, :2] = self.unified_goal_xy[mission_0_ids].clone()
                        self.goals[agent_][mission_0_ids, 2] = float(self.cfg.flight_altitude)
                        self.goals[agent_][mission_0_ids] += self.terrain.env_origins[mission_0_ids]

                        self.reset_goal_timer[agent_][mission_0_ids] = 0.0

                        # FIXME: Whether ⬇️ should exist?
                        self.prev_dist_to_goals[agent_][mission_0_ids] = torch.linalg.norm(
                            self.goals[agent_][mission_0_ids] - self.robots[agent_].data.root_pos_w[mission_0_ids], dim=1
                        )

                if len(mission_1_ids) > 0:
                    self.ang[mission_1_ids, i] += math.pi
                    self.goals[agent][mission_1_ids, :2] = torch.stack(
                        [torch.cos(self.ang[mission_1_ids, i]), torch.sin(self.ang[mission_1_ids, i])], dim=1
                    ) * self.rand_r[mission_1_ids].unsqueeze(-1)

                    self.goals[agent][mission_1_ids, 2] = float(self.cfg.flight_altitude)
                    self.goals[agent][mission_1_ids] += self.terrain.env_origins[mission_1_ids]

                if len(mission_2_ids) > 0:
                    rg = self.cfg.flight_range - self.success_dist_thr[mission_2_ids][0] - 0.25

                    rand_goal_p = torch.zeros(len(mission_2_ids), self.cfg.num_drones, 2, device=self.device)
                    for i_, agent_ in enumerate(self.possible_agents):
                        rand_goal_p[:, i_] = self.goals[agent_][mission_2_ids, :2].clone()
                    env_origins = self.terrain.env_origins[mission_2_ids]
                    done = torch.zeros(len(mission_2_ids), dtype=torch.bool, device=self.device)

                    for attempt in range(self.cfg.max_sampling_tries):
                        active = ~done
                        if not torch.any(active):
                            break

                        active_ids = active.nonzero(as_tuple=False).squeeze(-1)
                        rand_goal_p[active_ids, i] = (torch.rand(active_ids.numel(), 2, device=self.device) * 2 - 1) * rg + env_origins[active_ids, :2]  # [num_active, 2]
                        dmat = torch.cdist(rand_goal_p[active_ids], rand_goal_p[active_ids])  # [num_active, num_drones, num_drones]
                        eye = torch.eye(self.cfg.num_drones, dtype=torch.bool, device=self.device).expand(active_ids.numel(), -1, -1)

                        dmat.masked_fill_(eye, float("inf"))
                        dmin = dmat.amin(dim=(-2, -1))  # [num_active]

                        ok = dmin > self.cfg.collide_dist
                        if torch.any(ok):
                            done[active_ids[ok]] = True

                    if torch.any(~done):
                        failed_ids = mission_2_ids[~done].tolist()
                        logger.warning(
                            f"The search for goal position of a drone meeting constraints within a side-length {2 * rg} box failed for envs {failed_ids}, using the final sample #_#"
                        )

                    self.goals[agent][mission_2_ids, :2] = rand_goal_p[:, i]

                self.reset_goal_timer[agent][reset_goal_idx] = 0.0

                # FIXME: Whether ⬇️ should exist?
                self.prev_dist_to_goals[agent][reset_goal_idx] = torch.linalg.norm(
                    self.goals[agent][reset_goal_idx] - self.robots[agent].data.root_pos_w[reset_goal_idx], dim=1
                )
        end = time.perf_counter()
        logger.debug(f"Resetting goals takes {end - start:.5f}s")

        start = time.perf_counter()
        sin_max = math.sin(math.radians(self.cfg.max_angle_of_view))
        max_vis = self.cfg.max_visible_distance
        curr_observations = {}
        for i, agent_i in enumerate(self.possible_agents):
            body2goal_w = self.goals[agent_i] - self.robots[agent_i].data.root_pos_w

            idx_others = [j for j in range(len(self.possible_agents)) if j != i]
            rel_positions_w_orig = torch.stack([self.relative_positions_w[i][j] for j in idx_others], dim=1)  # [num_envs, num_drones - 1, 3]

            distances = torch.linalg.norm(rel_positions_w_orig, dim=-1)  # [num_envs, num_drones - 1]
            safe_dist = distances.clamp_min(1e-6)

            # Discard relative observations exceeding maximum visible distance
            mask_far = distances > max_vis  # [num_envs, num_drones - 1]

            # Transform relative positions from world to body frame in a vectorized manner
            inv_quat = quat_inv(self.robots[agent_i].data.root_link_quat_w)  # [num_envs, 4]
            B, N = distances.shape
            rel_flat = rel_positions_w_orig.reshape(B * N, 3)
            inv_quat_rep = inv_quat.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4)
            rel_b_flat = quat_apply(inv_quat_rep, rel_flat)  # [num_envs * (num_drones - 1), 3]
            rel_b = rel_b_flat.view(B, N, 3)  # [num_envs, num_drones - 1, 3]

            # Discard relative observations exceeding maximum elevation field of view
            abs_rel_pos_z_b = rel_b[..., 2].abs()
            mask_invisible = (abs_rel_pos_z_b / safe_dist) > sin_max  # [num_envs, num_drones - 1]

            mask_blocked = mask_far | mask_invisible

            rel_positions_w = rel_positions_w_orig.clone()
            rel_positions_w = rel_positions_w.masked_fill(mask_blocked.unsqueeze(-1), 0.0)
            observability_mask = torch.ones_like(distances, dtype=rel_positions_w.dtype)
            observability_mask = observability_mask.masked_fill(mask_blocked, 0.0)

            # Domain randomization
            if self.cfg.enable_domain_randomization:
                mask_observable = ~mask_blocked
                if mask_observable.any():
                    rel_pos = rel_positions_w_orig[mask_observable]  # [num_observable, 3]
                    dist = distances[mask_observable]  # [num_observable]

                    # Apply a gradually increasing noise to the distance as it grows
                    std_dist = (dist / max_vis) * self.cfg.max_dist_noise_std
                    dist_noisy = (dist + torch.randn_like(dist) * std_dist).clamp_min(0.0)

                    # Similarly apply noise to the bearing in spherical coordinates
                    x, y, z = rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2]
                    az = torch.atan2(y, x)  # Azimuth angle
                    el = torch.atan2(z, torch.sqrt(x * x + y * y))  # Elevation angle
                    std_bearing = (dist / max_vis) * self.cfg.max_bearing_noise_std
                    az_noisy = az + torch.randn_like(az) * std_bearing
                    el_noisy = el + torch.randn_like(el) * std_bearing

                    # Spherical to Cartesian coordinates
                    rel_pos_noisy = torch.stack(
                        [
                            dist_noisy * torch.cos(el_noisy) * torch.cos(az_noisy),
                            dist_noisy * torch.cos(el_noisy) * torch.sin(az_noisy),
                            dist_noisy * torch.sin(el_noisy),
                        ],
                        dim=1,
                    )

                    # Randomly drop relative observations
                    keep_mask = torch.rand_like(dist) > self.cfg.drop_prob  # [num_observable]

                    rel_positions_w[mask_observable] = torch.where(keep_mask.unsqueeze(-1), rel_pos_noisy, torch.zeros_like(rel_pos_noisy))
                    observability_mask[mask_observable] = keep_mask.to(rel_positions_w.dtype)

            rel_with_obs = torch.cat([rel_positions_w, observability_mask.unsqueeze(-1)], dim=-1)  # [num_envs, num_drones - 1, 4]
            self.relative_positions_with_observability[agent_i] = rel_with_obs.reshape(B, N * 4)

            obs = torch.cat(
                [
                    self.actions[agent_i].clone(),
                    body2goal_w[:, :2],
                    self.robots[agent_i].data.root_lin_vel_w[:, :2].clone(),  # TODO: Try to discard velocity observations to reduce sim2real gap
                    self.relative_positions_with_observability[agent_i].clone(),
                ],
                dim=1,
            )
            curr_observations[agent_i] = obs
        end = time.perf_counter()
        logger.debug(f"Generating observations takes {end - start:.5f}s")

        # Scroll or reset (fill in the first frame) the observation buffer
        start = time.perf_counter()
        stacked_observations = {}
        reset_idx = self.reset_history_buffer
        dont_reset_idx = ~self.reset_history_buffer
        for agent in self.cfg.possible_agents:
            buf = self.observation_buffer[agent]

            if reset_idx.any():
                # Reset the buffer by filling in the first observation
                curr_observation = curr_observations[agent].unsqueeze(0)
                buf[:, reset_idx] = curr_observation[:, reset_idx].repeat(self.cfg.history_length, 1, 1)

            if dont_reset_idx.any():
                # Update the final frame of the buffer with the latest observation
                buf[-1, dont_reset_idx] = curr_observations[agent][dont_reset_idx]

            # Record the lateset observable relative positions since the previous scrolling of the buffer
            rel_pos = self.relative_positions_with_observability[agent]
            last_observable_rel_pos = self.last_observable_relative_positions[agent]
            for j in range(self.cfg.num_drones - 1):
                rel_pos_j = rel_pos[:, 4 * j : 4 * (j + 1)]
                last_observable_rel_pos_j = last_observable_rel_pos[:, 4 * j : 4 * (j + 1)]

                # Identify observable relative positions
                mask = rel_pos_j[:, -1] == 1.0
                last_observable_rel_pos_j[mask] = rel_pos_j[mask].clone()

        scroll_buffer_idx = (self.scroll_counter % self.cfg.history_buffer_scroll_decimation == 0) & dont_reset_idx
        self_obs_dim = int(self.cfg.self_observation_dim)
        if scroll_buffer_idx.any():

            for agent in self.cfg.possible_agents:
                buf = self.observation_buffer[agent]

                # Scroll the buffer
                buf[:-1, scroll_buffer_idx] = buf[1:, scroll_buffer_idx].clone()

                # Fill the penultimate frame of the buffer with the lateset observable relative positions since the previous scrolling
                # (Because the latest relative positions might not be observable and rel pos obs is precious
                buf[-2, scroll_buffer_idx, self_obs_dim:] = self.last_observable_relative_positions[agent][scroll_buffer_idx].clone()
                self.last_observable_relative_positions[agent][scroll_buffer_idx] = torch.zeros_like(self.last_observable_relative_positions[agent][scroll_buffer_idx])

            self.scroll_counter[scroll_buffer_idx] = 0
        self.scroll_counter += 1

        for agent in self.cfg.possible_agents:
            buf = self.observation_buffer[agent]
            stacked_observations[agent] = buf.permute(1, 0, 2).reshape(self.num_envs, -1)
        end = time.perf_counter()
        logger.debug(f"Scrolling observation buffer takes {end - start:.5f}s")
        return stacked_observations

    def _get_states(self):
        curr_state = []
        for agent in self.possible_agents:
            curr_state.extend(
                [
                    self.actions[agent].clone(),
                    self.robots[agent].data.root_pos_w - self.terrain.env_origins,
                    self.goals[agent] - self.robots[agent].data.root_pos_w,
                    self.robots[agent].data.root_quat_w.clone(),
                    self.robots[agent].data.root_vel_w.clone(),
                ]
            )
        curr_state = torch.cat(curr_state, dim=1)
        return curr_state

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self.cfg.debug_vis_goal:
                if not hasattr(self, "goal_visualizers"):
                    self.goal_visualizers = {}
                    for i, agent in enumerate(self.possible_agents):
                        marker_cfg = CUBOID_MARKER_CFG.copy()
                        marker_cfg.markers["cuboid"].size = (0.07, 0.07, 0.07)
                        marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                        marker_cfg.prim_path = f"/Visuals/Command/goal_{i}"
                        self.goal_visualizers[agent] = VisualizationMarkers(marker_cfg)
                        self.goal_visualizers[agent].set_visibility(True)

            if self.cfg.debug_vis_collide_dist:
                if not hasattr(self, "collide_dist_visualizers"):
                    self.collide_dist_visualizers = {}
                    for i, agent in enumerate(self.possible_agents):
                        marker_cfg = VisualizationMarkersCfg(
                            prim_path=f"/Visuals/collide_dist_{i}",
                            markers={
                                "cylinder": sim_utils.CylinderCfg(
                                    radius=self.cfg.collide_dist / 2,
                                    height=0.005,
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.01, 0.01), roughness=0.0),
                                )
                            },
                        )
                        self.collide_dist_visualizers[agent] = VisualizationMarkers(marker_cfg)
                        self.collide_dist_visualizers[agent].set_visibility(True)

            if self.cfg.debug_vis_rel_pos:
                if not hasattr(self, "rel_pos_visualizers"):
                    self.num_vis_point = 13
                    self.vis_reset_interval = 3.0
                    self.last_reset_time = 0.0

                    self.selected_vis_agent = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
                    num_neighbors = len(self.possible_agents) - 1

                    self.rel_pos_visualizers = {}
                    for j in range(num_neighbors):
                        self.rel_pos_visualizers[j] = []
                        for p in range(self.num_vis_point):
                            marker_cfg = VisualizationMarkersCfg(
                                prim_path=f"/Visuals/rel_loc_{j}_{p}",
                                markers={"sphere": sim_utils.SphereCfg(radius=0.05, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.01, 0.01, 1.0)))},
                            )
                            self.rel_pos_visualizers[j].append(VisualizationMarkers(marker_cfg))
                            self.rel_pos_visualizers[j][p].set_visibility(True)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_visualizers"):
            for agent in self.possible_agents:
                self.goal_visualizers[agent].visualize(translations=self.goals[agent])

        if hasattr(self, "collide_dist_visualizers"):
            for agent in self.possible_agents:
                t = self.robots[agent].data.root_pos_w.clone()
                t[:, 2] -= 0.077
                self.collide_dist_visualizers[agent].visualize(translations=t)

        if hasattr(self, "rel_pos_visualizers"):
            t = self.common_step_counter * self.step_dt

            if t - self.last_reset_time > self.vis_reset_interval:
                self.last_reset_time = t
                self.selected_vis_agent = torch.randint(0, len(self.possible_agents), (self.num_envs,), device=self.device)

            rel_obs_list = []
            for agent in self.possible_agents:
                # Plot the latest frame of relative positions
                rel_obs = self.relative_positions_with_observability[agent]

                # Plot older relative observations in the history buffer
                # self_obs_dim = int(self.cfg.self_observation_dim)
                # rel_obs = self.observation_buffer[agent][-2, :, self_obs_dim:]

                rel_obs = rel_obs.view(self.num_envs, -1, 4)  # [num_envs, num_drones - 1, 4]
                rel_obs_list.append(rel_obs)
            # Stack → [num_envs, num_drones, num_drones - 1, 4]
            stack_rel_obs = torch.stack(rel_obs_list, dim=1)

            sel_idx = self.selected_vis_agent
            # Select → [num_envs, num_drones - 1, 4]
            sel_rel_obs = stack_rel_obs.gather(dim=1, index=sel_idx.view(self.num_envs, 1, 1, 1).expand(self.num_envs, 1, stack_rel_obs.size(2), 4)).squeeze(1)

            orig_list = [self.robots[a].data.root_pos_w for a in self.possible_agents]
            stack_orig = torch.stack(orig_list, dim=1)
            orig = stack_orig.gather(dim=1, index=sel_idx.view(self.num_envs, 1, 1).expand(self.num_envs, 1, 3)).squeeze(1)

            for j in range(sel_rel_obs.size(1)):
                rel_pos = sel_rel_obs[:, j, :3]
                for p in range(self.num_vis_point):
                    frac = float(p + 1) / (self.num_vis_point + 1)
                    self.rel_pos_visualizers[j][p].visualize(translations=orig + rel_pos * frac)

    def _publish_debug_signals(self):

        t = self._get_ros_timestamp()
        agent = "drone_0"
        env_id = 0

        # Publish states
        state = self.robots[agent].data.root_state_w[env_id]
        p_odom = state[:3].cpu().numpy()
        q_odom = state[3:7].cpu().numpy()
        v_odom = state[7:10].cpu().numpy()
        w_odom_w = state[10:13]
        w_odom_b = quat_apply(quat_inv(self.robots[agent].data.root_quat_w[env_id]), w_odom_w)
        w_odom = w_odom_b.cpu().numpy()

        odom_msg = Odometry()
        odom_msg.header.stamp = t
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = float(p_odom[0])
        odom_msg.pose.pose.position.y = float(p_odom[1])
        odom_msg.pose.pose.position.z = float(p_odom[2])
        odom_msg.pose.pose.orientation.w = float(q_odom[0])
        odom_msg.pose.pose.orientation.x = float(q_odom[1])
        odom_msg.pose.pose.orientation.y = float(q_odom[2])
        odom_msg.pose.pose.orientation.z = float(q_odom[3])
        odom_msg.twist.twist.linear.x = float(v_odom[0])
        odom_msg.twist.twist.linear.y = float(v_odom[1])
        odom_msg.twist.twist.linear.z = float(v_odom[2])
        odom_msg.twist.twist.angular.x = float(w_odom[0])
        odom_msg.twist.twist.angular.y = float(w_odom[1])
        odom_msg.twist.twist.angular.z = float(w_odom[2])
        self.odom_pub.publish(odom_msg)

        # Publish actions
        p_desired = self.p_desired[agent][env_id].cpu().numpy()
        v_desired = self.v_desired[agent][env_id].cpu().numpy()
        if self.cfg.realistic_ctrl:
            a_desired = self.a_desired[agent][env_id].cpu().numpy()
            a_desired_smoothed = self.a_desired_smoothed[agent][env_id].cpu().numpy()

            a_desired_total = self.a_desired_total[agent][env_id].cpu().numpy()
            q_desired = self.q_desired[agent][env_id].cpu().numpy()
            w_desired = self.w_desired[agent][env_id].cpu().numpy()
            m_desired = self.m_desired[agent][env_id].cpu().numpy()

        action_msg = Odometry()
        action_msg.header.stamp = t
        action_msg.header.frame_id = "world"
        action_msg.child_frame_id = "base_link"
        action_msg.pose.pose.position.x = float(p_desired[0])
        action_msg.pose.pose.position.y = float(p_desired[1])
        action_msg.pose.pose.position.z = float(p_desired[2])
        action_msg.twist.twist.linear.x = float(v_desired[0])
        action_msg.twist.twist.linear.y = float(v_desired[1])
        action_msg.twist.twist.linear.z = float(v_desired[2])
        if self.cfg.realistic_ctrl:
            action_msg.pose.pose.orientation.w = float(q_desired[0])
            action_msg.pose.pose.orientation.x = float(q_desired[1])
            action_msg.pose.pose.orientation.y = float(q_desired[2])
            action_msg.pose.pose.orientation.z = float(q_desired[3])
            action_msg.twist.twist.angular.x = float(w_desired[0])
            action_msg.twist.twist.angular.y = float(w_desired[1])
            action_msg.twist.twist.angular.z = float(w_desired[2])
        self.action_pub.publish(action_msg)

        if self.cfg.realistic_ctrl:
            a_desired_msg = AccelStamped()
            a_desired_msg.header.stamp = t
            a_desired_msg.accel.linear.x = float(a_desired[0])
            a_desired_msg.accel.linear.y = float(a_desired[1])
            a_desired_msg.accel.linear.z = float(a_desired[2])
            a_desired_msg.accel.angular.x = float(a_desired_smoothed[0])
            a_desired_msg.accel.angular.y = float(a_desired_smoothed[1])
            a_desired_msg.accel.angular.z = float(a_desired_smoothed[2])
            self.a_desired_pub.publish(a_desired_msg)

            a_desired_total_msg = AccelStamped()
            a_desired_total_msg.header.stamp = t
            a_desired_total_msg.accel.linear.x = float(a_desired_total[0])
            a_desired_total_msg.accel.linear.y = float(a_desired_total[1])
            a_desired_total_msg.accel.linear.z = float(a_desired_total[2])
            self.a_desired_total_pub.publish(a_desired_total_msg)

            m_desired_msg = Vector3Stamped()
            m_desired_msg.header.stamp = t
            m_desired_msg.vector.x = float(m_desired[0])
            m_desired_msg.vector.y = float(m_desired[1])
            m_desired_msg.vector.z = float(m_desired[2])
            self.m_desired_pub.publish(m_desired_msg)

    def _get_ros_timestamp(self) -> Time:
        sim_time = self._sim_step_counter * self.physics_dt

        stamp = Time()
        stamp.sec = int(sim_time)
        stamp.nanosec = int((sim_time - stamp.sec) * 1e9)

        return stamp


from config import agents


gym.register(
    id="FAST-Swarm-Acc",
    entry_point=SwarmAccEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SwarmAccEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:swarm_sb3_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_mappo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.swarm_rsl_rl_ppo_cfg:SwarmAccPPORunnerCfg",
    },
)
