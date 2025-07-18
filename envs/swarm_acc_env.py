from __future__ import annotations

import gymnasium as gym
import math
import random
import time
import torch
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

from envs.quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils.utils import quat_to_ang_between_z_body_and_z_world
from utils.controller import Controller


@configclass
class SwarmAccEnvCfg(DirectMARLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 60.0))

    # Reward weights
    to_live_reward_weight = 15.0  # 《活着》
    death_penalty_weight = 1.0
    approaching_goal_reward_weight = 5.0
    dist_to_goal_reward_weight = 0.0
    success_reward_weight = 15.0
    time_penalty_weight = 0.0
    mutual_collision_avoidance_reward_weight = 15.0
    max_lin_vel_penalty_weight = 0.0
    ang_vel_penalty_weight = 0.0
    action_diff_penalty_weight = 0.1

    # Exponential decay factors and tolerances
    dist_to_goal_scale = 0.5
    mutual_collision_avoidance_reward_scale = 1.0  # Correspond to safe_dist of 1.3, collide_dist of 0.6
    # mutual_collision_avoidance_reward_scale = 0.5  # Correspond to safe_dist of 3.0, collide_dist of 0.6
    max_lin_vel_penalty_scale = 2.0

    flight_altitude = 1.0  # Desired flight altitude
    safe_dist = 1.3
    collide_dist = 0.6
    goal_reset_delay = 1.0  # Delay for resetting goal after reaching it
    mission_names = ["migration", "crossover", "chaotic"]
    success_distance_threshold = 0.5  # Distance threshold for considering goal reached
    max_sampling_tries = 100  # Maximum number of attempts to sample a valid initial state or goal
    migration_goal_range = 5.0  # Range of xy coordinates of the goal in mission "migration"
    chaotic_goal_range = 4.0  # Range of xy coordinates of the goal in mission "chaotic"
    birth_circle_radius = 2.7

    # TODO: Improve dirty curriculum
    enable_dirty_curriculum = True
    curriculum_steps = 2e5
    init_death_penalty_weight = 1.0
    init_mutual_collision_avoidance_reward_weight = 1.0
    init_action_diff_penalty_weight = 0.01

    # Env
    episode_length_s = 20.0
    physics_freq = 200.0
    control_freq = 100.0
    action_freq = 5.0
    gui_render_freq = 50.0
    control_decimation = physics_freq // control_freq
    num_drones = 5  # Number of drones per environment
    decimation = math.ceil(physics_freq / action_freq)  # Environment decimation
    render_decimation = physics_freq // gui_render_freq
    clip_action = 1.0
    possible_agents = None
    action_spaces = None
    history_length = 1
    # transient_observasion_dim = 6 + 2 * (num_drones - 1)
    transient_observasion_dim = 8
    observation_spaces = None
    transient_state_dim = 16 * num_drones
    state_space = history_length * transient_state_dim

    # Domain randomization
    enable_domain_randomization = False

    # Experience replay
    enable_experience_replay = True
    collision_experience_replay_prob = 0.9
    max_collision_experience_buffer_size = 520
    min_recording_time_before_collision = 0.4
    max_recording_time_before_collision = 1.0

    max_experience_state_buffer_size = int(action_freq * episode_length_s + history_length - 1)
    min_recorded_steps_before_collision = int(action_freq * min_recording_time_before_collision)
    max_recorded_steps_before_collision = int(action_freq * max_recording_time_before_collision)

    def __post_init__(self):
        self.possible_agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.action_spaces = {agent: 2 for agent in self.possible_agents}
        self.observation_spaces = {agent: self.history_length * self.transient_observasion_dim for agent in self.possible_agents}
        self.a_max = {agent: 20.0 for agent in self.possible_agents}
        self.v_max = {agent: 4.0 for agent in self.possible_agents}

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
    init_gap = 2.0

    # Debug visualization
    debug_vis = True
    debug_vis_goal = True
    debug_vis_action = True
    debug_vis_collide_dist = True
    debug_vis_rel_pos = False


class SwarmAccEnv(DirectMARLEnv):
    cfg: SwarmAccEnvCfg

    def __init__(self, cfg: SwarmAccEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Action and control decimation must be greater than or equal to 1 #^#")

        self.goals = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.env_mission_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_goal_timer = {agent: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for agent in self.cfg.possible_agents}
        self.success_dist_thr = torch.zeros(self.num_envs, device=self.device)
        self.xy_boundary = 23.0 * torch.ones(self.num_envs, device=self.device)  # AJ was the limit

        # Mission migration params
        self.r = (2.3 * self.cfg.collide_dist) / 2.0 / math.sin(math.pi / self.cfg.num_drones)
        self.rand_goal_order = list(range(len(self.possible_agents)))
        # Mission crossover params
        self.rand_r = torch.zeros(self.num_envs, device=self.device)
        self.ang = torch.zeros(self.num_envs, self.cfg.num_drones, device=self.device)
        # Mission chaotic params
        self.rand_rg = torch.zeros(self.num_envs, device=self.device)

        # Get specific body indices for each drone
        self.body_ids = {agent: self.robots[agent].find_bodies("body")[0] for agent in self.cfg.possible_agents}

        self.robot_masses = {agent: self.robots[agent].root_physx_view.get_masses()[0, 0].to(self.device) for agent in self.cfg.possible_agents}
        self.robot_inertias = {agent: self.robots[agent].root_physx_view.get_inertias()[0, 0].to(self.device) for agent in self.cfg.possible_agents}
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)
        self.robot_weights = {agent: (self.robot_masses[agent] * self.gravity.norm()).item() for agent in self.cfg.possible_agents}

        # Controller
        self.a_desired_total, self.thrust_desired, self._thrust_desired, self.q_desired, self.w_desired, self.m_desired = {}, {}, {}, {}, {}, {}
        self.controllers = {
            agent: Controller(
                1 / self.cfg.control_freq, self.gravity, self.robot_masses[agent].to(self.device), self.robot_inertias[agent].to(self.device), self.num_envs
            )
            for agent in self.cfg.possible_agents
        }
        self.control_counter = 0
        self.a_xy_desired_normalized, self.prev_a_xy_desired_normalized = {}, {}

        # Denormalized actions
        self.p_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.v_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.a_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.j_desired = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.yaw_desired = {agent: torch.zeros(self.num_envs, 1, device=self.device) for agent in self.cfg.possible_agents}
        self.yaw_dot_desired = {agent: torch.zeros(self.num_envs, 1, device=self.device) for agent in self.cfg.possible_agents}

        self.prev_dist_to_goals = {}

        self.relative_positions_w = {
            i: {j: torch.zeros(self.num_envs, 3, device=self.device) for j in range(self.cfg.num_drones) if j != i} for i in range(self.cfg.num_drones)
        }
        self.relative_positions_with_observability = {}

        self.died = {agent: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for agent in self.cfg.possible_agents}
        self.reset_env_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.observation_buffer = {
            agent: torch.zeros(self.cfg.history_length, self.num_envs, self.cfg.transient_observasion_dim, device=self.device) for agent in self.cfg.possible_agents
        }
        self.state_buffer = torch.zeros(self.cfg.history_length, self.num_envs, self.cfg.transient_state_dim, device=self.device)

        self.experience_state_buffer = []
        self.collision_experience_buffer = []

        # Logging
        self.episode_sums = {}

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        self.visualize_new_cmd = False

        # ROS2
        self.node = Node("swarm_acc_env", namespace="swarm_acc_env")
        self.odom_pub = self.node.create_publisher(Odometry, "odom", 10)
        self.action_pub = self.node.create_publisher(Odometry, "action", 10)
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

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        # TODO: Where would it make more sense to place ⬇️?
        self.reset_env_ids[:] = False

        for agent in self.possible_agents:
            self.a_xy_desired_normalized[agent] = actions[agent].clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action
            a_xy_desired = self.a_xy_desired_normalized[agent] * self.cfg.a_max[agent]
            norm_xy = torch.norm(a_xy_desired, dim=1, keepdim=True)
            clip_scale = torch.clamp(norm_xy / self.cfg.a_max[agent], min=1.0)
            self.a_desired[agent][:, :2] = a_xy_desired / clip_scale

    def _apply_action(self) -> None:
        for agent in self.possible_agents:
            self.p_desired[agent][:, :2] = (
                self.robots[agent].data.root_pos_w[:, :2] + self.v_desired[agent][:, :2] * self.physics_dt + 0.5 * self.a_desired[agent][:, :2] * self.physics_dt**2
            )

            self.v_desired[agent][:, :2] = self.robots[agent].data.root_lin_vel_w[:, :2] + self.a_desired[agent][:, :2] * self.physics_dt
            speed_xy = torch.norm(self.v_desired[agent][:, :2], dim=1, keepdim=True)
            clip_scale = torch.clamp(speed_xy / self.cfg.v_max[agent], min=1.0)
            self.v_desired[agent][:, :2] /= clip_scale

        ### ============= Realistic velocity tracking ============= ###

        # if self.control_counter % self.cfg.control_decimation == 0:
        #     start = time.perf_counter()
        #     for agent in self.possible_agents:
        #         state_desired = torch.cat(
        #             (
        #                 self.p_desired[agent],
        #                 self.v_desired[agent],
        #                 self.a_desired[agent],
        #                 self.j_desired[agent],
        #                 self.yaw_desired[agent],
        #                 self.yaw_dot_desired[agent],
        #             ),
        #             dim=1,
        #         )

        #         (
        #             self.a_desired_total[agent],
        #             self.thrust_desired[agent],
        #             self.q_desired[agent],
        #             self.w_desired[agent],
        #             self.m_desired[agent],
        #         ) = self.controllers[agent].get_control(
        #             self.robots[agent].data.root_state_w,
        #             state_desired,
        #         )

        #         self._thrust_desired[agent] = torch.cat((torch.zeros(self.num_envs, 2, device=self.device), self.thrust_desired[agent].unsqueeze(1)), dim=1)

        #     end = time.perf_counter()
        #     logger.debug(f"get_control for all drones takes {end - start:.5f}s")

        #     self._publish_debug_signals()

        #     self.control_counter = 0
        # self.control_counter += 1

        # for agent in self.possible_agents:
        #     self.robots[agent].set_external_force_and_torque(self._thrust_desired[agent].unsqueeze(1), self.m_desired[agent].unsqueeze(1), body_ids=self.body_ids[agent])

        ### ============= Ideal velocity tracking ============= ###

        for agent in self.possible_agents:
            self._publish_debug_signals()

            v_desired = self.v_desired[agent].clone()
            v_desired[:, 2] += 100.0 * (self.p_desired[agent][:, 2] - self.robots[agent].data.root_pos_w[:, 2])
            # Set angular velocity to zero, treat the rigid body as a particle
            self.robots[agent].write_root_velocity_to_sim(torch.cat((v_desired, torch.zeros_like(v_desired)), dim=1))
    
    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        died_unified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        collision_died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:

            z_exceed_bounds = torch.logical_or(self.robots[agent].data.root_link_pos_w[:, 2] < 0.9, self.robots[agent].data.root_link_pos_w[:, 2] > 1.1)
            ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robots[agent].data.root_link_quat_w))
            self.died[agent] = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 80.0)

            x_exceed_bounds = torch.logical_or(
                self.robots[agent].data.root_link_pos_w[:, 0] - self.terrain.env_origins[:, 0] < -self.xy_boundary,
                self.robots[agent].data.root_link_pos_w[:, 0] - self.terrain.env_origins[:, 0] > self.xy_boundary,
            )
            y_exceed_bounds = torch.logical_or(
                self.robots[agent].data.root_link_pos_w[:, 1] - self.terrain.env_origins[:, 1] < -self.xy_boundary,
                self.robots[agent].data.root_link_pos_w[:, 1] - self.terrain.env_origins[:, 1] > self.xy_boundary,
            )
            self.died[agent] = torch.logical_or(self.died[agent], torch.logical_or(x_exceed_bounds, y_exceed_bounds))

            died_unified = torch.logical_or(died_unified, self.died[agent])

        # Update relative positions, detecting collisions along the way
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j] = self.robots[agent_j].data.root_pos_w - self.robots[agent_i].data.root_pos_w

                collision = torch.linalg.norm(self.relative_positions_w[i][j], dim=1) < self.cfg.collide_dist
                collision_died = torch.logical_or(collision_died, collision)

            #     self.died[agent_i] = torch.logical_or(self.died[agent_i], collision)
            # died_unified = torch.logical_or(died_unified, self.died[agent_i])

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        all_agent_states = []
        for agent in self.possible_agents:
            body2goal_w = self.goals[agent] - self.robots[agent].data.root_pos_w
            curr_state = torch.cat(
                [
                    self.robots[agent].data.root_pos_w - self.terrain.env_origins,
                    body2goal_w,
                    self.robots[agent].data.root_quat_w.clone(),
                    self.robots[agent].data.root_vel_w.clone(),
                ],
                dim=-1,
            )  # [num_envs, 16]
            xy_boundary = self.xy_boundary.clone().unsqueeze(-1)
            range = self.rand_rg.clone().unsqueeze(-1)
            curr_obs = torch.cat(
                [
                    self.a_xy_desired_normalized[agent].clone(),
                    self.robots[agent].data.root_pos_w[:, :2] - self.terrain.env_origins[:, :2],
                    self.goals[agent][:, :2] - self.terrain.env_origins[:, :2],
                    self.robots[agent].data.root_lin_vel_w[:, :2].clone(),
                ],
                dim=-1,
            )  # [num_envs, 8]
            all_agent_states.append(torch.cat([curr_state, xy_boundary, range, curr_obs], dim=-1))  # [num_envs, 26]

        self.experience_state_buffer.append(torch.stack(all_agent_states, dim=1))  # [num_envs, num_agents, 26]
        if len(self.experience_state_buffer) > self.cfg.max_experience_state_buffer_size:
            self.experience_state_buffer.pop(0)

        experience_states = torch.stack(self.experience_state_buffer, dim=0)  # [num_frames, num_envs, num_agents, state_size]
        mask = collision_died  # [num_envs] bool indicating which environments collided
        if mask.any() and len(self.experience_state_buffer) >= self.cfg.max_experience_state_buffer_size:
            min_frames_before_collision = self.cfg.min_recorded_steps_before_collision
            max_frames_before_collision = min(self.cfg.max_recorded_steps_before_collision, self.cfg.max_experience_state_buffer_size - self.cfg.history_length + 1)

            collided_envs = torch.nonzero(mask, as_tuple=True)[0]
            random_frame_idx = torch.randint(
                low=len(self.experience_state_buffer) - max_frames_before_collision,
                high=len(self.experience_state_buffer) - min_frames_before_collision,
                size=(collided_envs.shape[0],),
            )
            recorded_frames_idxs = random_frame_idx[:, None] - torch.arange(self.cfg.history_length - 1, -1, -1)[None, :]  # [num_collided_envs, history_length]
            recorded_frames = experience_states[
                recorded_frames_idxs, collided_envs.unsqueeze(1).expand(-1, self.cfg.history_length), :, :
            ]  # [num_collided_envs, history_length, num_agents, state_size]
            # Select valid frames which are not all zeros
            valid_recorded_frames = recorded_frames[~torch.all(recorded_frames == 0, dim=(1, 2, 3))]
            if valid_recorded_frames.shape[0] > 0:
                self.collision_experience_buffer.extend(valid_recorded_frames)
            while len(self.collision_experience_buffer) > self.cfg.max_collision_experience_buffer_size:
                self.collision_experience_buffer.pop(0)

        experience_states[:, torch.nonzero(died_unified, as_tuple=True)[0]] = 0.0
        self.experience_state_buffer = list(torch.unbind(experience_states, dim=0))

        return {agent: died_unified for agent in self.cfg.possible_agents}, {agent: time_out for agent in self.cfg.possible_agents}

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # TODO: Improve dirty curriculum
        if self.cfg.enable_dirty_curriculum:

            if not hasattr(self, "delta_death_penalty_weight"):
                self.final_death_penalty_weight = self.cfg.death_penalty_weight
                self.cfg.death_penalty_weight = self.cfg.init_death_penalty_weight
                self.delta_death_penalty_weight = (self.final_death_penalty_weight - self.cfg.init_death_penalty_weight) / self.cfg.curriculum_steps
            else:
                self.cfg.death_penalty_weight += self.delta_death_penalty_weight

            if not hasattr(self, "delta_mutual_collision_avoidance_reward_weight"):
                self.final_mutual_collision_avoidance_reward_weight = self.cfg.mutual_collision_avoidance_reward_weight
                self.cfg.mutual_collision_avoidance_reward_weight = self.cfg.init_mutual_collision_avoidance_reward_weight
                self.delta_mutual_collision_avoidance_reward_weight = (
                    self.final_mutual_collision_avoidance_reward_weight - self.cfg.init_mutual_collision_avoidance_reward_weight
                ) / self.cfg.curriculum_steps
            else:
                self.cfg.mutual_collision_avoidance_reward_weight += self.delta_mutual_collision_avoidance_reward_weight

            if not hasattr(self, "delta_action_diff_penalty_weight"):
                self.final_action_diff_penalty_weight = self.cfg.action_diff_penalty_weight
                self.cfg.action_diff_penalty_weight = self.cfg.init_action_diff_penalty_weight
                self.delta_action_diff_penalty_weight = (self.final_action_diff_penalty_weight - self.cfg.init_action_diff_penalty_weight) / self.cfg.curriculum_steps
            else:
                self.cfg.action_diff_penalty_weight += self.delta_action_diff_penalty_weight

        rewards = {}

        mutual_collision_avoidance_reward = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.possible_agents}
        for i, agent in enumerate(self.possible_agents):
            for j, _ in enumerate(self.possible_agents):
                if i == j:
                    continue

                dist_btw_drones = torch.linalg.norm(self.relative_positions_w[i][j], dim=1)

                # collision_penalty = 1.0 / (1.0 + torch.exp(52.0 * (dist_btw_drones - self.cfg.safe_dist)))
                collision_penalty = torch.where(
                    dist_btw_drones < self.cfg.safe_dist,
                    torch.exp(self.cfg.mutual_collision_avoidance_reward_scale * (self.cfg.safe_dist - dist_btw_drones)) - 1.0,
                    torch.zeros(self.num_envs, device=self.device),
                )
                mutual_collision_avoidance_reward[agent] -= collision_penalty

        for agent in self.possible_agents:
            dist_to_goal = torch.linalg.norm(self.goals[agent] - self.robots[agent].data.root_pos_w, dim=1)
            approaching_goal_reward = self.prev_dist_to_goals[agent] - dist_to_goal
            self.prev_dist_to_goals[agent] = dist_to_goal

            dist_to_goal_reward = torch.exp(-self.cfg.dist_to_goal_scale * dist_to_goal)

            success_i = dist_to_goal < self.success_dist_thr
            # Additional reward when the drone is close to goal
            success_reward = torch.where(success_i, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))
            # Time penalty for not reaching the goal
            time_reward = -torch.where(~success_i, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))

            death_reward = -torch.where(self.died[agent], torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))

            ### ============= Smoothing ============= ###
            ang_vel_reward = -torch.linalg.norm(self.robots[agent].data.root_ang_vel_w, dim=1)

            lin_vel = torch.linalg.norm(self.robots[agent].data.root_lin_vel_w, dim=1)
            # max_lin_vel_penalty = 1.0 / (1.0 + torch.exp(52.0 * (self.cfg.v_max[agent] - lin_vel)))
            max_lin_vel_penalty = torch.where(
                lin_vel > self.cfg.v_max[agent],
                torch.exp(self.cfg.max_lin_vel_penalty_scale * (lin_vel - self.cfg.v_max[agent])) - 1.0,
                torch.zeros(self.num_envs, device=self.device),
            )
            max_lin_vel_reward = -max_lin_vel_penalty

            action_diff_reward = -torch.linalg.norm(self.a_xy_desired_normalized[agent] - self.prev_a_xy_desired_normalized[agent], dim=1)

            reward = {
                "meaning_to_live": torch.ones(self.num_envs, device=self.device) * self.cfg.to_live_reward_weight * self.step_dt,
                "approaching_goal": approaching_goal_reward * self.cfg.approaching_goal_reward_weight * self.step_dt,
                "dist_to_goal": dist_to_goal_reward * self.cfg.dist_to_goal_reward_weight * self.step_dt,
                "success": success_reward * self.cfg.success_reward_weight * self.step_dt,
                "death_penalty": death_reward * self.cfg.death_penalty_weight,
                "time_penalty": time_reward * self.cfg.time_penalty_weight * self.step_dt,
                "mutual_collision_avoidance": mutual_collision_avoidance_reward[agent] * self.cfg.mutual_collision_avoidance_reward_weight * self.step_dt,
                ### ============= Smoothing ============= ###
                "max_lin_vel_penalty": max_lin_vel_reward * self.cfg.max_lin_vel_penalty_weight * self.step_dt,
                "ang_vel_penalty": ang_vel_reward * self.cfg.ang_vel_penalty_weight * self.step_dt,
                "action_diff_penalty": action_diff_reward * self.cfg.action_diff_penalty_weight * self.step_dt,
            }

            # Logging
            for key, value in reward.items():
                if key in self.episode_sums:
                    self.episode_sums[key] += value / self.num_agents
                else:
                    self.episode_sums[key] = value / self.num_agents

            reward = torch.sum(torch.stack(list(reward.values())), dim=0)

            rewards[agent] = reward
        return rewards

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["drone_0"]._ALL_INDICES

        # Most (but only most) of the time self.reset_env_ids is equal to self.reset_buf
        self.reset_env_ids[env_ids] = True

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
        self.env_mission_ids[env_ids] = torch.randint(0, len(self.cfg.mission_names), (len(env_ids),), device=self.device)
        self.env_mission_ids[env_ids] = 2
        mission_0_ids = env_ids[self.env_mission_ids[env_ids] == 0]  # The migration mission
        mission_1_ids = env_ids[self.env_mission_ids[env_ids] == 1]  # The crossover mission
        mission_2_ids = env_ids[self.env_mission_ids[env_ids] == 2]  # The chaotic mission

        self.success_dist_thr[mission_0_ids] = self.cfg.success_distance_threshold * self.cfg.num_drones / 2
        self.success_dist_thr[mission_1_ids] = self.cfg.success_distance_threshold
        self.success_dist_thr[mission_2_ids] = self.cfg.success_distance_threshold

        ### ============= Reset robot state and specify goal ============= ###
        # The migration mission: randomly permute initial root among agents
        if len(mission_0_ids) > 0:
            self.xy_boundary[mission_0_ids] = self.cfg.migration_goal_range + self.r + 1.314

            random.shuffle(self.rand_goal_order)
            rand_init_order = self.rand_goal_order.clone()
            random.shuffle(rand_init_order)

            unified_goal_xy = torch.zeros_like(self.goals["drone_0"][mission_0_ids, :2]).uniform_(-self.cfg.migration_goal_range, self.cfg.migration_goal_range)
            unified_init_xy = torch.zeros_like(unified_goal_xy).uniform_(-self.cfg.migration_goal_range, self.cfg.migration_goal_range)

        if len(mission_1_ids) > 0:
            r_min, r_max = self.cfg.birth_circle_radius, 2 * self.cfg.birth_circle_radius
            self.rand_r[mission_1_ids] = torch.rand(len(mission_1_ids), device=self.device) * (r_max - r_min) + r_min

            self.xy_boundary[mission_1_ids] = self.rand_r[mission_1_ids] + 1.314

            for idx in mission_1_ids.tolist():
                r = self.rand_r[idx]
                for attempt in range(self.cfg.max_sampling_tries):
                    rand_ang = torch.rand(self.cfg.num_drones, device=self.device) * 2 * math.pi
                    pts = torch.stack([torch.cos(rand_ang) * r, torch.sin(rand_ang) * r], dim=1)
                    dmat = torch.cdist(pts, pts)
                    dmat.fill_diagonal_(float("inf"))
                    last_rand_ang = rand_ang
                    if torch.min(dmat) >= 2.3 * self.cfg.collide_dist:
                        self.ang[idx] = rand_ang
                        break
                else:
                    logger.warning(f"The search for initial positions of the swarm meeting constraints on a radius {r} circle failed, using the final sample #_#")
                    self.ang[idx] = last_rand_ang

        if len(mission_2_ids) > 0:
            rg_min, rg_max = self.cfg.chaotic_goal_range, 1.01 * self.cfg.chaotic_goal_range
            self.rand_rg[mission_2_ids] = torch.rand(len(mission_2_ids), device=self.device) * (rg_max - rg_min) + rg_min
            init_p = torch.zeros(self.num_envs, self.cfg.num_drones, 2, device=self.device)
            goal_p = torch.zeros(self.num_envs, self.cfg.num_drones, 2, device=self.device)

            self.xy_boundary[mission_2_ids] = self.rand_rg[mission_2_ids] + 1.314

            for idx in mission_2_ids.tolist():
                rg = self.rand_rg[idx]
                for attempt in range(self.cfg.max_sampling_tries):
                    rand_pts = (torch.rand(self.cfg.num_drones, 2, device=self.device) * 2 - 1) * rg
                    dmat = torch.cdist(rand_pts, rand_pts)
                    dmat.fill_diagonal_(float("inf"))
                    last_rand_pts = rand_pts
                    if torch.min(dmat) >= 2.3 * self.cfg.collide_dist:
                        init_p[idx] = rand_pts
                        break
                else:
                    logger.warning(f"The search for initial positions of the swarm meeting constraints within a side-length {rg} box failed, using the final sample #_#")
                    init_p[idx] = last_rand_pts

                for attempt in range(self.cfg.max_sampling_tries):
                    rand_pts = (torch.rand(self.cfg.num_drones, 2, device=self.device) * 2 - 1) * rg
                    dmat = torch.cdist(rand_pts, rand_pts)
                    dmat.fill_diagonal_(float("inf"))
                    last_rand_pts = rand_pts
                    if torch.min(dmat) >= 2.3 * self.cfg.collide_dist:
                        goal_p[idx] = rand_pts
                        break
                else:
                    logger.warning(f"The search for goal positions of the swarm meeting constraints within a side-length {rg} box failed, using the final sample #_#")
                    goal_p[idx] = last_rand_pts

        self.experience_replay_states = None
        self.experience_replayed = (
            self.cfg.enable_experience_replay and random.random() < self.cfg.collision_experience_replay_prob and len(self.collision_experience_buffer) > 0
        )
        # Experience replay: set initial state and goal based on a randomly chosen state from the failure buffer, and update the observation and state buffers
        if self.experience_replayed:
            self.experience_replay_states = torch.stack(
                [random.choice(self.collision_experience_buffer) for _ in env_ids]
            )  # [num_envs_to_reset, history_length, num_agents, state_size]
            self.experience_replay_ids = env_ids

        for i, agent in enumerate(self.possible_agents):
            init_state = self.robots[agent].data.default_root_state.clone()

            # The migration mission: default init states + unified random target
            if len(mission_0_ids) > 0:
                ang = self.rand_init_order[i] * 2 * math.pi / self.cfg.num_drones
                init_state[mission_0_ids, :2] = unified_init_xy + torch.tensor([math.cos(ang), math.sin(ang)], device=self.device) * self.r

                ang = self.rand_goal_order[i] * 2 * math.pi / self.cfg.num_drones
                self.goals[agent][mission_0_ids, :2] = unified_goal_xy + torch.tensor([math.cos(ang), math.sin(ang)], device=self.device) * self.r

            # The crossover mission: init states uniformly distributed on a circle + target on the opposite side
            if len(mission_1_ids) > 0:
                ang = self.ang[mission_1_ids, i]
                r = self.rand_r[mission_1_ids].unsqueeze(1)

                init_state[mission_1_ids, :2] = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1) * r

                ang += math.pi  # Terminate angles
                self.goals[agent][mission_1_ids, :2] = torch.stack([torch.cos(ang), torch.sin(ang)], dim=1) * r

            # The chaotic mission: random init states + respective random target
            if len(mission_2_ids) > 0:
                init_state[mission_2_ids, :2] = init_p[mission_2_ids, i]

                self.goals[agent][mission_2_ids, :2] = goal_p[mission_2_ids, i]

            init_state[env_ids, 2] = float(self.cfg.flight_altitude)
            init_state[env_ids, :3] += self.terrain.env_origins[env_ids]

            self.goals[agent][env_ids, 2] = float(self.cfg.flight_altitude)
            self.goals[agent][env_ids] += self.terrain.env_origins[env_ids]

            if self.experience_replay_states is not None:
                # Select the last frame of the experience replay state to initialize the robot state
                states_ = self.experience_replay_states[:, -1, i, :18].clone()  # [num_envs_to_reset, 18]
                root_pos_w_ = states_[:, 0:3] + self.terrain.env_origins[env_ids]
                init_state[env_ids, 0:3] = root_pos_w_
                init_state[env_ids, 3:13] = states_[:, 6:16]  # Quats, lin_vels, ang_vels
                self.goals[agent][env_ids] = root_pos_w_ + states_[:, 3:6]  # Goals
                self.xy_boundary[env_ids] = states_[:, 16]
                self.rand_rg[env_ids] = states_[:, 17]

            self.robots[agent].write_root_pose_to_sim(init_state[env_ids, :7], env_ids)
            self.robots[agent].write_root_velocity_to_sim(init_state[env_ids, 7:], env_ids)
            self.robots[agent].write_joint_state_to_sim(
                self.robots[agent].data.default_joint_pos[env_ids], self.robots[agent].data.default_joint_vel[env_ids], None, env_ids
            )

            self.controllers[agent].reset(env_ids)

            self.p_desired[agent][env_ids] = self.robots[agent].data.root_pos_w[env_ids].clone()
            self.v_desired[agent][env_ids] = torch.zeros_like(self.robots[agent].data.root_lin_vel_w[env_ids])

            if agent in self.prev_dist_to_goals:
                self.prev_dist_to_goals[agent][env_ids] = torch.linalg.norm(self.goals[agent][env_ids] - self.robots[agent].data.root_pos_w[env_ids], dim=1)
            else:
                self.prev_dist_to_goals[agent] = torch.linalg.norm(self.goals[agent] - self.robots[agent].data.root_pos_w, dim=1)

            if agent in self.prev_a_xy_desired_normalized:
                self.prev_a_xy_desired_normalized[agent][env_ids] = torch.zeros_like(self.a_xy_desired_normalized[agent][env_ids])
            else:
                self.a_xy_desired_normalized[agent] = torch.zeros(self.num_envs, 2, device=self.device)
                self.prev_a_xy_desired_normalized[agent] = self.a_xy_desired_normalized[agent].clone()

            self.reset_goal_timer[agent][env_ids] = 0.0

        # Update relative positions
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j][env_ids] = self.robots[agent_j].data.root_pos_w[env_ids] - self.robots[agent_i].data.root_pos_w[env_ids]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Reset goal after _get_rewards before _get_observations and _get_states
        # Asynchronous goal resetting in all missions except migration
        # (A mix of synchronous and asynchronous goal resetting may cause state to lose Markovianity :(
        for i, agent in enumerate(self.possible_agents):
            dist_to_goal = torch.linalg.norm(self.goals[agent] - self.robots[agent].data.root_pos_w, dim=1)
            success_i = dist_to_goal < self.success_dist_thr

            if success_i.any():
                self.reset_goal_timer[agent][success_i] += self.step_dt

            reset_goal_idx = (self.reset_goal_timer[agent] > self.cfg.goal_reset_delay).nonzero(as_tuple=False).squeeze(-1)
            if len(reset_goal_idx) > 0:
                mission_0_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 0]  # The migration mission
                mission_1_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 1]  # The crossover mission
                mission_2_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 2]  # The chaotic mission

                if len(mission_0_ids) > 0:
                    random.shuffle(self.rand_goal_order)
                    unified_goal_xy = torch.zeros_like(self.goals["drone_0"][mission_0_ids, :2]).uniform_(-self.cfg.migration_goal_range, self.cfg.migration_goal_range)

                    # Synchronous goal resetting in mission migration
                    for i_, agent_ in enumerate(self.possible_agents):
                        ang = self.rand_goal_order[i_] * 2 * math.pi / self.cfg.num_drones
                        self.goals[agent_][mission_0_ids, :2] = unified_goal_xy + torch.tensor([math.cos(ang), math.sin(ang)], device=self.device) * self.r

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
                    ) * self.rand_r[mission_1_ids].unsqueeze(1)

                    self.goals[agent][mission_1_ids, 2] = float(self.cfg.flight_altitude)
                    self.goals[agent][mission_1_ids] += self.terrain.env_origins[mission_1_ids]

                if len(mission_2_ids) > 0:
                    goal_p = torch.zeros(self.num_envs, self.cfg.num_drones, 2, device=self.device)
                    for i_, agent_ in enumerate(self.possible_agents):
                        goal_p[mission_2_ids, i_] = self.goals[agent_][mission_2_ids, :2].clone()

                    for idx in mission_2_ids.tolist():
                        rg = self.rand_rg[idx]

                        for attempt in range(self.cfg.max_sampling_tries):
                            goal_p[idx, i] = (torch.rand(2, device=self.device) * 2 - 1) * rg + self.terrain.env_origins[idx, :2]
                            dmat = torch.cdist(goal_p[idx], goal_p[idx])
                            dmat.fill_diagonal_(float("inf"))
                            if torch.min(dmat) >= 2.3 * self.cfg.collide_dist:
                                break
                        else:
                            logger.warning(
                                f"The search for goal positions of the swarm meeting constraints within a side-length {rg} box failed, using the final sample #_#"
                            )

                    self.goals[agent][mission_2_ids, :2] = goal_p[mission_2_ids, i]

                self.reset_goal_timer[agent][reset_goal_idx] = 0.0

                # FIXME: Whether ⬇️ should exist?
                self.prev_dist_to_goals[agent][reset_goal_idx] = torch.linalg.norm(
                    self.goals[agent][reset_goal_idx] - self.robots[agent].data.root_pos_w[reset_goal_idx], dim=1
                )

        curr_observations = {}
        for i, agent_i in enumerate(self.possible_agents):
            body2goal_w = self.goals[agent_i] - self.robots[agent_i].data.root_pos_w

            # TODO: To be removed from observation
            body2goal_others_w = []
            goal_others_w = []

            relative_positions_with_observability = []
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue

                body2goal_others_w.append(self.goals[agent_j][:, :2] - self.robots[agent_i].data.root_pos_w[:, :2])
                goal_others_w.append(self.goals[agent_j][:, :2] - self.terrain.env_origins[:, :2])

                relative_positions_w = self.relative_positions_w[i][j].clone()
                distances = torch.linalg.norm(relative_positions_w, dim=1)
                observability_mask = torch.ones_like(distances)

                # Domain randomization
                if self.cfg.enable_domain_randomization:
                    std = 0.01 + 1.0 * distances / 10.0
                    relative_positions_w += torch.randn_like(relative_positions_w) * std

                    # Discard remote (> 10.0m) observations
                    relative_positions_w[distances > 10.0] = 0.0
                    observability_mask[distances > 10.0] = 0.0

                    # Discard medium-range (5.0m < distance <= 10.0m) observations with probability proportional to distance
                    mid_range = (distances > 5.0) & (distances <= 10.0)
                    if mid_range.any():
                        prob = (distances[mid_range] - 5.0) / 5.0
                        rand = torch.rand_like(prob)
                        discard = rand < prob
                        relative_positions_w[mid_range][discard] = 0.0  # Set discarded observations to zero
                        observability_mask[mid_range] = (~discard).float()

                # relative_positions_with_observability.append(torch.cat([relative_positions_w, observability_mask.unsqueeze(-1)], dim=-1))
                relative_positions_with_observability.append(self.relative_positions_w[i][j][:, :2].clone())
            self.relative_positions_with_observability[agent_i] = torch.cat(relative_positions_with_observability, dim=-1)

            body2goal_others_w = torch.cat(body2goal_others_w, dim=-1)
            goal_others_w = torch.cat(goal_others_w, dim=-1)

            obs = torch.cat(
                [
                    self.a_xy_desired_normalized[agent_i].clone(),
                    self.robots[agent_i].data.root_pos_w[:, :2] - self.terrain.env_origins[:, :2],
                    self.goals[agent_i][:, :2] - self.terrain.env_origins[:, :2],
                    # body2goal_w[:, :2].clone(),
                    # self.robots[agent_i].data.root_quat_w.clone(),
                    # self.robots[agent_i].data.root_vel_w.clone(),  # TODO: Try to discard velocity observations to reduce sim2real gap
                    self.robots[agent_i].data.root_lin_vel_w[:, :2].clone(),  # TODO: Try to discard velocity observations to reduce sim2real gap
                    # goal_others_w,
                    # body2goal_others_w,
                    # self.relative_positions_with_observability[agent_i].clone(),
                ],
                dim=-1,
            )
            curr_observations[agent_i] = obs

            # TODO: Where would it make more sense to place ⬇️?
            non_reset_env_ids = ~self.reset_env_ids
            if non_reset_env_ids.any():
                self.prev_a_xy_desired_normalized[agent_i][non_reset_env_ids] = self.a_xy_desired_normalized[agent_i][non_reset_env_ids].clone()

        # Scroll or reset (fill in the first frame) the observation buffer
        for i, agent in enumerate(self.cfg.possible_agents):
            buf = self.observation_buffer[agent]
            if self.reset_env_ids.any():
                if self.experience_replayed and self.experience_replay_states is not None:
                    # Use the experience replay states to initialize the observation buffer
                    replay_obs_ = self.experience_replay_states[:, :, i, 18:].permute(1, 0, 2).contiguous()  # [history_length, num_envs_to_reset, obs_dim]

                    # Fill the all zero frames with the last non-zero frame
                    mask_zero_ = replay_obs_.abs().sum(-1) == 0  # [history_length, num_envs_to_reset]
                    if mask_zero_.any():
                        first_non_zero_ = (~mask_zero_).int().argmax(dim=0)  # [num_envs_to_reset]
                        last_zero = first_non_zero_ - 1
                        t_ = torch.arange(replay_obs_.size(0), device=replay_obs_.device).unsqueeze(1)
                        replay_obs_ = torch.where(
                            (t_ <= last_zero.unsqueeze(0)).unsqueeze(2),
                            replay_obs_[first_non_zero_, torch.arange(replay_obs_.size(1), device=replay_obs_.device)].unsqueeze(0),
                            replay_obs_,
                        )

                    buf[:, self.reset_env_ids] = replay_obs_
                else:
                    curr_observation = curr_observations[agent].unsqueeze(0)
                    buf[:, self.reset_env_ids] = curr_observation[:, self.reset_env_ids].repeat(self.cfg.history_length, 1, 1)

            scroll_buffer = ~self.reset_env_ids
            if scroll_buffer.any():
                buf[:-1, scroll_buffer] = buf[1:, scroll_buffer].clone()
                buf[-1, scroll_buffer] = curr_observations[agent][scroll_buffer]

        stacked_observations = {}
        for agent in self.cfg.possible_agents:
            buf = self.observation_buffer[agent]
            stacked_observations[agent] = buf.permute(1, 0, 2).reshape(self.num_envs, -1)
        return stacked_observations

    def _get_states(self):
        curr_state = []
        for agent in self.possible_agents:
            body2goal_w = self.goals[agent] - self.robots[agent].data.root_pos_w
            curr_state.extend(
                [
                    self.robots[agent].data.root_pos_w - self.terrain.env_origins,
                    body2goal_w,
                    self.robots[agent].data.root_quat_w.clone(),
                    self.robots[agent].data.root_vel_w.clone(),
                ]
            )
        curr_state = torch.cat(curr_state, dim=-1)

        # Scroll or reset (fill in the first frame) the state buffer
        buf = self.state_buffer
        if self.reset_env_ids.any():
            if self.experience_replayed and self.experience_replay_states is not None:
                # Use the experience replay states to initialize the state buffer
                replay_states_ = (
                    self.experience_replay_states[..., :16].permute(1, 0, 2, 3).contiguous().flatten(start_dim=2)
                )  # [history_length, num_envs_to_reset, state_size * num_agents]

                # Fill the all zero frames with the last non-zero frame
                mask_zero_ = replay_states_.abs().sum(-1) == 0  # [history_length, num_envs_to_reset]
                if mask_zero_.any():
                    first_non_zero_ = (~mask_zero_).int().argmax(dim=0)  # [num_envs_to_reset]
                    last_zero = first_non_zero_ - 1
                    t_ = torch.arange(replay_states_.size(0), device=replay_states_.device).unsqueeze(1)
                    replay_states_ = torch.where(
                        (t_ <= last_zero.unsqueeze(0)).unsqueeze(2),
                        replay_states_[first_non_zero_, torch.arange(replay_states_.size(1), device=replay_states_.device)].unsqueeze(0),
                        replay_states_,
                    )

                buf[:, self.reset_env_ids, :] = replay_states_
            else:
                curr_state_ = curr_state.unsqueeze(0)
                buf[:, self.reset_env_ids] = curr_state_[:, self.reset_env_ids].repeat(self.cfg.history_length, 1, 1)

        scroll_buffer = ~self.reset_env_ids
        if scroll_buffer.any():
            buf[:-1, scroll_buffer] = buf[1:, scroll_buffer].clone()
            buf[-1, scroll_buffer] = curr_state[scroll_buffer]

        stacked_state = buf.permute(1, 0, 2).reshape(self.num_envs, -1)
        return stacked_state

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
                    self.num_vis_point = 23
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
                rel_obs = self.relative_positions_with_observability[agent].view(self.num_envs, -1, 4)  # [num_envs, num_drones - 1, 4]
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
                mask = sel_rel_obs[:, j, 3].bool()
                if mask.any():
                    for p in range(self.num_vis_point):
                        frac = float(p + 1) / (self.num_vis_point + 1)
                        self.rel_pos_visualizers[j][p].visualize(translations=orig[mask] + rel_pos[mask] * frac)

    def _publish_debug_signals(self):

        t = self._get_ros_timestamp()
        agent = "drone_0"
        env_id = 0

        # Publish states
        state = self.robots[agent].data.root_state_w[env_id]
        p_odom = state[:3].cpu().numpy()
        q_odom = state[3:7].cpu().numpy()
        v_odom = state[7:10].cpu().numpy()
        w_odom = state[10:13].cpu().numpy()

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
        a_desired = self.a_desired[agent][env_id].cpu().numpy()

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
        action_msg.pose.pose.orientation.w = float(q_desired[0])
        action_msg.pose.pose.orientation.x = float(q_desired[1])
        action_msg.pose.pose.orientation.y = float(q_desired[2])
        action_msg.pose.pose.orientation.z = float(q_desired[3])
        action_msg.twist.twist.linear.x = float(v_desired[0])
        action_msg.twist.twist.linear.y = float(v_desired[1])
        action_msg.twist.twist.linear.z = float(v_desired[2])
        action_msg.twist.twist.angular.x = float(w_desired[0])
        action_msg.twist.twist.angular.y = float(w_desired[1])
        action_msg.twist.twist.angular.z = float(w_desired[2])
        self.action_pub.publish(action_msg)

        a_desired_msg = AccelStamped()
        a_desired_msg.header.stamp = t
        a_desired_msg.accel.linear.x = float(a_desired[0])
        a_desired_msg.accel.linear.y = float(a_desired[1])
        a_desired_msg.accel.linear.z = float(a_desired[2])
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
