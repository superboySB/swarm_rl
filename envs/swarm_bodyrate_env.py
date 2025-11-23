from __future__ import annotations

import gymnasium as gym
import math
import time
import torch
from collections.abc import Sequence
from loguru import logger

from rclpy.node import Node
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, Vector3Stamped

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass, CircularBuffer, DelayBuffer
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.utils.math import quat_inv, quat_apply

from envs.quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils.utils import quat_to_ang_between_z_body_and_z_world
from utils.controller import bodyrate_control_without_thrust
from utils.custom_trajs import generate_custom_trajs, LissajousConfig


@configclass
class SwarmBodyrateEnvCfg(DirectMARLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 10.0))

    # Reward weights
    to_live_reward_weight = 1.0  # 《活着》
    death_penalty_weight = 0.0
    approaching_goal_reward_weight = 10.0
    success_reward_weight = 10.0
    mutual_collision_penalty_weight = 20.0
    mutual_collision_avoidance_soft_penalty_weight = 0.1
    ang_vel_penalty_weight = 0.05
    action_norm_penalty_weight = 0.05
    action_norm_near_goal_penalty_weight = 10.0
    action_diff_penalty_weight = 0.05
    # Exponential decay factors and tolerances
    mutual_collision_avoidance_reward_scale = 1.0

    # Mission settings
    flight_range = 3.5
    flight_altitude = 1.0  # Desired flight altitude
    safe_dist = 1.0
    collide_dist = 0.5
    goal_reset_time_range = (0.2, 5.0)  # Delay for resetting goal after reaching it
    mission_names = ["migration", "crossover", "chaotic"]
    mission_prob = [0.0, 0.5, 0.5]
    # mission_prob = [1.0, 0.0, 0.0]
    # mission_prob = [0.0, 1.0, 0.0]
    # mission_prob = [0.0, 0.0, 1.0]
    success_distance_threshold = 0.25  # Distance threshold for considering goal reached
    max_sampling_tries = 100  # Maximum number of attempts to sample a valid initial state or goal
    # Params for mission migration
    use_custom_traj = True  # Whether to use custom trajectory for migration mission
    num_custom_trajs = 128
    lissajous_cfg = LissajousConfig()
    # Params for mission crossover
    fix_range = False
    uniformly_distributed_prob = 0.5

    # Observation parameters
    lin_vel_obs_delay_ms = 20.0  # VIO delay: 5ms with imu propogation
    rel_pos_obs_delay_ms = 200.0  # Seeker Omni-4P streaming delay: 160ms + YOLO delay: 40ms
    max_visible_distance = 5.0
    max_angle_of_view = 40.0  # Maximum field of view of camera in tilt direction
    # Domain randomization
    enable_domain_randomization = True
    lin_vel_noise_std = 0.1
    min_dist_noise_std = 0.05
    max_dist_noise_std = 0.5
    min_bearing_noise_std = 0.005
    max_bearing_noise_std = 0.05
    drop_prob = 0.1

    # Parameters for environment and agents
    episode_length_s = 300.0
    physics_freq = 200
    control_freq = 100
    action_freq = 50
    gui_render_freq = 50
    control_decimation = max(1, physics_freq // control_freq)
    num_drones = 5  # Number of drones per environment
    decimation = max(1, math.ceil(physics_freq / action_freq))  # Environment decimation
    render_decimation = max(1, physics_freq // gui_render_freq)
    clip_action = 1.0
    history_length = 3
    self_observation_dim = 17
    relative_observation_dim = 4
    transient_observasion_dim = self_observation_dim + relative_observation_dim * (num_drones - 1)
    observation_spaces = None
    transient_state_dim = 20 * num_drones
    state_space = transient_state_dim
    possible_agents = [f"drone_{i}" for i in range(num_drones)]
    action_spaces = {agent: 4 for agent in possible_agents}
    thrust_to_weight = {agent: 2.0 for agent in possible_agents}
    w_max = {agent: 6.0 for agent in possible_agents}

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=10000, env_spacing=15, replicate_physics=True)

    # Robot
    drone_cfg: ArticulationCfg = DJI_FPV_CFG.copy()
    init_gap = 2.0  # TODO: Redundant feature, to be removed o_0

    # Debug visualization
    debug_vis = True
    debug_vis_goal = True
    debug_vis_collide_dist = True
    debug_vis_rel_pos = False


class SwarmBodyrateEnv(DirectMARLEnv):
    cfg: SwarmBodyrateEnvCfg

    def __init__(self, cfg: SwarmBodyrateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        logger.info(f"Action decimation = {self.cfg.decimation}")
        logger.info(f"Controller decimation = {self.cfg.control_decimation}")

        self.goals = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.prev_dist_to_goals = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.cfg.possible_agents}
        self.env_mission_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_goal_timer = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.cfg.possible_agents}
        self.died = {agent: torch.zeros(self.num_envs, dtype=torch.bool, device=self.device) for agent in self.cfg.possible_agents}
        self.success_dist_thr = torch.zeros(self.num_envs, device=self.device)

        self.mission_prob = torch.tensor(self.cfg.mission_prob, device=self.device)
        # Mission migration params
        self.unified_goal_xy = torch.zeros(self.num_envs, 2, device=self.device)
        if self.cfg.use_custom_traj:
            # Pre-generate multiple custom trajectories for migration mission
            sample_pos = torch.zeros(self.cfg.num_custom_trajs, 3, device=self.device)
            sample_pos[:, 2] = float(self.cfg.flight_altitude)
            sample_vel = torch.zeros(self.cfg.num_custom_trajs, 3, device=self.device)
            sample_acc = torch.zeros(self.cfg.num_custom_trajs, 3, device=self.device)
            trajs = generate_custom_trajs(
                type_id="lissajous",
                p_odom=sample_pos,
                v_odom=sample_vel,
                a_odom=sample_acc,
                p_init=sample_pos,
                custom_cfg=self.cfg.lissajous_cfg,
            )
            self.custom_traj_library = trajs
            self.custom_traj_durations = trajs.get_total_duration()
            self.custom_traj_exec_indexs = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            self.custom_traj_exec_timesteps = torch.zeros(self.num_envs, device=self.device)

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
        self.thrusts_desired = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.w_desired = {}
        self.m_desired = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.possible_agents}

        # Controller
        self.kPw = {agent: torch.tensor([0.05, 0.05, 0.05], device=self.device) for agent in self.cfg.possible_agents}
        self.control_counter = 0

        self.relative_positions_w = {
            i: {j: torch.zeros(self.num_envs, 3, device=self.device) for j in range(self.cfg.num_drones) if j != i} for i in range(self.cfg.num_drones)
        }
        self.rel_pos_w_noisy_with_observability = {}  # For visualization only

        # Delay for observation
        self.lin_vel_obs_max_lag = 0 if self.cfg.lin_vel_obs_delay_ms <= 0.0 else int(math.ceil(self.cfg.lin_vel_obs_delay_ms * 1e-3 / self.step_dt))
        logger.info(f"Max lin vel observation delay = {self.lin_vel_obs_max_lag} env steps")
        self.lin_vel_delay = {
            agent: DelayBuffer(
                history_length=self.lin_vel_obs_max_lag,
                batch_size=self.num_envs,
                device=self.device,
            )
            for agent in self.cfg.possible_agents
        }
        self.rel_pos_max_lag = 0 if self.cfg.rel_pos_obs_delay_ms <= 0.0 else int(math.ceil(self.cfg.rel_pos_obs_delay_ms * 1e-3 / self.step_dt))
        logger.info(f"Max rel pos observation delay = {self.rel_pos_max_lag} env steps")
        self.rel_pos_delay = {
            agent: DelayBuffer(
                history_length=self.rel_pos_max_lag,
                batch_size=self.num_envs,
                device=self.device,
            )
            for agent in self.cfg.possible_agents
        }

        # Sliding window for observation
        self.observation_windows = {
            agent: CircularBuffer(
                max_len=self.cfg.history_length,
                batch_size=self.num_envs,
                device=self.device,
            )
            for agent in self.cfg.possible_agents
        }

        # Logging
        self.episode_sums = {}

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        # ROS2
        self.node = Node("swarm_bodyrate_env", namespace="swarm_bodyrate_env")
        self.odom_pub = self.node.create_publisher(Odometry, "odom", 10)
        self.action_pub = self.node.create_publisher(TwistStamped, "action", 10)
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
        for agent in self.possible_agents:
            # Denormalize and clip the input signal
            self.actions[agent] = actions[agent].clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action
            thrusts_desired_normalized = (self.actions[agent][:, 0] + 1.0) / 2
            w_desired_normalized = self.actions[agent][:, 1:]

            self.thrusts_desired[agent][:, 0, 2] = self.cfg.thrust_to_weight[agent] * self.robot_weights[agent] * thrusts_desired_normalized
            self.w_desired[agent] = self.cfg.w_max[agent] * w_desired_normalized

    def _apply_action(self) -> None:
        if self.control_counter % self.cfg.control_decimation == 0:
            for agent in self.possible_agents:
                self.m_desired[agent][:, 0, :] = bodyrate_control_without_thrust(
                    self.robots[agent].data.root_ang_vel_w, self.w_desired[agent], self.robot_inertias[agent], self.kPw[agent]
                )

            self._publish_debug_signals()

            self.control_counter = 0
        self.control_counter += 1

        for agent in self.possible_agents:
            self.robots[agent].set_external_force_and_torque(self.thrusts_desired[agent], self.m_desired[agent], body_ids=self.body_ids[agent])

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        died_unified = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:

            z_exceed_bounds = torch.logical_or(self.robots[agent].data.root_pos_w[:, 2] < 0.5, self.robots[agent].data.root_pos_w[:, 2] > 1.5)
            ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robots[agent].data.root_quat_w))
            self.died[agent] = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 80.0)

            x_exceed_bounds = torch.logical_or(
                self.robots[agent].data.root_pos_w[:, 0] - self.terrain.env_origins[:, 0] < -self.cfg.flight_range,
                self.robots[agent].data.root_pos_w[:, 0] - self.terrain.env_origins[:, 0] > self.cfg.flight_range,
            )
            y_exceed_bounds = torch.logical_or(
                self.robots[agent].data.root_pos_w[:, 1] - self.terrain.env_origins[:, 1] < -self.cfg.flight_range,
                self.robots[agent].data.root_pos_w[:, 1] - self.terrain.env_origins[:, 1] > self.cfg.flight_range,
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

        return {agent: died_unified for agent in self.possible_agents}, {agent: time_out for agent in self.possible_agents}

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

            # Smoothing
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

            if self.cfg.use_custom_traj:
                # Randomly select a trajectory from the library
                self.custom_traj_exec_indexs[mission_0_ids] = torch.randint(0, self.cfg.num_custom_trajs, (len(mission_0_ids),), device=self.device)
                self.custom_traj_exec_timesteps[mission_0_ids] = torch.zeros(len(mission_0_ids), device=self.device)
                self.unified_goal_xy[mission_0_ids] = torch.zeros(len(mission_0_ids), 2, device=self.device)
            else:
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

            if torch.rand((), device=self.device) < self.cfg.uniformly_distributed_prob:
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
            self.prev_dist_to_goals[agent][env_ids] = torch.linalg.norm(self.goals[agent][env_ids] - self.robots[agent].data.root_pos_w[env_ids], dim=1)

            self.actions[agent][env_ids] = torch.zeros_like(self.actions[agent][env_ids])
            self.prev_actions[agent][env_ids] = torch.zeros_like(self.prev_actions[agent][env_ids])

            self.lin_vel_delay[agent].reset(env_ids)
            self.rel_pos_delay[agent].reset(env_ids)

            if self.lin_vel_obs_max_lag > 0:
                rand_lags = torch.randint(
                    low=math.floor(0.77 * self.lin_vel_obs_max_lag),  # Dončić ~~
                    high=self.lin_vel_obs_max_lag + 1,
                    size=(len(env_ids),),
                    dtype=torch.int,
                    device=self.device,
                )
            else:
                rand_lags = torch.zeros(len(env_ids), dtype=torch.int, device=self.device)
            self.lin_vel_delay[agent].set_time_lag(rand_lags, batch_ids=env_ids)

            if self.rel_pos_max_lag > 0:
                rand_lags = torch.randint(
                    low=math.floor(0.77 * self.rel_pos_max_lag),
                    high=self.rel_pos_max_lag + 1,
                    size=(len(env_ids),),
                    dtype=torch.int,
                    device=self.device,
                )
            else:
                rand_lags = torch.zeros(len(env_ids), dtype=torch.int, device=self.device)
            self.rel_pos_delay[agent].set_time_lag(rand_lags, batch_ids=env_ids)

            self.observation_windows[agent].reset(env_ids)

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
        start = time.perf_counter()

        # Synchronous goal updating along the custom trajectory in the migration mission
        if self.cfg.use_custom_traj:
            custom_traj_envs = (self.env_mission_ids == 0).nonzero(as_tuple=False).squeeze(-1)
            if len(custom_traj_envs) > 0:
                # Step to the next piece in the trajectory for each env
                self.custom_traj_exec_timesteps[custom_traj_envs] += self.step_dt
                traj_indices = self.custom_traj_exec_indexs[custom_traj_envs]
                traj_durations = self.custom_traj_durations[traj_indices]

                # Check if the trajectory is finished
                mask_ = self.custom_traj_exec_timesteps[custom_traj_envs] >= traj_durations
                if mask_.any():
                    completed_envs = custom_traj_envs[mask_]
                    self.custom_traj_exec_indexs[completed_envs] = torch.randint(0, self.cfg.num_custom_trajs, (mask_.sum().item(),), device=self.device)
                    self.custom_traj_exec_timesteps[completed_envs] = 0.0
                    traj_indices = self.custom_traj_exec_indexs[custom_traj_envs]
                    traj_durations = self.custom_traj_durations[traj_indices]

                # Get target position from the trajectory
                traj = self.custom_traj_library[traj_indices]
                current_goals = traj.get_pos(self.custom_traj_exec_timesteps[custom_traj_envs]) + self.terrain.env_origins[custom_traj_envs]
                for agent_name in self.possible_agents:
                    self.goals[agent_name][custom_traj_envs] = current_goals

        for i, agent in enumerate(self.possible_agents):
            dist_to_goal = torch.linalg.norm(self.goals[agent] - self.robots[agent].data.root_pos_w, dim=1)
            success_i = dist_to_goal < self.success_dist_thr

            if success_i.any():
                self.reset_goal_timer[agent][success_i] += self.step_dt

            low, high = self.cfg.goal_reset_time_range
            rand_wait = torch.rand(self.num_envs, device=self.device) * (high - low) + low
            reset_goal_idx = (self.reset_goal_timer[agent] > rand_wait).nonzero(as_tuple=False).squeeze(-1)

            if len(reset_goal_idx) > 0:
                mission_0_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 0]  # The migration mission
                mission_1_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 1]  # The crossover mission
                mission_2_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 2]  # The chaotic mission

                if len(mission_0_ids) > 0 and not self.cfg.use_custom_traj:
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
                        dist = torch.linalg.norm(unified_goal_xy[active_ids] - unified_new_goal_xy[active_ids])

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
                    self.ang[mission_1_ids, i] = (self.ang[mission_1_ids, i] + math.pi) % (2 * math.pi)
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
        stacked_observations = {}
        sin_max = math.sin(math.radians(self.cfg.max_angle_of_view))
        max_vis = self.cfg.max_visible_distance
        for i, agent_i in enumerate(self.possible_agents):
            # Add noise to linear velocity observation
            lin_vel_w = self.robots[agent_i].data.root_lin_vel_w[:, :2]
            lin_vel_w_noisy = lin_vel_w + torch.randn_like(lin_vel_w) * self.cfg.lin_vel_noise_std if self.cfg.enable_domain_randomization else lin_vel_w

            ### ============= Generate noisy relative position observation and observability ============= ###

            idx_others = [j for j in range(len(self.possible_agents)) if j != i]
            rel_pos_w = torch.stack([self.relative_positions_w[i][j] for j in idx_others], dim=1)  # [num_envs, num_drones - 1, 3]

            distances = torch.linalg.norm(rel_pos_w, dim=-1)  # [num_envs, num_drones - 1]
            safe_dist = distances.clamp_min(1e-6)

            # Discard relative observations exceeding maximum visible distance
            mask_far = distances > max_vis  # [num_envs, num_drones - 1]

            # Transform relative positions from world to body frame in a vectorized manner
            inv_quat = quat_inv(self.robots[agent_i].data.root_quat_w)  # [num_envs, 4]
            B, N = distances.shape
            rel_pos_w_flat = rel_pos_w.reshape(B * N, 3)
            rel_pos_b_flat = quat_apply(inv_quat.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4), rel_pos_w_flat)  # [num_envs * (num_drones - 1), 3]
            rel_pos_b = rel_pos_b_flat.view(B, N, 3)  # [num_envs, num_drones - 1, 3]

            # Discard relative observations exceeding maximum elevation field of view
            abs_rel_pos_z_b = rel_pos_b[..., 2].abs()
            mask_invisible = (abs_rel_pos_z_b / safe_dist) > sin_max  # [num_envs, num_drones - 1]

            mask_blocked = mask_far | mask_invisible

            # Domain randomization
            rel_pos_b_noisy = rel_pos_b.clone()
            rel_pos_b_noisy = rel_pos_b_noisy.masked_fill(mask_blocked.unsqueeze(-1), 0.0)
            observability_mask = torch.ones_like(distances, dtype=rel_pos_b_noisy.dtype)
            observability_mask = observability_mask.masked_fill(mask_blocked, 0.0)

            if self.cfg.enable_domain_randomization:
                mask_observable = ~mask_blocked
                if mask_observable.any():
                    rel_pos = rel_pos_b[mask_observable]  # [num_observable, 3]
                    dist = distances[mask_observable]  # [num_observable]

                    dist_normalized = (dist / max_vis).clamp(0.0, 1.0)

                    # Apply a gradually increasing noise to the distance as it grows
                    std_dist = self.cfg.min_dist_noise_std + dist_normalized * (self.cfg.max_dist_noise_std - self.cfg.min_dist_noise_std)
                    dist_noisy = (dist + torch.randn_like(dist) * std_dist).clamp_min(1e-6)

                    # Similarly apply noise to the bearing in spherical coordinates
                    x, y, z = rel_pos[:, 0], rel_pos[:, 1], rel_pos[:, 2]
                    az = torch.atan2(y, x)  # Azimuth angle
                    el = torch.atan2(z, torch.sqrt(x * x + y * y))  # Elevation angle
                    std_bearing = self.cfg.min_bearing_noise_std + dist_normalized * (self.cfg.max_bearing_noise_std - self.cfg.min_bearing_noise_std)
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

                    rel_pos_b_noisy[mask_observable] = torch.where(keep_mask.unsqueeze(-1), rel_pos_noisy, torch.zeros_like(rel_pos_noisy))
                    observability_mask[mask_observable] = keep_mask.to(rel_pos_b_noisy.dtype)

            # Sort neighbors by perceived (noisy) distance and invisible ones go last
            perceived_dist = torch.linalg.norm(rel_pos_b_noisy, dim=-1)
            sort_key = torch.where(observability_mask > 0.5, perceived_dist, torch.full_like(perceived_dist, float("inf")))
            sorted_idx = torch.argsort(sort_key, dim=1)
            rel_pos_b_noisy = rel_pos_b_noisy.gather(1, sorted_idx.unsqueeze(-1).expand(-1, -1, 3))
            observability_mask = observability_mask.gather(1, sorted_idx)

            rel_pos_b_noisy_with_observability = torch.cat([rel_pos_b_noisy, observability_mask.unsqueeze(-1)], dim=-1).reshape(B, N * 4)

            delayed_lin_vel_w_noisy = self.lin_vel_delay[agent_i].compute(lin_vel_w_noisy)
            delayed_rel_pos_b_noisy_with_observability = self.rel_pos_delay[agent_i].compute(rel_pos_b_noisy_with_observability)

            obs = torch.cat(
                [
                    self.actions[agent_i].clone(),
                    self.goals[agent_i] - self.robots[agent_i].data.root_pos_w,
                    self.robots[agent_i].data.root_quat_w.clone(),
                    delayed_lin_vel_w_noisy,  # TODO: Try to discard velocity observations to reduce sim2real gap
                    delayed_rel_pos_b_noisy_with_observability,
                ],
                dim=1,
            )

            self.observation_windows[agent_i].append(obs)
            stacked_observations[agent_i] = self.observation_windows[agent_i].buffer.flatten(1)

            if self.cfg.debug_vis_rel_pos:
                # Convert noisy relative positions back to world frame for visualization
                quat = self.robots[agent_i].data.root_quat_w
                rel_pos_b_noisy_flat = rel_pos_b_noisy.reshape(B * N, 3)
                rel_pos_w_noisy_flat = quat_apply(quat.unsqueeze(1).expand(B, N, 4).reshape(B * N, 4), rel_pos_b_noisy_flat)
                rel_pos_w_noisy = rel_pos_w_noisy_flat.view(B, N, 3)

                self.rel_pos_w_noisy_with_observability[agent_i] = torch.cat([rel_pos_w_noisy, observability_mask.unsqueeze(-1)], dim=-1).reshape(B, N * 4)

        end = time.perf_counter()
        logger.debug(f"Generating observations takes {end - start:.5f}s")

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
                rel_obs = self.rel_pos_w_noisy_with_observability[agent]

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
        p = state[:3].cpu().numpy()
        q = state[3:7].cpu().numpy()
        v = state[7:10].cpu().numpy()
        w = state[10:13].cpu().numpy()

        odom_msg = Odometry()
        odom_msg.header.stamp = t
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "base_link"
        odom_msg.pose.pose.position.x = float(p[0])
        odom_msg.pose.pose.position.y = float(p[1])
        odom_msg.pose.pose.position.z = float(p[2])
        odom_msg.pose.pose.orientation.w = float(q[0])
        odom_msg.pose.pose.orientation.x = float(q[1])
        odom_msg.pose.pose.orientation.y = float(q[2])
        odom_msg.pose.pose.orientation.z = float(q[3])
        odom_msg.twist.twist.linear.x = float(v[0])
        odom_msg.twist.twist.linear.y = float(v[1])
        odom_msg.twist.twist.linear.z = float(v[2])
        odom_msg.twist.twist.angular.x = float(w[0])
        odom_msg.twist.twist.angular.y = float(w[1])
        odom_msg.twist.twist.angular.z = float(w[2])
        self.odom_pub.publish(odom_msg)

        # Publish actions
        thrust_desired = self.thrusts_desired[agent][env_id, 0, 2].cpu().numpy()
        w_desired = self.w_desired[agent][env_id].cpu().numpy()
        m_desired = self.m_desired[agent][env_id, 0, :].cpu().numpy()

        action_msg = TwistStamped()
        action_msg.header.stamp = t
        action_msg.header.frame_id = "world"
        action_msg.twist.linear.x = float(thrust_desired)
        action_msg.twist.angular.x = float(w_desired[0])
        action_msg.twist.angular.y = float(w_desired[1])
        action_msg.twist.angular.z = float(w_desired[2])
        self.action_pub.publish(action_msg)

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
    id="FAST-Swarm-Bodyrate",
    entry_point=SwarmBodyrateEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SwarmBodyrateEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:swarm_sb3_ppo_cfg.yaml",
        "skrl_ppo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_mappo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.swarm_rsl_rl_ppo_cfg:SwarmBodyratePPORunnerCfg",
    },
)
