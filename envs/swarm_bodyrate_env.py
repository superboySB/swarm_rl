from __future__ import annotations

import gymnasium as gym
import math
import random
import torch
from collections.abc import Sequence
from loguru import logger

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
from utils.controller import bodyrate_control_without_thrust


@configclass
class SwarmBodyrateEnvCfg(DirectMARLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 40.0))

    # Reward weights
    death_penalty_weight = 1.0
    approaching_goal_reward_weight = 1.0
    dist_to_goal_reward_weight = 0.0
    success_reward_weight = 10.0
    time_penalty_weight = 0.0
    altitude_maintenance_reward_weight = 0.0  # Reward for maintaining height close to flight_altitude
    speed_maintenance_reward_weight = 0.0  # Reward for maintaining speed close to v_desired
    mutual_collision_avoidance_reward_weight = 1.0
    lin_vel_penalty_weight = 0.001
    ang_vel_penalty_weight = 0.0001
    ang_vel_diff_penalty_weight = 0.0001
    thrust_diff_penalty_weight = 0.0001

    # Exponential decay factors and tolerances
    dist_to_goal_scale = 0.5
    speed_deviation_tolerance = 0.5

    flight_altitude = 1.0  # Desired flight altitude
    safe_dist = 1.0
    collide_dist = 0.6
    goal_reset_delay = 1.0  # Delay for resetting goal after reaching it
    mission_names = ["migration", "crossover", "chaotic"]
    success_distance_threshold = 0.5  # Distance threshold for considering goal reached
    max_sampling_tries = 100  # Maximum number of attempts to sample a valid initial state or goal
    migration_goal_range = 5.0  # Range of xy coordinates of the goal in mission "migration"
    chaotic_goal_range = 3.0  # Range of xy coordinates of the goal in mission "chaotic"
    rand_init_states = True  # Whether to randomly permute initial states among agents in the migration mission
    birth_circle_radius = 2.5

    # TODO: Improve dirty curriculum
    enable_dirty_curriculum = True
    curriculum_steps = 1e5
    init_death_penalty_weight = 0.01
    init_mutual_collision_avoidance_reward_weight = 0.01

    # Env
    episode_length_s = 30.0
    physics_freq = 200.0
    control_freq = 100.0
    action_freq = 50.0
    gui_render_freq = 50.0
    control_decimation = physics_freq // control_freq
    num_drones = 3  # Number of drones per environment
    decimation = math.ceil(physics_freq / action_freq)  # Environment decimation
    render_decimation = physics_freq // gui_render_freq
    clip_action = 1.0
    possible_agents = None
    action_spaces = None
    history_length = 5
    transient_observasion_dim = 17 + 7 * (num_drones - 1)
    observation_spaces = None
    transient_state_dim = 16 * num_drones
    state_space = None

    def __post_init__(self):
        self.possible_agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.action_spaces = {agent: 4 for agent in self.possible_agents}
        self.observation_spaces = {agent: self.history_length * self.transient_observasion_dim for agent in self.possible_agents}
        self.state_space = self.history_length * self.transient_state_dim
        self.v_desired = {agent: 2.0 for agent in self.possible_agents}
        self.v_max = {agent: 3.0 for agent in self.possible_agents}
        self.thrust_to_weight = {agent: 2.0 for agent in self.possible_agents}
        self.w_max = {agent: 4.0 for agent in self.possible_agents}

    # [xdl]: domain randomization settings
    enable_domain_randomization = False

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=500, env_spacing=15, replicate_physics=True)

    # Robot
    drone_cfg: ArticulationCfg = DJI_FPV_CFG.copy()
    init_gap = 2.0

    # Debug visualization
    debug_vis = True
    debug_vis_goal = True
    debug_vis_safe_dist = False
    debug_vis_rel_pos = True


class SwarmBodyrateEnv(DirectMARLEnv):
    cfg: SwarmBodyrateEnvCfg

    def __init__(self, cfg: SwarmBodyrateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Replan and control decimation must be greater than or equal to 1 #^#")

        self.goals = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.env_mission_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.success_dist_thr = torch.zeros(self.num_envs, device=self.device)
        self.rand_r = torch.zeros(self.num_envs, device=self.device)
        self.ang = torch.zeros(self.num_envs, self.cfg.num_drones, device=self.device)
        self.rand_rg = torch.zeros(self.num_envs, device=self.device)

        # Get specific body indices for each drone
        self.body_ids = {agent: self.robots[agent].find_bodies("body")[0] for agent in self.cfg.possible_agents}

        self.robot_masses = {agent: self.robots[agent].root_physx_view.get_masses()[0, 0].to(self.device) for agent in self.cfg.possible_agents}
        self.robot_inertias = {agent: self.robots[agent].root_physx_view.get_inertias()[0, 0].to(self.device) for agent in self.cfg.possible_agents}
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)
        self.robot_weights = {agent: (self.robot_masses[agent] * self.gravity.norm()).item() for agent in self.cfg.possible_agents}

        # Controller
        self.actions = {}
        self.thrusts_desired_normalized, self.w_desired_normalized = {}, {}  # Componenets of actions
        self.prev_thrusts_desired_normalized, self.prev_w_desired_normalized = {}, {}  # Previous actions

        # Denormalized actions
        self.thrusts_desired = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.w_desired = {}
        self.m_desired = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.possible_agents}

        self.kPw = {agent: torch.tensor([0.05, 0.05, 0.05], device=self.device) for agent in self.cfg.possible_agents}
        self.control_counter = 0

        self.prev_dist_to_goals = {}

        self.relative_positions_w = {
            i: {j: torch.zeros(self.num_envs, 3, device=self.device) for j in range(self.cfg.num_drones) if j != i} for i in range(self.cfg.num_drones)
        }
        self.relative_positions_with_observability = {}

        self.reset_env_ids = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.observation_buffer = {
            agent: torch.zeros(self.cfg.history_length, self.num_envs, self.cfg.transient_observasion_dim, device=self.device) for agent in self.cfg.possible_agents
        }
        self.state_buffer = torch.zeros(self.cfg.history_length, self.num_envs, self.cfg.transient_state_dim, device=self.device)

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        self.visualize_new_cmd = False

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
            self.actions[agent] = actions[agent].clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action
            self.thrusts_desired_normalized[agent] = (self.actions[agent][:, 0] + 1.0) / 2
            self.w_desired_normalized[agent] = self.actions[agent][:, 1:]

            self.thrusts_desired[agent][:, 0, 2] = self.cfg.thrust_to_weight[agent] * self.robot_weights[agent] * self.thrusts_desired_normalized[agent]
            self.w_desired[agent] = self.cfg.w_max[agent] * self.w_desired_normalized[agent]

    def _apply_action(self) -> None:
        if self.control_counter % self.cfg.control_decimation == 0:
            for agent in self.possible_agents:
                self.m_desired[agent][:, 0, :] = bodyrate_control_without_thrust(
                    self.robots[agent].data.root_ang_vel_w, self.w_desired[agent], self.robot_inertias[agent], self.kPw[agent]
                )
            self.control_counter = 0
        self.control_counter += 1

        for agent in self.possible_agents:
            self.robots[agent].set_external_force_and_torque(self.thrusts_desired[agent], self.m_desired[agent], body_ids=self.body_ids[agent])

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:
            z_exceed_bounds = torch.logical_or(self.robots[agent].data.root_link_pos_w[:, 2] < 0.9, self.robots[agent].data.root_link_pos_w[:, 2] > 1.1)
            ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robots[agent].data.root_link_quat_w))
            _died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 60.0)

            died = torch.logical_or(died, _died)

        # Update relative positions, detecting collisions along the way
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j] = self.robots[agent_j].data.root_pos_w - self.robots[agent_i].data.root_pos_w

                died = torch.logical_or(died, torch.linalg.norm(self.relative_positions_w[i][j], dim=1) < self.cfg.collide_dist)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return {agent: died for agent in self.cfg.possible_agents}, {agent: time_out for agent in self.cfg.possible_agents}

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

        rewards = {}

        death_reward = -torch.where(self.terminated_dict["drone_0"], torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))

        mutual_collision_avoidance_reward = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.possible_agents}
        for i, agent in enumerate(self.possible_agents):
            for j, _ in enumerate(self.possible_agents):
                if i == j:
                    continue

                dist_btw_drones = torch.linalg.norm(self.relative_positions_w[i][j], dim=1)

                collision_penalty = 1.0 / (1.0 + torch.exp(77 * (dist_btw_drones - self.cfg.safe_dist)))
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

            # Reward for maintaining height close to flight_altitude
            z_curr = self.robots[agent].data.root_pos_w[:, 2]
            altitude_maintenance_reward = -torch.abs(z_curr - self.cfg.flight_altitude)

            # Reward for maintaining speed close to v_desired
            v_curr = torch.linalg.norm(self.robots[agent].data.root_lin_vel_w, dim=1)
            speed_maintenance_reward = torch.exp(-((torch.abs(v_curr - self.cfg.v_desired[agent]) / self.cfg.speed_deviation_tolerance) ** 2))

            ### ============= Smoothing ============= ###
            ang_vel_reward = -torch.linalg.norm(self.w_desired_normalized[agent], dim=1)

            lin_vel = torch.linalg.norm(self.robots[agent].data.root_lin_vel_w, dim=1)
            lin_vel_reward = torch.where(lin_vel > self.cfg.v_max[agent], -torch.exp(torch.abs(lin_vel - self.cfg.v_max[agent])) + 1.0, torch.zeros_like(lin_vel))
            lin_vel_reward = torch.where(lin_vel_reward < -200, -200 * torch.ones_like(lin_vel_reward), lin_vel_reward)

            thrust_diff_reward = -torch.abs(self.thrusts_desired_normalized[agent] - self.prev_thrusts_desired_normalized[agent])

            ang_vel_diff_reward = -torch.linalg.norm(self.w_desired_normalized[agent] - self.prev_w_desired_normalized[agent], dim=1)

            reward = {
                "approaching_goal": approaching_goal_reward * self.cfg.approaching_goal_reward_weight * self.step_dt,
                "dist_to_goal": dist_to_goal_reward * self.cfg.dist_to_goal_reward_weight * self.step_dt,
                "success": success_reward * self.cfg.success_reward_weight * self.step_dt,
                "death_penalty": death_reward * self.cfg.death_penalty_weight,
                "time_penalty": time_reward * self.cfg.time_penalty_weight * self.step_dt,
                "altitude_maintenance": altitude_maintenance_reward * self.cfg.altitude_maintenance_reward_weight * self.step_dt,
                "speed_maintenance": speed_maintenance_reward * self.cfg.speed_maintenance_reward_weight * self.step_dt,
                "mutual_collision_avoidance": mutual_collision_avoidance_reward[agent]
                / self.cfg.num_drones
                * self.cfg.mutual_collision_avoidance_reward_weight
                * self.step_dt,
                ### ============= Smoothing ============= ###
                "ang_vel_penalty": ang_vel_reward * self.cfg.ang_vel_penalty_weight * self.step_dt,
                "lin_vel_penalty": lin_vel_reward * self.cfg.lin_vel_penalty_weight * self.step_dt,
                "thrust_diff_penalty": thrust_diff_reward * self.cfg.thrust_diff_penalty_weight * self.step_dt,
                "ang_vel_diff_penalty": ang_vel_diff_reward * self.cfg.ang_vel_diff_penalty_weight * self.step_dt,
            }

            reward = torch.sum(torch.stack(list(reward.values())), dim=0)

            rewards[agent] = reward
        return rewards

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["drone_0"]._ALL_INDICES

        # Most (but only most) of the time self.reset_env_ids is equal to self.reset_buf
        self.reset_env_ids[env_ids] = True

        # FIXME: Logging
        self.extras = {}

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
            agents = list(self.possible_agents)
            permuted = agents.copy()
            if self.cfg.rand_init_states:
                random.shuffle(permuted)
            mapping = dict(zip(agents, permuted))

        if len(mission_1_ids) > 0:
            r_min, r_max = self.cfg.birth_circle_radius, 2 * self.cfg.birth_circle_radius
            self.rand_r[mission_1_ids] = torch.rand(len(mission_1_ids), device=self.device) * (r_max - r_min) + r_min

            for idx in mission_1_ids.tolist():
                r = self.rand_r[idx]
                for attempt in range(self.cfg.max_sampling_tries):
                    rand_ang = torch.rand(self.cfg.num_drones, device=self.device) * 2 * math.pi
                    pts = torch.stack([torch.cos(rand_ang) * r, torch.sin(rand_ang) * r], dim=1)
                    dmat = torch.cdist(pts, pts)
                    dmat.fill_diagonal_(float("inf"))
                    last_rand_ang = rand_ang
                    if torch.min(dmat) >= 1.314 * self.cfg.safe_dist:
                        self.ang[idx] = rand_ang
                        break
                else:
                    logger.warning(f"The search for initial positions of the swarm meeting constraints on a radius {r} circle failed, using the final sample #_#")
                    self.ang[idx] = last_rand_ang

        if len(mission_2_ids) > 0:
            rg_min, rg_max = self.cfg.chaotic_goal_range, 1.5 * self.cfg.chaotic_goal_range
            self.rand_rg[mission_2_ids] = torch.rand(len(mission_2_ids), device=self.device) * (rg_max - rg_min) + rg_min
            init_p = torch.zeros(self.num_envs, self.cfg.num_drones, 2, device=self.device)
            goal_p = torch.zeros(self.num_envs, self.cfg.num_drones, 2, device=self.device)

            for idx in mission_2_ids.tolist():
                rg = self.rand_rg[idx]
                for attempt in range(self.cfg.max_sampling_tries):
                    rand_pts = (torch.rand(self.cfg.num_drones, 2, device=self.device) * 2 - 1) * rg
                    dmat = torch.cdist(rand_pts, rand_pts)
                    dmat.fill_diagonal_(float("inf"))
                    last_rand_pts = rand_pts
                    if torch.min(dmat) >= 1.314 * self.cfg.safe_dist:
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
                    if torch.min(dmat) >= 1.314 * self.cfg.safe_dist:
                        goal_p[idx] = rand_pts
                        break
                else:
                    logger.warning(f"The search for goal positions of the swarm meeting constraints within a side-length {rg} box failed, using the final sample #_#")
                    goal_p[idx] = last_rand_pts

        for i, agent in enumerate(self.possible_agents):
            init_state = self.robots[agent].data.default_root_state.clone()

            # The migration mission: default init states + unified random target
            if len(mission_0_ids) > 0:
                rand_other_agent = mapping[agent]
                init_state[mission_0_ids] = self.robots[rand_other_agent].data.default_root_state[mission_0_ids].clone()

                if agent == "drone_0":
                    unified_goal_xy = torch.zeros_like(self.goals[agent][mission_0_ids, :2]).uniform_(-self.cfg.migration_goal_range, self.cfg.migration_goal_range)
                self.goals[agent][mission_0_ids, :2] = unified_goal_xy.clone()

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

            self.goals[agent][env_ids, 2] = float(self.cfg.flight_altitude)
            self.goals[agent][env_ids] += self.terrain.env_origins[env_ids]
            init_state[env_ids, :3] += self.terrain.env_origins[env_ids]

            self.robots[agent].write_root_pose_to_sim(init_state[env_ids, :7], env_ids)
            self.robots[agent].write_root_velocity_to_sim(init_state[env_ids, 7:], env_ids)
            self.robots[agent].write_joint_state_to_sim(
                self.robots[agent].data.default_joint_pos[env_ids], self.robots[agent].data.default_joint_vel[env_ids], None, env_ids
            )

            if agent in self.prev_dist_to_goals:
                self.prev_dist_to_goals[agent][env_ids] = torch.linalg.norm(self.goals[agent][env_ids] - self.robots[agent].data.root_pos_w[env_ids], dim=1)
            else:
                self.prev_dist_to_goals[agent] = torch.linalg.norm(self.goals[agent] - self.robots[agent].data.root_pos_w, dim=1)

            if agent in self.prev_thrusts_desired_normalized:
                self.prev_thrusts_desired_normalized[agent][env_ids] = (torch.zeros_like(self.thrusts_desired_normalized[agent][env_ids]) + 1.0) / 2
            else:
                self.prev_thrusts_desired_normalized[agent] = (torch.zeros(self.num_envs, device=self.device) + 1.0) / 2

            if agent in self.prev_w_desired_normalized:
                self.prev_w_desired_normalized[agent][env_ids] = torch.zeros_like(self.w_desired_normalized[agent][env_ids])
            else:
                self.prev_w_desired_normalized[agent] = torch.zeros(self.num_envs, 3, device=self.device)

        self.reset_goal_timer[env_ids] = 0.0

        # Update relative positions
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j][env_ids] = self.robots[agent_j].data.root_pos_w[env_ids] - self.robots[agent_i].data.root_pos_w[env_ids]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Reset goal after _get_rewards before _get_observations and _get_states
        # Synchronous goal resetting in all missions
        # (A mix of synchronous and asynchronous goal resetting may cause state to lose Markovianity :(
        success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:
            dist_to_goal = torch.linalg.norm(self.goals[agent] - self.robots[agent].data.root_pos_w, dim=1)
            success = torch.logical_or(success, dist_to_goal < self.success_dist_thr)

        if success.any():
            self.reset_goal_timer[success] += self.step_dt

        reset_goal_idx = (self.reset_goal_timer > self.cfg.goal_reset_delay).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_goal_idx) > 0:
            mission_0_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 0]  # The migration mission
            mission_1_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 1]  # The crossover mission
            mission_2_ids = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 2]  # The chaotic mission

            if len(mission_0_ids) > 0:
                for agent in self.possible_agents:
                    if agent == "drone_0":
                        unified_goal_xy = torch.zeros_like(self.goals[agent][mission_0_ids, :2]).uniform_(-self.cfg.migration_goal_range, self.cfg.migration_goal_range)
                    self.goals[agent][mission_0_ids, :2] = unified_goal_xy.clone()

            if len(mission_1_ids) > 0:
                for i, agent in enumerate(self.possible_agents):
                    self.ang[mission_1_ids, i] += math.pi
                    self.goals[agent][mission_1_ids, :2] = torch.stack(
                        [torch.cos(self.ang[mission_1_ids, i]), torch.sin(self.ang[mission_1_ids, i])], dim=1
                    ) * self.rand_r[mission_1_ids].unsqueeze(1)

            if len(mission_2_ids) > 0:
                goal_p = torch.zeros(self.num_envs, self.cfg.num_drones, 2, device=self.device)
                for idx in mission_2_ids.tolist():
                    rg = self.rand_rg[idx]

                    for attempt in range(self.cfg.max_sampling_tries):
                        rand_pts = (torch.rand(self.cfg.num_drones, 2, device=self.device) * 2 - 1) * rg
                        dmat = torch.cdist(rand_pts, rand_pts)
                        dmat.fill_diagonal_(float("inf"))
                        last_rand_pts = rand_pts
                        if torch.min(dmat) >= 1.314 * self.cfg.safe_dist:
                            goal_p[idx] = rand_pts
                            break
                    else:
                        logger.warning(f"The search for goal positions of the swarm meeting constraints within a side-length {rg} box failed, using the final sample #_#")
                        goal_p[idx] = last_rand_pts

                for i, agent in enumerate(self.possible_agents):
                    self.goals[agent][mission_2_ids, :2] = goal_p[mission_2_ids, i]

            for agent in self.possible_agents:
                self.goals[agent][reset_goal_idx, 2] = float(self.cfg.flight_altitude)
                self.goals[agent][reset_goal_idx] += self.terrain.env_origins[reset_goal_idx]

                # FIXME: Whether ⬇️ should exist?
                # self.prev_dist_to_goals[agent][reset_goal_idx] = torch.linalg.norm(
                #     self.goals[agent][reset_goal_idx] - self.robots[agent].data.root_pos_w[reset_goal_idx], dim=1
                # )

        self.reset_goal_timer[reset_goal_idx] = 0.0

        curr_observations = {}
        for i, agent_i in enumerate(self.possible_agents):
            body2goal_w = self.goals[agent_i] - self.robots[agent_i].data.root_pos_w

            # TODO: To be removed from observation
            body2goal_others_w = []

            relative_positions_with_observability = []
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue

                body2goal_others_w.append(self.goals[agent_j] - self.robots[agent_i].data.root_pos_w)

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

                relative_positions_with_observability.append(torch.cat([relative_positions_w, observability_mask.unsqueeze(-1)], dim=-1))
            self.relative_positions_with_observability[agent_i] = torch.cat(relative_positions_with_observability, dim=-1)

            body2goal_others_w = torch.cat(body2goal_others_w, dim=-1)

            obs = torch.cat(
                [
                    self.prev_thrusts_desired_normalized[agent_i].unsqueeze(-1),
                    self.prev_w_desired_normalized[agent_i].clone(),
                    body2goal_w,
                    self.robots[agent_i].data.root_quat_w.clone(),
                    self.robots[agent_i].data.root_vel_w.clone(),  # TODO: Try to discard velocity observations to reduce sim2real gap
                    body2goal_others_w,
                    self.relative_positions_with_observability[agent_i].clone(),
                ],
                dim=-1,
            )
            curr_observations[agent_i] = obs

            # TODO: Where would it make more sense to place ⬇️?
            non_reset_env_ids = ~self.reset_env_ids
            if non_reset_env_ids.any():
                self.prev_thrusts_desired_normalized[agent_i][non_reset_env_ids] = self.thrusts_desired_normalized[agent_i][non_reset_env_ids].clone()
                self.prev_w_desired_normalized[agent_i][non_reset_env_ids] = self.w_desired_normalized[agent_i][non_reset_env_ids].clone()

        # Scroll or reset (fill in the first frame) the observation buffer
        for agent in self.cfg.possible_agents:
            buf = self.observation_buffer[agent]
            if self.reset_env_ids.any():
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

            if self.cfg.debug_vis_safe_dist:
                if not hasattr(self, "safe_dist_visualizers"):
                    self.safe_dist_visualizers = {}
                    for i, agent in enumerate(self.possible_agents):
                        marker_cfg = VisualizationMarkersCfg(
                            prim_path=f"/Visuals/safe_dist_{i}",
                            markers={
                                "cylinder": sim_utils.CylinderCfg(
                                    radius=self.cfg.collide_dist / 2,
                                    height=0.005,
                                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.01, 0.01), roughness=0.0),
                                )
                            },
                        )
                        self.safe_dist_visualizers[agent] = VisualizationMarkers(marker_cfg)
                        self.safe_dist_visualizers[agent].set_visibility(True)

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

        if hasattr(self, "safe_dist_visualizers"):
            for agent in self.possible_agents:
                t = self.robots[agent].data.root_pos_w.clone()
                t[:, 2] -= 0.077
                self.safe_dist_visualizers[agent].visualize(translations=t)

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
            # Select → [num_envs, num_drones-1, 4]
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


from config import agents


gym.register(
    id="FAST-Swarm-Bodyrate",
    entry_point=SwarmBodyrateEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SwarmBodyrateEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:swarm_sb3_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_mappo_cfg.yaml",
    },
)
