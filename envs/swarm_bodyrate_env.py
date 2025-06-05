from __future__ import annotations

import gymnasium as gym
import math
import random
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, ViewerCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_inv, quat_rotate
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from envs.quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils.utils import quat_to_ang_between_z_body_and_z_world
from utils.controller import bodyrate_control_without_thrust


@configclass
class SwarmBodyrateEnvCfg(DirectMARLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # Reward weights
    approaching_goal_reward_weight = 1.0
    dist_to_goal_reward_weight = 0.0
    success_reward_weight = 100.0
    time_penalty_weight = 0.0
    speed_maintenance_reward_weight = 0.0  # Reward for maintaining speed close to v_desired
    mutual_collision_avoidance_reward_weight = 10.0

    # Exponential decay factors and tolerances
    dist_to_goal_scale = 0.5
    speed_deviation_tolerance = 0.5

    flight_altitude = 1.0  # Desired flight altitude
    success_distance_threshold = 0.2  # Distance threshold for considering goal reached
    safe_dist = 2.0
    goal_reset_period = 10.0  # Time period for resetting goal
    mission_names = ["migration", "crossover", "chaotic"]
    goal_range = 4.0  # Range of xy coordinates of the goal
    rand_init_states = True  # Whether to randomly permute initial states among agents in the migration mission
    init_circle_radius = 5.0

    # TODO: Improve dirty cirriculum
    enable_dirty_cirriculum = True
    training_steps = 2e5
    init_mutual_collision_avoidance_reward_weight = 0.1
    init_success_distance_threshold = 1.0
    init_safe_dist = 1.0
    init_goal_range = 10.0

    # Env
    episode_length_s = 30.0
    physics_freq = 200.0
    control_freq = 100.0
    action_freq = 10.0
    gui_render_freq = 50.0
    control_decimation = physics_freq // control_freq
    num_drones = 6  # Number of drones per environment
    decimation = math.ceil(physics_freq / action_freq)  # Environment decimation
    render_decimation = physics_freq // gui_render_freq
    clip_action = 1.0
    possible_agents = None
    action_spaces = None
    observation_spaces = None
    state_space = None

    def __post_init__(self):
        self.possible_agents = [f"drone_{i}" for i in range(self.num_drones)]
        self.action_spaces = {agent: 4 for agent in self.possible_agents}
        self.observation_spaces = {agent: 16 + 3 * (self.num_drones - 1) for agent in self.possible_agents}
        self.state_space = 16 * self.num_drones
        self.v_desired = {agent: 2.0 for agent in self.possible_agents}
        self.thrust_to_weight = {agent: 2.0 for agent in self.possible_agents}
        self.w_max = {agent: 1.0 for agent in self.possible_agents}

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=500, env_spacing=10, replicate_physics=True)

    # Robot
    drone_cfg: ArticulationCfg = DJI_FPV_CFG.copy()
    init_gap = 2.0

    # Debug visualization
    debug_vis = True
    debug_vis_goal = True


class SwarmBodyrateEnv(DirectMARLEnv):
    cfg: SwarmBodyrateEnvCfg

    def __init__(self, cfg: SwarmBodyrateEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Replan and control decimation must be greater than or equal to 1 #^#")

        self.goal = {agent: torch.zeros(self.num_envs, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.env_mission_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.reset_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.ang = {agent: torch.zeros(self.num_envs, device=self.device) for agent in self.cfg.possible_agents}

        # Get specific body indices for each drone
        self.body_ids = {agent: self.robots[agent].find_bodies("body")[0] for agent in self.cfg.possible_agents}

        self.robot_masses = {agent: self.robots[agent].root_physx_view.get_masses()[0, 0].to(self.device) for agent in self.cfg.possible_agents}
        self.robot_inertias = {agent: self.robots[agent].root_physx_view.get_inertias()[0, 0].to(self.device) for agent in self.cfg.possible_agents}
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)
        self.robot_weights = {agent: (self.robot_masses[agent] * self.gravity.norm()).item() for agent in self.cfg.possible_agents}

        # Controller
        self.actions, self.w_desired = {}, {}
        self.thrusts = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.moments = {agent: torch.zeros(self.num_envs, 1, 3, device=self.device) for agent in self.cfg.possible_agents}
        self.kPw = {agent: torch.tensor([0.05, 0.05, 0.05], device=self.device) for agent in self.cfg.possible_agents}
        self.control_counter = 0

        self.prev_dist_to_goal = {}

        self.relative_positions_w = {}
        for i in range(self.cfg.num_drones):
            self.relative_positions_w[i] = {}
            for j in range(self.cfg.num_drones):
                if i == j:
                    continue
                self.relative_positions_w[i][j] = torch.zeros(self.num_envs, 3, device=self.device)

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
        for agent in self.possible_agents:
            self.actions[agent] = actions[agent].clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action
            self.thrusts[agent][:, 0, 2] = self.cfg.thrust_to_weight[agent] * self.robot_weights[agent] * (self.actions[agent][:, 0] + 1.0) / 2.0
            self.w_desired[agent] = self.actions[agent][:, 1:] * self.cfg.w_max[agent]

    def _apply_action(self) -> None:
        if self.control_counter % self.cfg.control_decimation == 0:
            for agent in self.possible_agents:
                self.moments[agent][:, 0, :] = bodyrate_control_without_thrust(
                    self.robots[agent].data.root_ang_vel_w, self.w_desired[agent], self.robot_inertias[agent], self.kPw[agent]
                )
            self.control_counter = 0
        self.control_counter += 1

        for agent in self.possible_agents:
            self.robots[agent].set_external_force_and_torque(self.thrusts[agent], self.moments[agent], body_ids=self.body_ids[agent])

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:
            z_exceed_bounds = torch.logical_or(self.robots[agent].data.root_link_pos_w[:, 2] < 0.5, self.robots[agent].data.root_link_pos_w[:, 2] > 2.0)
            ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robots[agent].data.root_link_quat_w))
            _died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 60.0)

            died = torch.logical_or(died, _died)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return {agent: died for agent in self.cfg.possible_agents}, {agent: time_out for agent in self.cfg.possible_agents}

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        # Update relative positions before _get_rewards
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j] = self.robots[agent_j].data.root_pos_w - self.robots[agent_i].data.root_pos_w

        # TODO: Improve dirty cirriculum
        if self.cfg.enable_dirty_cirriculum:
            if not hasattr(self, "delta_mutual_collision_avoidance_reward_weight"):
                self.final_mutual_collision_avoidance_reward_weight = self.cfg.mutual_collision_avoidance_reward_weight
                self.cfg.mutual_collision_avoidance_reward_weight = self.cfg.init_mutual_collision_avoidance_reward_weight
                self.delta_mutual_collision_avoidance_reward_weight = (
                    self.final_mutual_collision_avoidance_reward_weight - self.cfg.init_mutual_collision_avoidance_reward_weight
                ) / self.cfg.training_steps
            else:
                self.cfg.mutual_collision_avoidance_reward_weight += self.delta_mutual_collision_avoidance_reward_weight

            if not hasattr(self, "delta_success_distance_threshold"):
                self.final_success_distance_threshold = self.cfg.success_distance_threshold
                self.cfg.success_distance_threshold = self.cfg.init_success_distance_threshold
                self.delta_success_distance_threshold = (self.final_success_distance_threshold - self.cfg.init_success_distance_threshold) / self.cfg.training_steps
            else:
                self.cfg.success_distance_threshold += self.delta_success_distance_threshold

            if not hasattr(self, "delta_safe_dist"):
                self.final_safe_dist = self.cfg.safe_dist
                self.cfg.safe_dist = self.cfg.init_safe_dist
                self.delta_safe_dist = (self.final_safe_dist - self.cfg.init_safe_dist) / self.cfg.training_steps
            else:
                self.cfg.safe_dist += self.delta_safe_dist

            if not hasattr(self, "delta_goal_range"):
                self.final_goal_range = self.cfg.goal_range
                self.cfg.goal_range = self.cfg.init_goal_range
                self.delta_goal_range = (self.final_goal_range - self.cfg.init_goal_range) / self.cfg.training_steps
            else:
                self.cfg.goal_range += self.delta_goal_range

        rewards = {}

        mutual_collision_avoidance_reward = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.cfg.num_drones):
            for j in range(i + 1, self.cfg.num_drones):
                dist_btw_drones = torch.linalg.norm(self.relative_positions_w[i][j], dim=1)

                collision_penalty = 1.0 / (1.0 + torch.exp(77 * (dist_btw_drones - self.cfg.safe_dist)))
                mutual_collision_avoidance_reward -= collision_penalty

        for agent in self.possible_agents:
            dist_to_goal = torch.linalg.norm(self.goal[agent] - self.robots[agent].data.root_pos_w, dim=1)
            approaching_goal_reward = torch.zeros(self.num_envs, device=self.device)
            if agent in self.prev_dist_to_goal:
                approaching_goal_reward = self.prev_dist_to_goal[agent] - dist_to_goal
            self.prev_dist_to_goal[agent] = dist_to_goal

            dist_to_goal_reward = torch.exp(-self.cfg.dist_to_goal_scale * dist_to_goal)

            success = dist_to_goal < self.cfg.success_distance_threshold
            unsuccess = ~success
            # Additional reward when the drone is close to goal
            success_reward = torch.where(success, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))
            # Time penalty for not reaching the goal
            time_reward = -torch.where(unsuccess, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))

            # Reward for maintaining speed close to v_desired
            v_curr = torch.linalg.norm(self.robots[agent].data.root_lin_vel_w, dim=1)
            speed_maintenance_reward = torch.exp(-((torch.abs(v_curr - self.cfg.v_desired[agent]) / self.cfg.speed_deviation_tolerance) ** 2))

            reward = {
                "approaching_goal": approaching_goal_reward * self.cfg.approaching_goal_reward_weight * self.step_dt,
                "dist_to_goal": dist_to_goal_reward * self.cfg.dist_to_goal_reward_weight * self.step_dt,
                "success": success_reward * self.cfg.success_reward_weight * self.step_dt,
                "time_penalty": time_reward * self.cfg.time_penalty_weight * self.step_dt,
                "speed_maintenance": speed_maintenance_reward * self.cfg.speed_maintenance_reward_weight * self.step_dt,
                "mutual_collision_avoidance": mutual_collision_avoidance_reward / self.cfg.num_drones * self.cfg.mutual_collision_avoidance_reward_weight * self.step_dt,
            }
            reward = torch.sum(torch.stack(list(reward.values())), dim=0)

            rewards[agent] = reward
        return rewards

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["drone_0"]._ALL_INDICES

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
        mission_0_envs = env_ids[self.env_mission_ids[env_ids] == 0]  # The migration mission
        mission_1_envs = env_ids[self.env_mission_ids[env_ids] == 1]  # The crossover mission
        mission_2_envs = env_ids[self.env_mission_ids[env_ids] == 2]  # The chaotic mission

        # Reset robot state
        # Randomly permute initial root and joint states among agents
        agents = list(self.possible_agents)
        permuted = agents.copy()
        if self.cfg.rand_init_states:
            random.shuffle(permuted)
        mapping = dict(zip(agents, permuted))

        rand_init_ang = torch.rand(len(mission_1_envs), device=self.device)
        rand_init_order = list(range(len(self.possible_agents)))
        random.shuffle(rand_init_order)

        for i, agent in enumerate(self.possible_agents):
            init_state = self.robots[agent].data.default_root_state

            # The migration mission: default init states + unified random target
            if len(mission_0_envs) > 0:
                rand_other_agent = mapping[agent]
                init_state[mission_0_envs] = self.robots[rand_other_agent].data.default_root_state[mission_0_envs]

                self.goal[agent][mission_0_envs, :2] = torch.zeros_like(self.goal[agent][mission_0_envs, :2]).uniform_(-self.cfg.goal_range, self.cfg.goal_range)

            # The crossover mission: init states uniformly distributed on a circle + target on the opposite side
            if len(mission_1_envs) > 0:
                self.ang[agent][mission_1_envs] = (rand_init_ang + rand_init_order[i] / self.cfg.num_drones) * 2 * math.pi  # Start angles
                init_state[mission_1_envs, :2] = (
                    torch.stack([torch.cos(self.ang[agent][mission_1_envs]), torch.sin(self.ang[agent][mission_1_envs])], dim=1) * self.cfg.init_circle_radius
                )

                self.ang[agent][mission_1_envs] += math.pi  # Terminate angles
                self.goal[agent][mission_1_envs, :2] = (
                    torch.stack([torch.cos(self.ang[agent][mission_1_envs]), torch.sin(self.ang[agent][mission_1_envs])], dim=1) * self.cfg.init_circle_radius
                )

            # The chaotic mission: random init states + respective random target
            if len(mission_2_envs) > 0:
                init_state[mission_2_envs, :2] = torch.zeros_like(init_state[mission_2_envs, :2]).uniform_(-self.cfg.goal_range, self.cfg.goal_range)

                self.goal[agent][mission_2_envs, :2] = torch.zeros_like(self.goal[agent][mission_2_envs, :2]).uniform_(-self.cfg.goal_range, self.cfg.goal_range)

            self.goal[agent][env_ids, 2] = float(self.cfg.flight_altitude)
            self.goal[agent][env_ids] += self.terrain.env_origins[env_ids]
            init_state[env_ids, :3] += self.terrain.env_origins[env_ids]

            self.robots[agent].write_root_pose_to_sim(init_state[env_ids, :7], env_ids)
            self.robots[agent].write_root_velocity_to_sim(init_state[env_ids, 7:], env_ids)
            self.robots[agent].write_joint_state_to_sim(
                self.robots[agent].data.default_joint_pos[env_ids], self.robots[agent].data.default_joint_vel[env_ids], None, env_ids
            )

            if agent in self.prev_dist_to_goal:
                self.prev_dist_to_goal[agent][env_ids] = torch.linalg.norm(self.goal[agent][env_ids] - self.robots[agent].data.root_pos_w[env_ids], dim=1)

        self.reset_goal_timer[env_ids] = 0.0

        # Update relative positions
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j][env_ids] = self.robots[agent_j].data.root_pos_w[env_ids] - self.robots[agent_i].data.root_pos_w[env_ids]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        # Reset goal after _get_rewards before _get_observations and _get_states
        success = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:
            dist_to_goal = torch.linalg.norm(self.goal[agent] - self.robots[agent].data.root_pos_w, dim=1)
            success = torch.logical_and(success, dist_to_goal < self.cfg.success_distance_threshold)

        self.reset_goal_timer += self.step_dt

        reset_goal_idx = ((self.reset_goal_timer > self.cfg.goal_reset_period) | success).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_goal_idx) > 0:
            mission_0_envs = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 0]  # The migration mission
            mission_1_envs = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 1]  # The crossover mission
            mission_2_envs = reset_goal_idx[self.env_mission_ids[reset_goal_idx] == 2]  # The chaotic mission

            for agent in self.possible_agents:
                if len(mission_0_envs) > 0:
                    self.goal[agent][mission_0_envs, :2] = torch.zeros_like(self.goal[agent][mission_0_envs, :2]).uniform_(-self.cfg.goal_range, self.cfg.goal_range)

                if len(mission_1_envs) > 0:
                    self.ang[agent][mission_1_envs] += math.pi
                    self.goal[agent][mission_1_envs, :2] = (
                        torch.stack([torch.cos(self.ang[agent][mission_1_envs]), torch.sin(self.ang[agent][mission_1_envs])], dim=1) * self.cfg.init_circle_radius
                    )

                if len(mission_2_envs) > 0:
                    self.goal[agent][mission_2_envs, :2] = torch.zeros_like(self.goal[agent][mission_2_envs, :2]).uniform_(-self.cfg.goal_range, self.cfg.goal_range)

                self.goal[agent][reset_goal_idx, 2] = float(self.cfg.flight_altitude)
                self.goal[agent][reset_goal_idx] += self.terrain.env_origins[reset_goal_idx]

            self.reset_goal_timer[reset_goal_idx] = 0.0

        observations = {}

        for i, agent in enumerate(self.possible_agents):
            body2goal_w = self.goal[agent] - self.robots[agent].data.root_pos_w

            relative_positions_w = []
            for j, _ in enumerate(self.possible_agents):
                if i == j:
                    continue
                relative_positions_w.append(self.relative_positions_w[i][j].clone())
            relative_positions_w = torch.cat(relative_positions_w, dim=-1)

            obs = torch.cat(
                [
                    body2goal_w,
                    self.robots[agent].data.root_quat_w.clone(),
                    self.robots[agent].data.projected_gravity_b.clone(),
                    self.robots[agent].data.root_vel_w.clone(),  # TODO: Try to discard velocity observations to reduce sim2real gap
                    relative_positions_w,
                ],
                dim=-1,
            )
            observations[agent] = obs

        return observations

    def _get_states(self):
        state = []
        for agent in self.possible_agents:
            body2goal_w = self.goal[agent] - self.robots[agent].data.root_pos_w
            state.append(body2goal_w)
            state.append(self.robots[agent].data.root_quat_w.clone())
            state.append(self.robots[agent].data.projected_gravity_b.clone())
            state.append(self.robots[agent].data.root_vel_w.clone())

        return torch.cat(state, dim=-1)

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

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_visualizers"):
            for agent in self.possible_agents:
                self.goal_visualizers[agent].visualize(translations=self.goal[agent])


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
