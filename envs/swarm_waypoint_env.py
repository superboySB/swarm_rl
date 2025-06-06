from __future__ import annotations

import gymnasium as gym
from loguru import logger
import math
import random
import time
import torch
from collections.abc import Sequence

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
from utils.minco import MinJerkOpt
from utils.controller import Controller


@configclass
class SwarmWaypointEnvCfg(DirectMARLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # Reward weights
    approaching_goal_reward_weight = 1.0
    dist_to_goal_reward_weight = 0.0
    tail_wp_dist_to_goal_reward_weight = 0.0
    success_reward_weight = 100.0
    time_penalty_weight = 0.0
    speed_maintenance_reward_weight = 0.0  # Reward for maintaining speed close to v_max
    mutual_collision_avoidance_reward_weight = 0.001

    # Exponential decay factors and tolerances
    dist_to_goal_scale = 0.5
    tail_wp_dist_to_goal_scale = 0.5
    speed_deviation_tolerance = 0.5

    flight_altitude = 1.0  # Desired flight altitude
    success_distance_threshold = 1.0  # Distance threshold for considering goal reached
    goal_reset_period = 10.0  # Time period for resetting goal
    goal_range = 10.0  # Range of xy coordinates of the goal
    safe_dist = 0.5
    rand_init_states = False  # Whether to randomly permute initial states among agents

    # Env
    episode_length_s = 30.0
    physics_freq = 200.0
    control_freq = 100.0
    mpc_freq = 10.0
    gui_render_freq = 50.0
    control_decimation = physics_freq // control_freq
    num_drones = 4  # Number of drones per environment
    decimation = math.ceil(physics_freq / mpc_freq)  # Environment (replan) decimation
    render_decimation = physics_freq // gui_render_freq
    possible_agents = [f"drone_{i}" for i in range(num_drones)]
    state_space = -1

    # MINCO trajectory
    num_pieces = 1
    duration = 0.5
    a_max = {agent: 10.0 for agent in possible_agents}
    v_max = {agent: 3.0 for agent in possible_agents}
    p_max = {agent: 1.0 for agent in possible_agents}

    # FIXME: @configclass doesn't support the following syntax #^#
    # observation_spaces = {agent: 16 + 3 * (num_drones - 1) for agent in possible_agents}
    observation_spaces = {agent: 16 + 3 * (4 - 1) for agent in possible_agents}
    # action_space = {agent: 3 * (num_pieces + 2) for agent in possible_agents}  # inner_pts 3 x (num_pieces - 1) + tail_pva 3 x 3
    action_spaces = {agent: 3 * (1 + 2) for agent in possible_agents}
    clip_action = 1.0

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
    debug_vis_action = True
    debug_vis_traj = True


class SwarmWaypointEnv(DirectMARLEnv):
    cfg: SwarmWaypointEnvCfg

    def __init__(self, cfg: SwarmWaypointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Replan and control decimation must be greater than or equal to 1 #^#")

        if 1 / self.cfg.mpc_freq > self.cfg.num_pieces * self.cfg.duration:
            raise ValueError("Replan period must be less than or equal to the total trajectory duration #^#")

        # Goal position
        self.goals = torch.zeros(self.num_envs, 3, device=self.device)
        self.reset_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Get specific body indices for each drone
        self.body_ids = {agent: self.robots[agent].find_bodies("body")[0] for agent in self.cfg.possible_agents}

        self.robot_masses = {agent: self.robots[agent].root_physx_view.get_masses()[0, 0] for agent in self.cfg.possible_agents}
        self.robot_inertias = {agent: self.robots[agent].root_physx_view.get_inertias()[0, 0] for agent in self.cfg.possible_agents}
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # Traj
        self.waypoints, self.trajs, self.p_tail = {}, {}, {}
        self.has_prev_traj = torch.tensor([False] * self.num_envs, device=self.device)

        # Controllers
        self.actions, self.a_desired_total, self.thrust_desired, self._thrust_desired, self.q_desired, self.w_desired, self.m_desired = {}, {}, {}, {}, {}, {}, {}
        self.controllers = {
            agent: Controller(
                1 / self.cfg.control_freq, self.gravity, self.robot_masses[agent].to(self.device), self.robot_inertias[agent].to(self.device), self.num_envs
            )
            for agent in self.cfg.possible_agents
        }
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
        head_pva_all, inner_pts_all, tail_pva_all, durations_all = [], [], [], []
        for agent in self.possible_agents:
            # Action parametrization: waypoints in body frame
            self.waypoints[agent] = actions[agent].clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action

            for i in range(self.cfg.num_pieces - 1):
                self.waypoints[agent][:, 3 * (i + 1) : 3 * (i + 2)] += self.waypoints[agent][:, 3 * i : 3 * (i + 1)]

            # Head states
            p_odom = self.robots[agent].data.root_state_w[:, :3]
            q_odom = self.robots[agent].data.root_state_w[:, 3:7]
            v_odom = self.robots[agent].data.root_state_w[:, 7:10]
            a_odom = torch.zeros_like(v_odom)
            if self.trajs:
                a_odom = torch.where(self.has_prev_traj.unsqueeze(1), self.trajs[agent].get_acc(self.execution_time), a_odom)
            head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)
            head_pva_all.append(head_pva)

            # Waypoints
            inner_pts = torch.zeros((self.num_envs, 3, self.cfg.num_pieces - 1), device=self.device)
            for i in range(self.cfg.num_pieces - 1):
                # Transform to world frame
                inner_pts[:, :, i] = self.waypoints[agent][:, 3 * i : 3 * (i + 1)] * self.cfg.p_max[agent] + p_odom
            inner_pts_all.append(inner_pts)

            # Tail states, transformed to world frame
            self.p_tail[agent] = self.waypoints[agent][:, 3 * (self.cfg.num_pieces - 1) : 3 * (self.cfg.num_pieces + 0)] * self.cfg.p_max[agent] + p_odom
            v_tail = self.waypoints[agent][:, 3 * (self.cfg.num_pieces + 0) : 3 * (self.cfg.num_pieces + 1)] * self.cfg.v_max[agent]
            a_tail = self.waypoints[agent][:, 3 * (self.cfg.num_pieces + 1) : 3 * (self.cfg.num_pieces + 2)] * self.cfg.a_max[agent]
            tail_pva = torch.stack([self.p_tail[agent], v_tail, a_tail], dim=2)
            tail_pva_all.append(tail_pva)

            durations = torch.full((self.num_envs, self.cfg.num_pieces), self.cfg.duration, device=self.device)
            durations_all.append(durations)

        head_pva_all = torch.cat(head_pva_all, dim=0)
        inner_pts_all = torch.cat(inner_pts_all, dim=0)
        tail_pva_all = torch.cat(tail_pva_all, dim=0)
        durations_all = torch.cat(durations_all, dim=0)

        MJO = MinJerkOpt(head_pva_all, tail_pva_all, self.cfg.num_pieces)
        start = time.perf_counter()
        MJO.generate(inner_pts_all, durations_all)
        end = time.perf_counter()
        logger.trace(f"Local trajectories generation takes {end - start:.5f}s")

        traj_all = MJO.get_traj()
        self.trajs = {agent: traj_all[i * self.num_envs : (i + 1) * self.num_envs] for i, agent in enumerate(self.possible_agents)}
        self.execution_time = torch.zeros(self.num_envs, device=self.device)
        self.has_prev_traj.fill_(True)

        self.p_odom_for_vis = {agent: self.robots[agent].data.root_state_w[:, :3].clone() for agent in self.possible_agents}
        self.visualize_new_cmd = True

    def _apply_action(self) -> None:
        if self.control_counter % self.cfg.control_decimation == 0:
            start = time.perf_counter()
            for agent in self.possible_agents:
                self.actions[agent] = torch.cat(
                    (
                        self.trajs[agent].get_pos(self.execution_time),
                        self.trajs[agent].get_vel(self.execution_time),
                        self.trajs[agent].get_acc(self.execution_time),
                        self.trajs[agent].get_jer(self.execution_time),
                        torch.zeros_like(self.execution_time).unsqueeze(1),
                        torch.zeros_like(self.execution_time).unsqueeze(1),
                    ),
                    dim=1,
                )

                self.a_desired_total[agent], self.thrust_desired[agent], self.q_desired[agent], self.w_desired[agent], self.m_desired[agent] = self.controllers[
                    agent
                ].get_control(self.robots[agent].data.root_state_w, self.actions[agent])

                self._thrust_desired[agent] = torch.cat((torch.zeros(self.num_envs, 2, device=self.device), self.thrust_desired[agent].unsqueeze(1)), dim=1)

            end = time.perf_counter()
            logger.trace(f"get_control for all drones takes {end - start:.5f}s")

            self.control_counter = 0
        self.control_counter += 1

        for agent in self.possible_agents:
            self.robots[agent].set_external_force_and_torque(self._thrust_desired[agent].unsqueeze(1), self.m_desired[agent].unsqueeze(1), body_ids=self.body_ids[agent])
        self.execution_time += self.physics_dt

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        # Update relative positions before _get_rewards
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j] = self.robots[agent_j].data.root_pos_w - self.robots[agent_i].data.root_pos_w

        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:
            z_exceed_bounds = torch.logical_or(self.robots[agent].data.root_link_pos_w[:, 2] < 0.5, self.robots[agent].data.root_link_pos_w[:, 2] > 2.0)
            ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robots[agent].data.root_link_quat_w))
            _died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 60.0)

            died = torch.logical_or(died, _died)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return {agent: died for agent in self.cfg.possible_agents}, {agent: time_out for agent in self.cfg.possible_agents}

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards = {}

        mutual_collision_avoidance_reward = torch.zeros(self.num_envs, device=self.device)
        for i in range(self.cfg.num_drones):
            for j in range(i + 1, self.cfg.num_drones):
                dist_btw_drones = torch.linalg.norm(self.relative_positions_w[i][j], dim=1)

                collision_penalty = 1.0 / (1.0 + torch.exp(77 * (dist_btw_drones - self.cfg.safe_dist)))
                mutual_collision_avoidance_reward -= collision_penalty

        for agent in self.possible_agents:
            dist_to_goal = torch.linalg.norm(self.goals - self.robots[agent].data.root_pos_w, dim=1)
            approaching_goal_reward = torch.zeros(self.num_envs, device=self.device)
            if agent in self.prev_dist_to_goal:
                approaching_goal_reward = self.prev_dist_to_goal[agent] - dist_to_goal
            self.prev_dist_to_goal[agent] = dist_to_goal

            dist_to_goal_reward = torch.exp(-self.cfg.dist_to_goal_scale * dist_to_goal)

            tail_wp_dist_to_goal = torch.linalg.norm(self.goals - self.p_tail[agent], dim=1)
            tail_wp_dist_to_goal_reward = torch.exp(-self.cfg.tail_wp_dist_to_goal_scale * tail_wp_dist_to_goal)

            success = dist_to_goal < self.cfg.success_distance_threshold
            unsuccess = ~success
            # Additional reward when the drone is close to goal
            success_reward = torch.where(success, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))
            # Time penalty for not reaching the goal
            time_penalty = torch.where(unsuccess, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))

            # Reward for maintaining speed close to v_max
            v_curr = torch.linalg.norm(self.robots[agent].data.root_lin_vel_w, dim=1)
            speed_maintenance_reward = torch.exp(-((torch.abs(v_curr - self.cfg.v_max[agent]) / self.cfg.speed_deviation_tolerance) ** 2))

            reward = {
                "approaching_goal": approaching_goal_reward * self.cfg.approaching_goal_reward_weight * self.step_dt,
                "dist_to_goal": dist_to_goal_reward * self.cfg.dist_to_goal_reward_weight * self.step_dt,
                "tail_wp_dist_to_goal": tail_wp_dist_to_goal_reward * self.cfg.tail_wp_dist_to_goal_reward_weight * self.step_dt,
                "success": success_reward * self.cfg.success_reward_weight * self.step_dt,
                "time_penalty": -time_penalty * self.cfg.time_penalty_weight * self.step_dt,
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

        self.has_prev_traj[env_ids].fill_(False)

        # Sample new commands
        self.goals[env_ids, :2] = torch.zeros_like(self.goals[env_ids, :2]).uniform_(-self.cfg.goal_range, self.cfg.goal_range)
        self.goals[env_ids, :2] += self.terrain.env_origins[env_ids, :2]
        self.goals[env_ids, 2] = torch.ones_like(self.goals[env_ids, 2]) * self.cfg.flight_altitude
        self.reset_goal_timer[env_ids] = 0.0

        # Reset robot state
        # Randomly permute initial root and joint states among agents
        agents = list(self.possible_agents)
        permuted = agents.copy()
        if self.cfg.rand_init_states:
            random.shuffle(permuted)
        mapping = dict(zip(agents, permuted))
        for agent in self.possible_agents:
            rand_other_agent = mapping[agent]
            joint_pos = self.robots[rand_other_agent].data.default_joint_pos[env_ids]
            joint_vel = self.robots[rand_other_agent].data.default_joint_vel[env_ids]
            default_root_state = self.robots[rand_other_agent].data.default_root_state[env_ids]
            default_root_state[:, :3] += self.terrain.env_origins[env_ids]
            self.robots[agent].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[agent].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[agent].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

            self.controllers[agent].reset(env_ids)

            if agent in self.prev_dist_to_goal:
                self.prev_dist_to_goal[agent][env_ids] = torch.linalg.norm(self.goals[env_ids] - self.robots[agent].data.root_pos_w[env_ids], dim=1)

        # Update relative positions
        for i, agent_i in enumerate(self.possible_agents):
            for j, agent_j in enumerate(self.possible_agents):
                if i == j:
                    continue
                self.relative_positions_w[i][j][env_ids] = self.robots[agent_j].data.root_pos_w[env_ids] - self.robots[agent_i].data.root_pos_w[env_ids]

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.reset_goal_timer += self.step_dt
        reset_goal_idx = self.reset_goal_timer > self.cfg.goal_reset_period
        if reset_goal_idx.any():
            self.goals[reset_goal_idx, :2] = torch.zeros_like(self.goals[reset_goal_idx, :2]).uniform_(-self.cfg.goal_range, self.cfg.goal_range)
            self.goals[reset_goal_idx, :2] += self.terrain.env_origins[reset_goal_idx, :2]
            self.goals[reset_goal_idx, 2] = torch.ones_like(self.goals[reset_goal_idx, 2]) * self.cfg.flight_altitude
            self.reset_goal_timer[reset_goal_idx] = 0.0

        observations = {}
        for i, agent in enumerate(self.possible_agents):
            body2goal_w = self.goals - self.robots[agent].data.root_pos_w

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

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self.cfg.debug_vis_goal:
                if not hasattr(self, "goal_visualizer"):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (0.07, 0.07, 0.07)
                    marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                    marker_cfg.prim_path = "/Visuals/Command/goal"
                    self.goal_visualizer = VisualizationMarkers(marker_cfg)
                    self.goal_visualizer.set_visibility(True)

            if self.cfg.debug_vis_action:
                if not hasattr(self, "waypoint_visualizers"):
                    self.waypoint_visualizers = {}
                    r_min, r_max = 0.01, 0.035
                    color_s, color_e = (1.0, 0.0, 0.0), (0.1, 0.0, 0.0)
                    for i, agent in enumerate(self.possible_agents):
                        self.waypoint_visualizers[agent] = []
                        for j in range(self.cfg.num_pieces):
                            ratio = j / max(self.cfg.num_pieces - 1, 1)
                            r = r_min + (r_max - r_min) * ratio
                            c = (
                                color_s[0] + (color_e[0] - color_s[0]) * ratio,
                                color_s[1] + (color_e[1] - color_s[1]) * ratio,
                                color_s[2] + (color_e[2] - color_s[2]) * ratio,
                            )
                            marker_cfg = VisualizationMarkersCfg(
                                prim_path=f"/Visuals/Command/Robot_{i}/waypoint_{j}",
                                markers={"sphere": sim_utils.SphereCfg(radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c))},
                            )
                            self.waypoint_visualizers[agent].append(VisualizationMarkers(marker_cfg))
                            self.waypoint_visualizers[agent][j].set_visibility(True)

            if self.cfg.debug_vis_traj:
                if not hasattr(self, "traj_visualizers"):
                    self.traj_visualizers = {}
                    self.resolution = 100
                    for i, agent in enumerate(self.possible_agents):
                        self.traj_visualizers[agent] = []
                        for j in range(self.cfg.num_pieces * self.resolution):
                            marker_cfg = VisualizationMarkersCfg(
                                prim_path=f"/Visuals/Command/Robot_{i}//traj_pt_{j}",
                                markers={"sphere": sim_utils.SphereCfg(radius=0.005, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 1.0)))},
                            )
                            self.traj_visualizers[agent].append(VisualizationMarkers(marker_cfg))
                            self.traj_visualizers[agent][j].set_visibility(True)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.visualize(translations=self.goals)

        if self.visualize_new_cmd:
            if hasattr(self, "waypoint_visualizers"):
                for agent in self.possible_agents:
                    for i in range(self.cfg.num_pieces):
                        waypoint_world = self.waypoints[agent][:, 3 * i : 3 * (i + 1)] * self.cfg.p_max[agent] + self.p_odom_for_vis[agent]
                        self.waypoint_visualizers[agent][i].visualize(translations=waypoint_world)

            if hasattr(self, "traj_visualizers"):
                for agent in self.possible_agents:
                    traj_dur = self.trajs[agent].get_total_duration()
                    resolution = traj_dur / (self.cfg.num_pieces * self.resolution)
                    for i in range(self.cfg.num_pieces * self.resolution):
                        traj_pt_world = self.trajs[agent].get_pos(i * resolution)
                        self.traj_visualizers[agent][i].visualize(translations=traj_pt_world)

            self.visualize_new_cmd = False


from config import agents


gym.register(
    id="FAST-Swarm-Waypoint",
    entry_point=SwarmWaypointEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": SwarmWaypointEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:swarm_sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:swarm_skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:swarm_skrl_mappo_cfg.yaml",
    },
)
