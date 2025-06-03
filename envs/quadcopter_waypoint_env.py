from __future__ import annotations

import gymnasium as gym
from loguru import logger
import math
import time
import torch

from rclpy.node import Node
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import AccelStamped, Vector3Stamped, PointStamped

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
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
class QuadcopterWaypointEnvCfg(DirectRLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # Reward weights
    to_live_reward_weight = 0.0  # 《活着》
    death_penalty_weight = 0.0
    approaching_goal_reward_weight = 1.0
    dist_to_goal_reward_weight = 0.0
    tail_wp_dist_to_goal_reward_weight = 0.0
    success_reward_weight = 100.0
    time_penalty_weight = 0.0
    speed_maintenance_reward_weight = 0.0  # Reward for maintaining speed close to v_max
    wps_z_deviation_reward_weight = 0.0  # Penalty for z-deviation from desired flight altitude
    ang_vel_penalty_weight = 0.0
    dist_btw_wps_uniformity_reward_weight = 0.0  # Reward for uniform distances between waypoints
    angle_restriction_reward_weight = 0.0  # Reward for restricting angles between consecutive path segments
    action_temporal_smoothness_reward_weight = 0.0  # Reward for temporal smoothness of actions

    # Exponential decay factors and tolerances
    dist_to_goal_scale = 0.3
    tail_wp_dist_to_goal_scale = 0.3
    speed_deviation_tolerance = 0.5
    wps_z_deviation_scale = 50.0
    dist_btw_wps_uniformity_scale = 10.0
    angle_restriction_scale = 10.0
    action_temporal_smoothness_scale = 1.0

    flight_altitude = 1.0  # Desired flight altitude
    success_distance_threshold = 0.5  # Distance threshold for considering goal reached
    goal_reset_period = 10.0  # Time period for resetting goal
    goal_range = 10.0  # Range of xy coordinates of the goal

    # Env
    episode_length_s = 30.0
    physics_freq = 200.0
    control_freq = 100.0
    mpc_freq = 10.0
    gui_render_freq = 50.0
    control_decimation = physics_freq // control_freq
    decimation = math.ceil(physics_freq / mpc_freq)  # Environment (replan) decimation
    render_decimation = physics_freq // gui_render_freq
    observation_space = 16
    state_space = 0

    # MINCO trajectory
    num_pieces = 1
    duration = 0.5
    a_max = 10.0
    v_max = 3.0
    p_max = 1.0
    action_space = 3 * (num_pieces + 2)  # inner_pts 3 x (num_pieces - 1) + tail_pva 3 x 3
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1000, env_spacing=5, replicate_physics=True)

    # Robot
    robot: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Debug visualization
    debug_vis = True
    debug_vis_goal = True
    debug_vis_action = True
    debug_vis_traj = True


class QuadcopterWaypointEnv(DirectRLEnv):
    cfg: QuadcopterWaypointEnvCfg

    def __init__(self, cfg: QuadcopterWaypointEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Replan and control decimation must be greater than or equal to 1 #^#")

        if 1 / self.cfg.mpc_freq > self.cfg.num_pieces * self.cfg.duration:
            raise ValueError("Replan period must be less than or equal to the total trajectory duration #^#")

        # Goal position
        self.goal = torch.zeros(self.num_envs, 3, device=self.device)
        self.reset_goal_timer = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Logging
        self.episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "meaning_to_live",
                "death_penalty",
                "approaching_goal",
                "dist_to_goal",
                "tail_wp_dist_to_goal",
                "success",
                "time_penalty",
                "speed_maintenance",
                "wps_z_deviation",
                "ang_vel_penalty",
                "dist_btw_wps_uniformity",
                "angle_restriction",
                "action_temporal_smoothness",
            ]
        }

        # Get specific indices
        self.body_id = self.robot.find_bodies("body")[0]

        self.robot_mass = self.robot.root_physx_view.get_masses()[0, 0]
        self.robot_inertia = self.robot.root_physx_view.get_inertias()[0, 0]
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # Traj
        self.traj = None
        self.has_prev_traj = torch.tensor([False] * self.num_envs, device=self.device)

        # Controller
        self.controller = Controller(1 / self.cfg.control_freq, self.gravity, self.robot_mass.to(self.device), self.robot_inertia.to(self.device), self.num_envs)
        self.control_counter = 0

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        self.visualize_new_cmd = False

        # ROS2
        self.node = Node("quadcopter_waypoint_env", namespace="quadcopter_waypoint_env")
        self.odom_pub = self.node.create_publisher(Odometry, "odom", 10)
        self.action_pub = self.node.create_publisher(Odometry, "action", 10)
        self.a_desired_pub = self.node.create_publisher(AccelStamped, "a_desired", 10)
        self.a_desired_total_pub = self.node.create_publisher(AccelStamped, "a_desired_total", 10)
        self.j_desired_pub = self.node.create_publisher(Vector3Stamped, "j_desired", 10)
        self.m_desired_pub = self.node.create_publisher(Vector3Stamped, "m_desired", 10)
        self.yaw_yaw_dot_thrust_desired_pub = self.node.create_publisher(PointStamped, "yaw_yaw_dot_thrust_desired", 10)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # Clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)

        # Add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        # Action parametrization: waypoints in body frame
        self.waypoints = actions.clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action

        for i in range(self.cfg.num_pieces - 1):
            self.waypoints[:, 3 * (i + 1) : 3 * (i + 2)] += self.waypoints[:, 3 * i : 3 * (i + 1)]

        # Head states
        p_odom = self.robot.data.root_state_w[:, :3]
        v_odom = self.robot.data.root_state_w[:, 7:10]
        a_odom = torch.zeros_like(v_odom)
        if self.traj is not None:
            a_odom = torch.where(self.has_prev_traj.unsqueeze(1), self.traj.get_acc(self.execution_time), a_odom)
        head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)

        # Waypoints
        self.wps_z_deviation = []
        inner_pts = torch.zeros((self.num_envs, 3, self.cfg.num_pieces - 1), device=self.device)
        for i in range(self.cfg.num_pieces - 1):
            # Transform to world frame
            inner_pts[:, :, i] = self.waypoints[:, 3 * i : 3 * (i + 1)] * self.cfg.p_max + p_odom
            self.wps_z_deviation.append(torch.abs(inner_pts[:, 2, i] - self.cfg.flight_altitude))

        # Tail states, transformed to world frame
        self.p_tail = self.waypoints[:, 3 * (self.cfg.num_pieces - 1) : 3 * (self.cfg.num_pieces + 0)] * self.cfg.p_max + p_odom
        v_tail = self.waypoints[:, 3 * (self.cfg.num_pieces + 0) : 3 * (self.cfg.num_pieces + 1)] * self.cfg.v_max
        a_tail = self.waypoints[:, 3 * (self.cfg.num_pieces + 1) : 3 * (self.cfg.num_pieces + 2)] * self.cfg.a_max
        tail_pva = torch.stack([self.p_tail, v_tail, a_tail], dim=2)
        self.wps_z_deviation.append(torch.abs(self.p_tail[:, 2] - self.cfg.flight_altitude))

        durations = torch.full((self.num_envs, self.cfg.num_pieces), self.cfg.duration, device=self.device)

        MJO = MinJerkOpt(head_pva, tail_pva, self.cfg.num_pieces)
        start = time.perf_counter()
        MJO.generate(inner_pts, durations)
        end = time.perf_counter()
        logger.trace(f"Local trajectory generation takes {end - start:.5f}s")

        self.traj = MJO.get_traj()
        self.execution_time = torch.zeros(self.num_envs, device=self.device)
        self.has_prev_traj.fill_(True)

        self.p_odom_for_vis = self.robot.data.root_state_w[:, :3].clone()
        self.visualize_new_cmd = True

    def _apply_action(self):
        if self.control_counter % self.cfg.control_decimation == 0:
            self.actions = torch.cat(
                (
                    self.traj.get_pos(self.execution_time),
                    self.traj.get_vel(self.execution_time),
                    self.traj.get_acc(self.execution_time),
                    self.traj.get_jer(self.execution_time),
                    torch.zeros_like(self.execution_time).unsqueeze(1),
                    torch.zeros_like(self.execution_time).unsqueeze(1),
                ),
                dim=1,
            )

            start = time.perf_counter()
            self.a_desired_total, self.thrust_desired, self.q_desired, self.w_desired, self.m_desired = self.controller.get_control(
                self.robot.data.root_state_w, self.actions
            )
            end = time.perf_counter()
            logger.trace(f"get_control takes {end - start:.5f}s")

            self._thrust_desired = torch.cat((torch.zeros(self.num_envs, 2, device=self.device), self.thrust_desired.unsqueeze(1)), dim=1)

            start = time.perf_counter()
            self._publish_debug_signals()
            end = time.perf_counter()
            logger.trace(f"publish_debug_signals takes {end - start:.5f}s")

            self.control_counter = 0
        self.control_counter += 1

        self.robot.set_external_force_and_torque(self._thrust_desired.unsqueeze(1), self.m_desired.unsqueeze(1), body_ids=self.body_id)
        self.execution_time += self.physics_dt

        # TODO: Only for visualization 0_0 Not working due to unknown reason :(
        self.robot.set_joint_velocity_target(self.robot.data.default_joint_vel, env_ids=self.robot._ALL_INDICES)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_exceed_bounds = torch.logical_or(self.robot.data.root_pos_w[:, 2] < 0.5, self.robot.data.root_pos_w[:, 2] > 2.0)
        ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robot.data.root_quat_w))
        died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 60.0)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        died, _ = self._get_dones()
        death_reward = -torch.where(died, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))

        # Goal reaching reward
        dist_to_goal = torch.linalg.norm(self.goal - self.robot.data.root_pos_w, dim=1)
        approaching_goal_reward = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self, "prev_dist_to_goal"):
            approaching_goal_reward = self.prev_dist_to_goal - dist_to_goal
        self.prev_dist_to_goal = dist_to_goal

        dist_to_goal_reward = torch.exp(-self.cfg.dist_to_goal_scale * dist_to_goal)

        tail_wp_dist_to_goal = torch.linalg.norm(self.goal - self.p_tail, dim=1)
        tail_wp_dist_to_goal_reward = torch.exp(-self.cfg.tail_wp_dist_to_goal_scale * tail_wp_dist_to_goal)

        success = dist_to_goal < self.cfg.success_distance_threshold
        unsuccess = ~success
        # Additional reward when the drone is close to goal
        success_reward = torch.where(success, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))
        # Time penalty for not reaching the goal
        time_reward = -torch.where(unsuccess, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))

        # Reward for maintaining speed close to v_max
        v_curr = torch.linalg.norm(self.robot.data.root_lin_vel_w, dim=1)
        speed_maintenance_reward = torch.exp(-((torch.abs(v_curr - self.cfg.v_max) / self.cfg.speed_deviation_tolerance) ** 2))

        # Reward for uniformity of distances between waypoints
        wps = [torch.zeros(self.num_envs, 3, device=self.device)]
        for i in range(self.cfg.num_pieces):
            wps.append(self.waypoints[:, 3 * i : 3 * (i + 1)] * self.cfg.p_max)

        wps_z_deviation_mean = torch.mean(torch.stack(self.wps_z_deviation, dim=1), dim=1)
        # wps_z_deviation_reward = torch.exp(-self.cfg.wps_z_deviation_scale * wps_z_deviation_mean)
        wps_z_deviation_reward = -wps_z_deviation_mean

        ## ============= Smoothing ============= ##
        ang_vel_reward = -torch.linalg.norm(self.robot.data.root_ang_vel_w, dim=1)

        # Calculate coefficient of variation (CV) of distances
        if self.cfg.num_pieces > 1:
            dist_btw_wps = []
            for i in range(self.cfg.num_pieces):
                dist_btw_wps.append(torch.linalg.norm(wps[i + 1] - wps[i], dim=1))

            dist_btw_wps = torch.stack(dist_btw_wps, dim=1)
            dist_mean = torch.mean(dist_btw_wps, dim=1)
            dist_std = torch.std(dist_btw_wps, dim=1)
            dist_cv = torch.where(dist_mean > 1.0, dist_std / dist_mean, dist_std)
            # dist_btw_wps_uniformity_reward = torch.exp(-self.cfg.dist_btw_wps_uniformity_scale * dist_cv)
            dist_btw_wps_uniformity_reward = -dist_cv
        else:
            dist_btw_wps_uniformity_reward = torch.zeros(self.num_envs, device=self.device)

        # Reward for restricting angles between consecutive path segments
        if self.cfg.num_pieces > 1:
            angles = []
            for i in range(self.cfg.num_pieces - 1):
                vec1 = wps[i + 1] - wps[i]
                vec2 = wps[i + 2] - wps[i + 1]

                vec1_norm = torch.linalg.norm(vec1, dim=1)
                vec2_norm = torch.linalg.norm(vec2, dim=1)

                dot_product = torch.sum(vec1 * vec2, dim=1)
                cos_ang = torch.where((vec1_norm > 1e-6) & (vec2_norm > 1e-6), dot_product / (vec1_norm * vec2_norm), torch.ones_like(dot_product))
                angles.append(torch.acos(torch.clip(cos_ang, -1.0, 1.0)))

            angles = torch.stack(angles, dim=1)
            # angle_restriction_reward = torch.exp(-self.cfg.angle_restriction_scale * torch.mean(angles / math.pi, dim=1))
            angle_restriction_reward = -torch.mean(angles / math.pi, dim=1)
        else:
            angle_restriction_reward = torch.zeros(self.num_envs, device=self.device)

        # Reward for temporal smoothness of actions
        action_temporal_smoothness_reward = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self, "prev_waypoints"):
            waypoint_diff = torch.linalg.norm(self.waypoints - self.prev_waypoints, dim=1)
            action_temporal_smoothness_reward = torch.exp(-self.cfg.action_temporal_smoothness_scale * waypoint_diff)
        self.prev_waypoints = self.waypoints.clone()

        reward = {
            "meaning_to_live": torch.ones(self.num_envs, device=self.device) * self.cfg.to_live_reward_weight * self.step_dt,
            "death_penalty": death_reward * self.cfg.death_penalty_weight,
            "approaching_goal": approaching_goal_reward * self.cfg.approaching_goal_reward_weight * self.step_dt,
            "dist_to_goal": dist_to_goal_reward * self.cfg.dist_to_goal_reward_weight * self.step_dt,
            "tail_wp_dist_to_goal": tail_wp_dist_to_goal_reward * self.cfg.tail_wp_dist_to_goal_reward_weight * self.step_dt,
            "success": success_reward * self.cfg.success_reward_weight * self.step_dt,
            "time_penalty": time_reward * self.cfg.time_penalty_weight * self.step_dt,
            "speed_maintenance": speed_maintenance_reward * self.cfg.speed_maintenance_reward_weight * self.step_dt,
            "wps_z_deviation": wps_z_deviation_reward * self.cfg.wps_z_deviation_reward_weight * self.step_dt,
            ## ============= Smoothing ============= ##
            "ang_vel_penalty": ang_vel_reward * self.cfg.ang_vel_penalty_weight * self.step_dt,
            "dist_btw_wps_uniformity": dist_btw_wps_uniformity_reward * self.cfg.dist_btw_wps_uniformity_reward_weight * self.step_dt,
            "angle_restriction": angle_restriction_reward * self.cfg.angle_restriction_reward_weight * self.step_dt,
            "action_temporal_smoothness": action_temporal_smoothness_reward * self.cfg.action_temporal_smoothness_reward_weight * self.step_dt,
        }

        # Logging
        for key, value in reward.items():
            self.episode_sums[key] += value

        reward = torch.sum(torch.stack(list(reward.values())), dim=0)

        return reward

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # Logging
        extras = dict()
        for key in self.episode_sums.keys():
            episodic_sum_avg = torch.mean(self.episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg
            self.episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        self.extras["log"].update(extras)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.num_envs > 13 and len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.has_prev_traj[env_ids].fill_(False)

        # Sample new commands
        self.goal[env_ids, :2] = torch.zeros_like(self.goal[env_ids, :2]).uniform_(-self.cfg.goal_range, self.cfg.goal_range)
        self.goal[env_ids, :2] += self.terrain.env_origins[env_ids, :2]
        self.goal[env_ids, 2] = torch.ones_like(self.goal[env_ids, 2]) * self.cfg.flight_altitude
        self.reset_goal_timer[env_ids] = 0.0

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.controller.reset(env_ids)

        if hasattr(self, "prev_dist_to_goal"):
            self.prev_dist_to_goal[env_ids] = torch.linalg.norm(self.goal[env_ids] - self.robot.data.root_pos_w[env_ids], dim=1)

    def _get_observations(self) -> dict:
        self.reset_goal_timer += self.step_dt
        reset_goal_idx = self.reset_goal_timer > self.cfg.goal_reset_period
        if reset_goal_idx.any():
            self.goal[reset_goal_idx, :2] = torch.zeros_like(self.goal[reset_goal_idx, :2]).uniform_(-self.cfg.goal_range, self.cfg.goal_range)
            self.goal[reset_goal_idx, :2] += self.terrain.env_origins[reset_goal_idx, :2]
            self.goal[reset_goal_idx, 2] = torch.ones_like(self.goal[reset_goal_idx, 2]) * self.cfg.flight_altitude
            self.reset_goal_timer[reset_goal_idx] = 0.0

        body2goal_w = self.goal - self.robot.data.root_pos_w
        obs = torch.cat(
            [
                body2goal_w,
                self.robot.data.root_quat_w.clone(),
                self.robot.data.projected_gravity_b.clone(),
                self.robot.data.root_vel_w.clone(),  # TODO: Try to discard velocity observations to reduce sim2real gap
            ],
            dim=-1,
        )

        return {"policy": obs, "odom": self.robot.data.root_state_w.clone()}

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
                    self.waypoint_visualizers = []
                    r_min, r_max = 0.01, 0.035
                    color_s, color_e = (1.0, 0.0, 0.0), (0.1, 0.0, 0.0)
                    for i in range(self.cfg.num_pieces):
                        ratio = i / max(self.cfg.num_pieces - 1, 1)
                        r = r_min + (r_max - r_min) * ratio
                        c = (
                            color_s[0] + (color_e[0] - color_s[0]) * ratio,
                            color_s[1] + (color_e[1] - color_s[1]) * ratio,
                            color_s[2] + (color_e[2] - color_s[2]) * ratio,
                        )
                        marker_cfg = VisualizationMarkersCfg(
                            prim_path=f"/Visuals/Command/waypoint_{i}",
                            markers={"sphere": sim_utils.SphereCfg(radius=r, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=c))},
                        )
                        self.waypoint_visualizers.append(VisualizationMarkers(marker_cfg))
                        self.waypoint_visualizers[i].set_visibility(True)

            if self.cfg.debug_vis_traj:
                if not hasattr(self, "traj_visualizers"):
                    self.traj_visualizers = []
                    self.resolution = 100
                    for i in range(self.cfg.num_pieces * self.resolution):
                        marker_cfg = VisualizationMarkersCfg(
                            prim_path=f"/Visuals/Command/traj_pt_{i}",
                            markers={"sphere": sim_utils.SphereCfg(radius=0.005, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 1.0)))},
                        )
                        self.traj_visualizers.append(VisualizationMarkers(marker_cfg))
                        self.traj_visualizers[i].set_visibility(True)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.visualize(translations=self.goal)

        if self.visualize_new_cmd:
            if hasattr(self, "waypoint_visualizers"):
                for i in range(self.cfg.num_pieces):
                    waypoint_world = self.waypoints[:, 3 * i : 3 * (i + 1)] * self.cfg.p_max + self.p_odom_for_vis
                    self.waypoint_visualizers[i].visualize(translations=waypoint_world)

            if hasattr(self, "traj_visualizers"):
                traj_dur = self.traj.get_total_duration()
                resolution = traj_dur / (self.cfg.num_pieces * self.resolution)
                for i in range(self.cfg.num_pieces * self.resolution):
                    traj_pt_world = self.traj.get_pos(i * resolution)
                    self.traj_visualizers[i].visualize(translations=traj_pt_world)

            self.visualize_new_cmd = False

    def _publish_debug_signals(self):

        t = self._get_ros_timestamp()
        env_id = 0

        # Publish states
        state = self.robot.data.root_state_w[env_id]
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
        action = self.actions[env_id]
        p_desired = action[:3].cpu().numpy()
        v_desired = action[3:6].cpu().numpy()
        a_desired = action[6:9].cpu().numpy()
        j_desired = action[9:12].cpu().numpy()
        yaw_desired = action[12].cpu().numpy()
        yaw_dot_desired = action[13].cpu().numpy()

        a_desired_total = self.a_desired_total[env_id].cpu().numpy()
        thrust_desired = self.thrust_desired[env_id].cpu().numpy()
        q_desired = self.q_desired[env_id].cpu().numpy()
        w_desired = self.w_desired[env_id].cpu().numpy()
        m_desired = self.m_desired[env_id].cpu().numpy()

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

        j_desired_msg = Vector3Stamped()
        j_desired_msg.header.stamp = t
        j_desired_msg.vector.x = float(j_desired[0])
        j_desired_msg.vector.y = float(j_desired[1])
        j_desired_msg.vector.z = float(j_desired[2])
        self.j_desired_pub.publish(j_desired_msg)

        m_desired_msg = Vector3Stamped()
        m_desired_msg.header.stamp = t
        m_desired_msg.vector.x = float(m_desired[0])
        m_desired_msg.vector.y = float(m_desired[1])
        m_desired_msg.vector.z = float(m_desired[2])
        self.m_desired_pub.publish(m_desired_msg)

        yaw_yaw_dot_thrust_desired_msg = PointStamped()
        yaw_yaw_dot_thrust_desired_msg.header.stamp = t
        yaw_yaw_dot_thrust_desired_msg.point.x = float(yaw_desired)
        yaw_yaw_dot_thrust_desired_msg.point.y = float(yaw_dot_desired)
        yaw_yaw_dot_thrust_desired_msg.point.z = float(thrust_desired)
        self.yaw_yaw_dot_thrust_desired_pub.publish(yaw_yaw_dot_thrust_desired_msg)

    def _get_ros_timestamp(self) -> Time:
        sim_time = self._sim_step_counter * self.physics_dt

        stamp = Time()
        stamp.sec = int(sim_time)
        stamp.nanosec = int((sim_time - stamp.sec) * 1e9)

        return stamp


from config import agents


gym.register(
    id="FAST-Quadcopter-Waypoint",
    entry_point=QuadcopterWaypointEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterWaypointEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:quadcopter_skrl_ppo_cfg.yaml",
    },
)
