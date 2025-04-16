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
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_inv, quat_rotate
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from envs.quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils.utils import quat_to_ang_between_z_body_and_z_world
from utils.minco import MinJerkOpt
from utils.controller import Controller


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment"""

    def __init__(self, env: QuadcopterEnv, window_name: str = "IsaacLab"):
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
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(-10.0, -10.0, 4.0))

    # Reward weights
    dist_to_goal_reward_weight = 1.0  # Reward for approaching the goal
    tail_wp_dist_to_goal_reward_weight = 0.0  # Reward for tail waypoint to approach the goal
    dist_btw_wps_uniformity_reward_weight = 0.0  # Reward for uniform distances between waypoints
    angle_restriction_reward_weight = 0.0  # Reward for restricting angles between consecutive path segments
    action_temporal_smoothness_reward_weight = 0.0  # Reward for temporal smoothness of actions

    # Env
    episode_length_s = 13.0
    physics_freq = 200
    control_freq = 100
    mpc_freq = 10
    gui_render_freq = 50
    control_decimation = physics_freq // control_freq
    decimation = math.ceil(physics_freq / mpc_freq)  # Environment (replan) decimation
    render_decimation = physics_freq // gui_render_freq
    observation_space = 13
    state_space = 0

    # MINCO trajectory
    num_pieces = 6
    duration = 0.3
    a_max = 10.0
    v_max = 5.0
    p_max = num_pieces * v_max * duration
    action_space = 3 * (num_pieces + 2)  # inner_pts 3 x (num_pieces - 1) + tail_pva 3 x 3
    clip_action = 100  # Default bound for box action spaces in IsaacLab Sb3VecEnvWrapper

    ui_window_class_type = QuadcopterEnvWindow

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)

    # Robot
    robot: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Debug visualization
    debug_vis = True
    debug_vis_goal = True
    debug_vis_action = True


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Replan and control decimation must be greater than or equal to 1 #^#")

        if 1 / self.cfg.mpc_freq > self.cfg.num_pieces * self.cfg.duration:
            raise ValueError("Replan period must be less than or equal to the total trajectory duration #^#")

        # Goal position
        self.desired_position = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self.episode_sums = {key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in ["lin_vel", "ang_vel", "dist_to_goal"]}

        # Get specific indices
        self.body_id = self.robot.find_bodies("body")[0]
        self.joint_id = self.robot.find_joints(".*joint")[0]

        self.robot_mass = self.robot.root_physx_view.get_masses()[0, 0]
        self.robot_inertia = self.robot.root_physx_view.get_inertias()[0, 0]
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # Traj
        self.traj = None
        self.has_prev_traj = torch.tensor([False] * self.num_envs, device=self.device)

        # Controller
        self.controller = Controller(self.step_dt, self.gravity, self.robot_mass.to(self.device), self.robot_inertia.to(self.device))
        self.control_counter = 0

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        self.visualize_new_cmd = False

        # ROS2
        self.node = Node("quadcopter_env", namespace="quadcopter_env")
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
        q_odom = self.robot.data.root_state_w[:, 3:7]
        v_odom = self.robot.data.root_state_w[:, 7:10]
        a_odom = torch.zeros_like(v_odom)
        if self.traj is not None:
            a_odom = torch.where(self.has_prev_traj.unsqueeze(1), self.traj.get_acc(self.execution_time), a_odom)
        head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)

        # Waypoints
        inner_pts = torch.zeros((self.num_envs, 3, self.cfg.num_pieces - 1), device=self.device)
        for i in range(self.cfg.num_pieces - 1):
            # Transform to world frame
            inner_pts[:, :, i] = quat_rotate(q_odom, self.waypoints[:, 3 * i : 3 * (i + 1)] * self.cfg.p_max) + p_odom

        # Tail states, transformed to world frame
        self.p_tail = quat_rotate(q_odom, self.waypoints[:, 3 * (self.cfg.num_pieces - 1) : 3 * (self.cfg.num_pieces + 0)] * self.cfg.p_max) + p_odom
        v_tail = quat_rotate(q_odom, self.waypoints[:, 3 * (self.cfg.num_pieces + 0) : 3 * (self.cfg.num_pieces + 1)] * self.cfg.v_max)
        a_tail = quat_rotate(q_odom, self.waypoints[:, 3 * (self.cfg.num_pieces + 1) : 3 * (self.cfg.num_pieces + 2)] * self.cfg.a_max)
        tail_pva = torch.stack([self.p_tail, v_tail, a_tail], dim=2)

        durations = torch.full((self.num_envs, self.cfg.num_pieces), self.cfg.duration, device=self.device)

        MJO = MinJerkOpt(head_pva, tail_pva, self.cfg.num_pieces)
        start = time.perf_counter()
        MJO.generate(inner_pts, durations)
        end = time.perf_counter()
        logger.debug(f"Local trajectory generation takes {end - start:.6f}s")

        self.traj = MJO.get_traj()
        self.execution_time = torch.zeros(self.num_envs, device=self.device)
        self.has_prev_traj.fill_(True)

        self.p_odom_for_vis = self.robot.data.root_state_w[:, :3].clone()
        self.q_odom_for_vis = self.robot.data.root_state_w[:, 3:7].clone()
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
            logger.debug(f"get_control takes {end - start:.6f}s")

            self._thrust_desired = torch.cat((torch.zeros(self.num_envs, 2, device=self.device), self.thrust_desired.unsqueeze(1)), dim=1)

            start = time.perf_counter()
            self._publish_debug_signals()
            end = time.perf_counter()
            logger.debug(f"publish_debug_signals takes {end - start:.6f}s")

            self.control_counter = 0
        self.control_counter += 1

        self.robot.set_external_force_and_torque(self._thrust_desired.unsqueeze(1), self.m_desired.unsqueeze(1), body_ids=self.body_id)
        self.execution_time += self.physics_dt

        # TODO: Only for visualization 0_0 Not working due to unknown reason :(
        self.robot.set_joint_velocity_target(self.robot.data.default_joint_vel, env_ids=self.robot._ALL_INDICES)

    def _get_observations(self) -> dict:
        goal_in_body_frame = quat_rotate(quat_inv(self.robot.data.root_quat_w), self.desired_position - self.robot.data.root_pos_w)
        obs = torch.cat(
            [
                goal_in_body_frame,
                self.robot.data.root_quat_w.clone(),
                self.robot.data.root_vel_w.clone(),
            ],
            dim=-1,
        )
        return {"policy": obs, "odom": self.robot.data.root_state_w.clone()}

    def _get_rewards(self) -> torch.Tensor:
        # 1. Goal reaching reward
        dist_to_goal = torch.linalg.norm(self.desired_position - self.robot.data.root_pos_w, dim=1)
        dist_to_goal_reward = torch.exp(-1.0 * dist_to_goal)

        tail_wp_dist_to_goal = torch.linalg.norm(self.desired_position - self.p_tail, dim=1)
        tail_wp_dist_to_goal_reward = torch.exp(-1.0 * tail_wp_dist_to_goal)

        # 2. Waypoint distance uniformity reward
        wps = [torch.zeros(self.num_envs, 3, device=self.device)]
        for i in range(self.cfg.num_pieces):
            wps.append(self.waypoints[:, 3 * i : 3 * (i + 1)] * self.cfg.p_max)

        # Calculate coefficient of variation (CV) of distances
        if self.cfg.num_pieces > 1:
            dist_btw_wps = []
            for i in range(self.cfg.num_pieces):
                dist_btw_wps.append(torch.linalg.norm(wps[i + 1] - wps[i], dim=1))

            dist_btw_wps = torch.stack(dist_btw_wps, dim=1)
            dist_mean = torch.mean(dist_btw_wps, dim=1)
            dist_std = torch.std(dist_btw_wps, dim=1)
            dist_cv = torch.where(dist_mean > 1.0, dist_std / dist_mean, dist_std)
            dist_btw_wps_uniformity_reward = torch.exp(-1.0 * dist_cv)
        else:
            dist_btw_wps_uniformity_reward = torch.zeros(self.num_envs, device=self.device)

        # 3. Reward for restricting angles between consecutive path segments
        if self.cfg.num_pieces > 1:
            cos_angs = []
            for i in range(self.cfg.num_pieces - 1):
                vec1 = wps[i + 1] - wps[i]
                vec2 = wps[i + 2] - wps[i + 1]

                vec1_norm = torch.linalg.norm(vec1, dim=1)
                vec2_norm = torch.linalg.norm(vec2, dim=1)

                dot_product = torch.sum(vec1 * vec2, dim=1)
                cos_ang = torch.where((vec1_norm > 1e-6) & (vec2_norm > 1e-6), dot_product / (vec1_norm * vec2_norm), torch.ones_like(dot_product))
                cos_angs.append(cos_ang)

            cos_angs = torch.stack(cos_angs, dim=1)
            angle_restriction_reward = torch.mean(torch.exp(-5.0 * (1 - cos_angs) / 2), dim=1)
        else:
            angle_restriction_reward = torch.zeros(self.num_envs, device=self.device)

        # 4. Reward for temporal smoothness of actions
        action_temporal_smoothness_reward = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self, "prev_waypoints"):
            waypoint_diff = torch.linalg.norm(self.waypoints - self.prev_waypoints, dim=1)
            action_temporal_smoothness_reward = torch.exp(-0.5 * waypoint_diff)
        self.prev_waypoints = self.waypoints.clone()

        reward = {
            "dist_to_goal": dist_to_goal_reward * self.cfg.dist_to_goal_reward_weight * self.step_dt,
            "tail_wp_dist_to_goal": tail_wp_dist_to_goal_reward * self.cfg.tail_wp_dist_to_goal_reward_weight * self.step_dt,
            "dist_btw_wps_uniformity": dist_btw_wps_uniformity_reward * self.cfg.dist_btw_wps_uniformity_reward_weight * self.step_dt,
            "angle_restriction": angle_restriction_reward * self.cfg.angle_restriction_reward_weight * self.step_dt,
            "action_temporal_smoothness": action_temporal_smoothness_reward * self.cfg.action_temporal_smoothness_reward_weight * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(reward.values())), dim=0)

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_exceed_bounds = torch.logical_or(self.robot.data.root_pos_w[:, 2] < 0.5, self.robot.data.root_pos_w[:, 2] > 10.0)
        ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robot.data.root_quat_w))
        died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 60.0)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # Logging
        final_dist_to_goal = torch.linalg.norm(self.desired_position[env_ids] - self.robot.data.root_pos_w[env_ids], dim=1).mean()
        extras = dict()
        for key in self.episode_sums.keys():
            episodic_sum_avg = torch.mean(self.episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_dist_to_goal"] = final_dist_to_goal.item()
        self.extras["log"].update(extras)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.has_prev_traj[env_ids].fill_(False)

        # Sample new commands
        self.desired_position[env_ids, :2] = torch.zeros_like(self.desired_position[env_ids, :2]).uniform_(0.0, 10.0)
        self.desired_position[env_ids, :2] += self.terrain.env_origins[env_ids, :2]
        self.desired_position[env_ids, 2] = torch.zeros_like(self.desired_position[env_ids, 2]).uniform_(1.0, 1.3)

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self.cfg.debug_vis_goal:
                if not hasattr(self, "goal_pos_visualizer"):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (0.07, 0.07, 0.07)
                    marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))
                    marker_cfg.prim_path = "/Visuals/Command/goal"
                    self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
                self.goal_pos_visualizer.set_visibility(True)

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

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_pos_visualizer"):
            self.goal_pos_visualizer.visualize(translations=self.desired_position)

        if self.visualize_new_cmd and hasattr(self, "waypoint_visualizers"):
            for i in range(self.cfg.num_pieces):
                waypoint_world = quat_rotate(self.q_odom_for_vis, self.waypoints[:, 3 * i : 3 * (i + 1)] * self.cfg.p_max) + self.p_odom_for_vis
                self.waypoint_visualizers[i].visualize(translations=waypoint_world)
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
    id="FAST-Quadcopter-Direct-v0",
    entry_point=QuadcopterEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_sb3_ppo_cfg.yaml",
    },
)
