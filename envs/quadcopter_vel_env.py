from __future__ import annotations

import gymnasium as gym
import math
import torch

from rclpy.node import Node
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped, Vector3Stamped

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
from utils.controller import Controller


@configclass
class QuadcopterVelEnvCfg(DirectRLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # Reward weights
    to_live_reward_weight = 0.0  # 《活着》
    death_penalty_weight = 0.0
    approaching_goal_reward_weight = 1.0
    dist_to_goal_reward_weight = 0.0
    success_reward_weight = 100.0
    time_penalty_weight = 0.0
    ang_vel_penalty_weight = 0.0
    action_temporal_smoothness_reward_weight = 0.0  # Reward for temporal smoothness of actions

    # Exponential decay factors and tolerances
    dist_to_goal_scale = 0.3
    speed_deviation_tolerance = 0.5
    action_temporal_smoothness_scale = 1.0

    flight_altitude = 1.0  # Desired flight altitude
    success_distance_threshold = 0.5  # Distance threshold for considering goal reached
    goal_reset_period = 10.0  # Time period for resetting goal
    goal_range = 10.0  # Range of xy coordinates of the goal

    # Env
    episode_length_s = 30.0
    physics_freq = 200.0
    control_freq = 100.0
    action_freq = 50.0
    gui_render_freq = 50.0
    control_decimation = physics_freq // control_freq
    decimation = math.ceil(physics_freq / action_freq)  # Environment decimation
    render_decimation = physics_freq // gui_render_freq
    observation_space = 6
    state_space = 0
    action_space = 2
    clip_action = 1.0

    v_max = 2.0

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


class QuadcopterVelEnv(DirectRLEnv):
    cfg: QuadcopterVelEnvCfg

    def __init__(self, cfg: QuadcopterVelEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Action and control decimation must be greater than or equal to 1 #^#")

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
                "success",
                "time_penalty",
                "speed_maintenance",
                "ang_vel_penalty",
                "action_temporal_smoothness",
            ]
        }

        # Get specific indices
        self.body_id = self.robot.find_bodies("body")[0]

        self.robot_mass = self.robot.root_physx_view.get_masses()[0, 0].to(self.device)
        self.robot_inertia = self.robot.root_physx_view.get_inertias()[0, 0].to(self.device)
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)
        self.robot_weight = (self.robot_mass * self.gravity.norm()).item()

        # Controller
        self.controller = Controller(1 / self.cfg.control_freq, self.gravity, self.robot_mass.to(self.device), self.robot_inertia.to(self.device), self.num_envs)
        self.p_desired = torch.zeros(self.num_envs, 3, device=self.device)
        self.v_desired = torch.zeros(self.num_envs, 3, device=self.device)
        self.a_desired = torch.zeros(self.num_envs, 3, device=self.device)
        self.j_desired = torch.zeros(self.num_envs, 3, device=self.device)
        self.yaw_desired = torch.zeros(self.num_envs, 1, device=self.device)
        self.yaw_dot_desired = torch.zeros(self.num_envs, 1, device=self.device)
        self.control_counter = 0

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        self.visualize_new_cmd = False

        # ROS2
        self.node = Node("quadcopter_vel_env", namespace="quadcopter_vel_env")
        self.odom_pub = self.node.create_publisher(Odometry, "odom", 10)
        self.p_desired_pub = self.node.create_publisher(TwistStamped, "p_desired", 10)
        self.v_desired_pub = self.node.create_publisher(TwistStamped, "v_desired", 10)

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
        self.v_xy_desired_normalized = actions.clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action
        self.v_desired[:, :2] = self.v_xy_desired_normalized * self.cfg.v_max
        self.p_desired[:, :2] = self.robot.data.root_pos_w[:, :2] + self.v_desired[:, :2] * self.step_dt
        # self.p_desired[:, :2] += self.v_desired[:, :2] * self.step_dt

    def _apply_action(self):
        if self.control_counter % self.cfg.control_decimation == 0:
            state_desired = torch.cat((self.p_desired, self.v_desired, self.a_desired, self.j_desired, self.yaw_desired, self.yaw_dot_desired), dim=1)

            self.a_desired_total, self.thrust_desired, self.q_desired, self.w_desired, self.m_desired = self.controller.get_control(
                self.robot.data.root_state_w, state_desired
            )

            self._thrust_desired = torch.cat((torch.zeros(self.num_envs, 2, device=self.device), self.thrust_desired.unsqueeze(1)), dim=1)

            self._publish_debug_signals()

            self.control_counter = 0
        self.control_counter += 1

        self.robot.set_external_force_and_torque(self._thrust_desired.unsqueeze(1), self.m_desired.unsqueeze(1), body_ids=self.body_id)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_exceed_bounds = torch.logical_or(self.robot.data.root_pos_w[:, 2] < 0.5, self.robot.data.root_pos_w[:, 2] > 1.5)
        ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robot.data.root_quat_w))
        died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 80.0)

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

        success = dist_to_goal < self.cfg.success_distance_threshold
        unsuccess = ~success
        # Additional reward when the drone is close to goal
        success_reward = torch.where(success, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))
        # Time penalty for not reaching the goal
        time_reward = -torch.where(unsuccess, torch.ones(self.num_envs, device=self.device), torch.zeros(self.num_envs, device=self.device))

        ### ============= Smoothing ============= ###
        ang_vel_reward = -torch.linalg.norm(self.robot.data.root_ang_vel_w, dim=1)

        # Reward for temporal smoothness of actions
        action_temporal_smoothness_reward = torch.zeros(self.num_envs, device=self.device)
        if hasattr(self, "prev_actions"):
            actions_diff = torch.linalg.norm(self.actions - self.prev_actions, dim=1)
            action_temporal_smoothness_reward = torch.exp(-self.cfg.action_temporal_smoothness_scale * actions_diff)
        self.prev_actions = self.actions.clone()

        reward = {
            "meaning_to_live": torch.ones(self.num_envs, device=self.device) * self.cfg.to_live_reward_weight * self.step_dt,
            "death_penalty": death_reward * self.cfg.death_penalty_weight,
            "approaching_goal": approaching_goal_reward * self.cfg.approaching_goal_reward_weight * self.step_dt,
            "dist_to_goal": dist_to_goal_reward * self.cfg.dist_to_goal_reward_weight * self.step_dt,
            "success": success_reward * self.cfg.success_reward_weight * self.step_dt,
            "time_penalty": time_reward * self.cfg.time_penalty_weight * self.step_dt,
            ### ============= Smoothing ============= ###
            "ang_vel_penalty": ang_vel_reward * self.cfg.ang_vel_penalty_weight * self.step_dt,
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

        self.p_desired[env_ids] = self.robot.data.root_pos_w[env_ids].clone()

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
                self.robot.data.root_lin_vel_w.clone(),  # TODO: Try to discard velocity observations to reduce sim2real gap
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
                if not hasattr(self, "p_desired_visualizer"):
                    marker_cfg = VisualizationMarkersCfg(
                        prim_path=f"/Visuals/Command/p_desired",
                        markers={"sphere": sim_utils.SphereCfg(radius=0.03, visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.1, 1.0)))},
                    )
                    self.p_desired_visualizer = VisualizationMarkers(marker_cfg)
                    self.p_desired_visualizer.set_visibility(True)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.visualize(translations=self.goal)

        if hasattr(self, "p_desired_visualizer"):
            self.p_desired_visualizer.visualize(translations=self.p_desired)

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
        p_desired = self.p_desired[env_id].cpu().numpy()
        v_desired = self.v_desired[env_id].cpu().numpy()

        v_desired_msg = TwistStamped()
        v_desired_msg.header.stamp = t
        v_desired_msg.header.frame_id = "world"
        v_desired_msg.twist.linear.x = float(v_desired[0])
        v_desired_msg.twist.linear.y = float(v_desired[1])
        v_desired_msg.twist.linear.z = float(v_desired[2])
        self.v_desired_pub.publish(v_desired_msg)

    def _get_ros_timestamp(self) -> Time:
        sim_time = self._sim_step_counter * self.physics_dt

        stamp = Time()
        stamp.sec = int(sim_time)
        stamp.nanosec = int((sim_time - stamp.sec) * 1e9)

        return stamp


from config import agents


gym.register(
    id="FAST-Quadcopter-Vel",
    entry_point=QuadcopterVelEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterVelEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:quadcopter_skrl_ppo_cfg.yaml",
    },
)
