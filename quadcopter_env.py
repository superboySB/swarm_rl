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
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils import quat_to_ang_between_z_body_and_z_world
from controller import Controller


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
    viewer = ViewerCfg(eye=(-5.0, -5.0, 2.5))

    # Env
    episode_length_s = 60.0
    decimation = 2
    action_space = 14
    observation_space = 13
    state_space = 0
    debug_vis = False

    ui_window_class_type = QuadcopterEnvWindow

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation * 2,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=3, replicate_physics=True)

    # Robot
    robot: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    p_max = 500.0
    v_max = 5.0
    yaw_dot_max = math.pi / 2

    # Reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self.episode_sums = {key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in ["lin_vel", "ang_vel", "distance_to_goal"]}

        # Get specific indices
        self.body_id = self.robot.find_bodies("body")[0]
        self.joint_id = self.robot.find_joints(".*joint")[0]

        self.robot_mass = self.robot.root_physx_view.get_masses()[0, 0]
        self.robot_inertia = self.robot.root_physx_view.get_inertias()[0, 0]
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # Controller
        self.controller = Controller(
            self.step_dt, self.gravity, self.robot_mass.to(self.device), self.robot_inertia.to(self.device), self.num_envs, self.device
        )

        # ROS2
        self.node = Node("quadcopter_env", namespace="quadcopter_env")
        self.odom_pub = self.node.create_publisher(Odometry, "odom", 10)
        self.action_pub = self.node.create_publisher(Odometry, "action", 10)
        self.a_desired_pub = self.node.create_publisher(AccelStamped, "a_desired", 10)
        self.a_desired_total_pub = self.node.create_publisher(AccelStamped, "a_desired_total", 10)
        self.j_desired_pub = self.node.create_publisher(Vector3Stamped, "j_desired", 10)
        self.m_desired_pub = self.node.create_publisher(Vector3Stamped, "m_desired", 10)
        self.yaw_yaw_dot_thrust_desired_pub = self.node.create_publisher(PointStamped, "yaw_yaw_dot_thrust_desired", 10)

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

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

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone().clamp(-1.0, 1.0)
        self.actions[:, :3] *= self.cfg.p_max
        self.actions[:, :3] += self.terrain.env_origins
        self.actions[:, 3:6] *= self.cfg.v_max
        self.actions[:, 12] *= math.pi
        self.actions[:, 13] *= self.cfg.yaw_dot_max

        start = time.perf_counter()
        self.a_desired_total, self.thrust_desired, self.q_desired, self.w_desired, self.m_desired = self.controller.get_control(
            self.robot.data.root_link_state_w, self.actions
        )
        end = time.perf_counter()
        logger.debug(f"get_control takes {end - start:.6f}s")

        self._thrust_desired = torch.cat((torch.zeros(self.num_envs, 2, device=self.device), self.thrust_desired.unsqueeze(1)), dim=1)

        start = time.perf_counter()
        self._publish_debug_signals()
        end = time.perf_counter()
        logger.debug(f"publish_debug_signals takes {end - start:.6f}s")

    def _apply_action(self):
        self.robot.set_external_force_and_torque(self._thrust_desired.unsqueeze(1), self.m_desired.unsqueeze(1), body_ids=self.body_id)

        # TODO: only for visualization 0_0 Not working due to unknown reason :(
        self.robot.set_joint_velocity_target(self.robot.data.default_joint_vel, joint_ids=self.joint_id, env_ids=self.robot._ALL_INDICES)

    def _get_observations(self) -> dict:
        # TODO: only for debug. To be deleted
        diff = self.robot.data.root_link_state_w - self.robot.data.root_com_state_w
        mask = torch.abs(diff) > 1e-5
        for i in range(mask.shape[0]):
            if mask[i].any():
                logger.warning(f"Root link state and COM state of environment[{i}] are slightly not consistent o_o")

        odom = self.robot.data.root_link_state_w.clone()
        odom[:, :3] -= self.terrain.env_origins
        return {"policy": odom}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self.robot.data.root_com_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self.robot.data.root_com_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self.robot.data.root_link_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self.episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_exceed_bounds = torch.logical_or(self.robot.data.root_link_pos_w[:, 2] < -0.1, self.robot.data.root_link_pos_w[:, 2] > 10.0)
        ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robot.data.root_link_quat_w))
        died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 60.0)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(self._desired_pos_w[env_ids] - self.robot.data.root_link_pos_w[env_ids], dim=1).mean()
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
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self.terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # Create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # Set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # Update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)

    def _publish_debug_signals(self):

        t = self._get_ros_timestamp()
        env_id = 0

        # Publish states
        state = self.robot.data.root_link_state_w[env_id]
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
        sim_time = self.common_step_counter * self.step_dt

        stamp = Time()
        stamp.sec = int(sim_time)
        stamp.nanosec = int((sim_time - stamp.sec) * 1e9)

        return stamp


gym.register(
    id="FAST-Quadcopter-Direct-v0",
    entry_point=QuadcopterEnv,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": QuadcopterEnvCfg},
)
