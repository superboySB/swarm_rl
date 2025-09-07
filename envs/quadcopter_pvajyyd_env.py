from __future__ import annotations

import gymnasium as gym
from loguru import logger
import math
import time
import torch
from collections import deque

from rclpy.node import Node
from builtin_interfaces.msg import Time
from nav_msgs.msg import Odometry
from geometry_msgs.msg import AccelStamped, Vector3Stamped, PointStamped

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_inv, quat_apply

from envs.quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils.utils import quat_to_ang_between_z_body_and_z_world
from utils.controller import Controller


@configclass
class QuadcopterPVAJYYdEnvCfg(DirectRLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(3.0, -3.0, 30.0))

    # Env
    episode_length_s = 30.0
    physics_freq = 200.0
    control_freq = 50.0
    action_freq = 50.0
    gui_render_freq = 50.0
    control_decimation = physics_freq // control_freq
    decimation = math.ceil(physics_freq / action_freq)  # Environment (replan) decimation
    render_decimation = physics_freq // gui_render_freq
    observation_space = 16
    state_space = 0
    action_space = 14

    action_delay_s = 0.0

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


class QuadcopterPVAJYYdEnv(DirectRLEnv):
    cfg: QuadcopterPVAJYYdEnvCfg

    def __init__(self, cfg: QuadcopterPVAJYYdEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        delay_steps = math.ceil(cfg.action_delay_s / self.physics_dt)
        self.delay_steps = max(delay_steps, 1)
        self.thrust_buffer = deque([torch.zeros(self.num_envs, 3, device=self.device) for _ in range(self.delay_steps)])
        self.m_buffer = deque([torch.zeros(self.num_envs, 3, device=self.device) for _ in range(self.delay_steps)])

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Replan and control decimation must be greater than or equal to 1 #^#")

        # Get specific indices
        self.body_id = self.robot.find_bodies("body")[0]

        self.robot_mass = self.robot.root_physx_view.get_masses()[0, 0]
        self.robot_inertia = self.robot.root_physx_view.get_inertias()[0, 0]
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # Controller
        self.controller = Controller(1 / self.cfg.control_freq, self.gravity, self.robot_mass.to(self.device), self.robot_inertia.to(self.device), self.num_envs)
        self.control_counter = 0

        # ROS2
        self.node = Node("quadcopter_pvajyyd_env", namespace="quadcopter_pvajyyd_env")
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
        self.actions = actions.clone()

    def _apply_action(self):
        if self.control_counter % self.cfg.control_decimation == 0:
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

        delayed_thrust = self.thrust_buffer.popleft()
        delayed_m = self.m_buffer.popleft()
        self.thrust_buffer.append(self._thrust_desired.clone())
        self.m_buffer.append(self.m_desired.clone())

        self.robot.set_external_force_and_torque(delayed_thrust.unsqueeze(1), delayed_m.unsqueeze(1), body_ids=self.body_id)

        # TODO: Only for visualization 0_0 Not working due to unknown reason :(
        self.robot.set_joint_velocity_target(self.robot.data.default_joint_vel, env_ids=self.robot._ALL_INDICES)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        z_exceed_bounds = torch.logical_or(self.robot.data.root_pos_w[:, 2] < 0.5, self.robot.data.root_pos_w[:, 2] > 1.5)
        ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robot.data.root_quat_w))
        died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 80.0)

        time_out = self.episode_length_buf >= self.max_episode_length - 1

        return died, time_out

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robot._ALL_INDICES

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if self.num_envs > 13 and len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        # Reset robot state
        joint_pos = self.robot.data.default_joint_pos[env_ids]
        joint_vel = self.robot.data.default_joint_vel[env_ids]
        default_root_state = self.robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self.terrain.env_origins[env_ids]
        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.controller.reset(env_ids)

    def _get_observations(self) -> dict:
        return {"odom": self.robot.data.root_state_w.clone()}

    def _publish_debug_signals(self):

        t = self._get_ros_timestamp()
        env_id = 0

        # Publish states
        state = self.robot.data.root_state_w[env_id]
        p_odom = state[:3].cpu().numpy()
        q_odom = state[3:7].cpu().numpy()
        v_odom = state[7:10].cpu().numpy()
        w_odom_w = state[10:13]
        w_odom_b = quat_apply(quat_inv(self.robot.data.root_quat_w[env_id]), w_odom_w)
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
    id="FAST-Quadcopter-PVAJYYd",
    entry_point=QuadcopterPVAJYYdEnv,
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": QuadcopterPVAJYYdEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_sb3_ppo_cfg.yaml",
        "skrl_cfg_entry_point": f"{agents.__name__}:quadcopter_skrl_ppo_cfg.yaml",
    },
)
