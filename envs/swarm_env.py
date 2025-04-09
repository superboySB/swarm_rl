from __future__ import annotations

import gymnasium as gym
from loguru import logger
import math
import time
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, ViewerCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_rotate, quat_inv
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from envs.quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils.utils import quat_to_ang_between_z_body_and_z_world
from utils.minco import MinJerkOpt
from utils.controller import Controller


class RewardLogger:
    """记录奖励组件的工具类"""
    
    @staticmethod
    def format_tensor_for_tensorboard(tensor, num_envs, device):
        """格式化张量以用于tensorboard记录"""
        if isinstance(tensor, torch.Tensor):
            return torch.full((num_envs,), tensor.mean().item(), device=device)
        else:
            return torch.full((num_envs,), tensor, device=device)
    
    @staticmethod
    def log_reward_components(extras, rewards_dict, num_envs, device, prefix="CurrentStep"):
        """记录当前步骤的奖励组件"""
        if "log" not in extras:
            extras["log"] = {}
            
        for key, value in rewards_dict.items():
            extras["log"][f"{prefix}/{key}"] = RewardLogger.format_tensor_for_tensorboard(value, num_envs, device)
        
        return extras


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment"""

    def __init__(self, env: QuadcopterSwarmEnv, window_name: str = "IsaacLab"):
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
class QuadcopterSwarmEnvCfg(DirectMARLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(-5.0, -5.0, 4.0))

    # Env
    episode_length_s = 30.0
    physics_freq = 200
    control_freq = 100
    mpc_freq = 10
    gui_render_freq = 50
    control_decimation = physics_freq // control_freq
    num_drones = 5  # Number of drones per environment
    decimation = math.ceil(physics_freq / mpc_freq)  # Environment (replan) decimation
    render_decimation = physics_freq // gui_render_freq
    possible_agents = [f"drone_{i}" for i in range(num_drones)]
    observation_spaces = {agent: 13 for agent in possible_agents}
    state_space = 0

    # Debug visualization 
    debug_vis = True
    debug_vis_goal = True
    debug_vis_action = True

    # MINCO trajectory
    num_pieces = 6
    duration = 0.3
    a_max = {agent: 10.0 for agent in possible_agents}
    v_max = {agent: 5.0 for agent in possible_agents}

    # FIXME: @configclass doesn't support the following syntax #^#
    # p_max = {agent: num_pieces * v_max[agent] * duration for agent in possible_agents}
    # action_space = {agent: 3 * (num_pieces + 2) for agent in possible_agents}  # inner_pts 3 x (num_pieces - 1) + tail_pva 3 x 3
    p_max = {agent: 6 * 5.0 * 0.3 for agent in possible_agents}
    action_spaces = {agent: 3 * (6 + 2) for agent in possible_agents}  # inner_pts 3 x (num_pieces - 1) + tail_pva 3 x 3

    clip_action = 100  # Default bound for box action spaces in IsaacLab Sb3VecEnvWrapper

    ui_window_class_type = QuadcopterEnvWindow

    # Simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=6, replicate_physics=True)

    # Robot
    drone_cfg: ArticulationCfg = DJI_FPV_CFG.copy()
    init_gap = 0.8

    # 奖励系数设置
    lin_vel_reward_scale = -0.0    # 禁用线速度惩罚
    ang_vel_reward_scale = -0.0    # 禁用角速度惩罚
    distance_to_goal_reward_scale = 5.0  # 增加靠近目标的奖励
    survival_reward_scale = 0.0    # 禁用生存奖励
    died_reward_scale = 0.0      # 减轻死亡惩罚
    
class QuadcopterSwarmEnv(DirectMARLEnv):
    cfg: QuadcopterSwarmEnvCfg
    has_debug_vis_implementation = True  # 在类级别定义属性

    def __init__(self, cfg: QuadcopterSwarmEnvCfg, render_mode: str | None = None, **kwargs):
        # 添加调试可视化实现标志
        self.has_debug_vis_implementation = True
        super().__init__(cfg, render_mode, **kwargs)

        if self.cfg.decimation < 1 or self.cfg.control_decimation < 1:
            raise ValueError("Replan and control decimation must be greater than or equal to 1 #^#")

        if 1 / self.cfg.mpc_freq > self.cfg.num_pieces * self.cfg.duration:
            raise ValueError("Replan period must be less than or equal to the total trajectory duration #^#")

        # Get specific body indices for each drone
        self.body_ids = {agent: self.robots[agent].find_bodies("body")[0] for agent in self.cfg.possible_agents}

        self.robot_masses = {agent: self.robots[agent].root_physx_view.get_masses()[0, 0] for agent in self.cfg.possible_agents}
        self.robot_inertias = {agent: self.robots[agent].root_physx_view.get_inertias()[0, 0] for agent in self.cfg.possible_agents}
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # Initialize desired positions for each agent
        self.desired_pos_w = {agent: torch.zeros((self.num_envs, 3), device=self.device) for agent in self.cfg.possible_agents}

        # Initialize episode sums for each agent and each reward component
        self.episode_sums = {}
        for agent in self.cfg.possible_agents:
            for key in ["lin_vel", "ang_vel", "distance_to_goal", "survival", "died"]:  # 添加died到跟踪的奖励组件中
                self.episode_sums[f"{agent}_{key}"] = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # Initialize extras dictionary with proper structure
        self.extras = {"log": {}}
        # Initialize base metrics as tensors for all envs
        for agent in self.cfg.possible_agents:
            for key in ["lin_vel", "ang_vel", "distance_to_goal", "survival", "died"]:  # 同样更新log字典
                self.extras["log"][f"Rewards/{agent}/{key}"] = torch.zeros(self.num_envs, device=self.device)
            self.extras["log"][f"Metrics/final_distance_to_goal/{agent}"] = torch.zeros(self.num_envs, device=self.device)
        self.extras["log"]["Episode_Termination/died"] = torch.zeros(self.num_envs, device=self.device)
        self.extras["log"]["Episode_Termination/time_out"] = torch.zeros(self.num_envs, device=self.device)

        # Traj
        self.waypoints, self.trajs = {}, {}
        # elesheep why here isn't it a dict?
        self.has_prev_traj = torch.tensor([False] * self.num_envs, device=self.device)

        # Controllers
        ## elesheep 
        #         self.actions = torch.cat(
        #     (
        #         self.traj.get_pos(self.execution_time),
        #         self.traj.get_vel(self.execution_time),
        #         self.traj.get_acc(self.execution_time),
        #         self.traj.get_jer(self.execution_time),
        #         torch.zeros_like(self.execution_time).unsqueeze(1),
        #         torch.zeros_like(self.execution_time).unsqueeze(1),
        #     ),
        #     dim=1,
        # )
        self.actions = {}
        self.a_desired_total, self.thrust_desired, self._thrust_desired, self.q_desired, self.w_desired, self.m_desired = {}, {}, {}, {}, {}, {}
        self.controllers = {
            agent: Controller(self.step_dt, self.gravity, self.robot_masses[agent].to(self.device), self.robot_inertias[agent].to(self.device))
            for agent in self.cfg.possible_agents
        }
        self.control_counter = 0

        # 添加调试可视化
        self.set_debug_vis(self.cfg.debug_vis)
        self.visualize_new_cmd = False

        ## elesheep need to add visualizer

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
        
        # 直接调用实现方法，避免依赖基类的set_debug_vis
        self._set_debug_vis_impl(self.cfg.debug_vis)
        self.visualize_new_cmd = False

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if self.cfg.debug_vis_goal:
                # 为每个智能体创建目标位置可视化器
                self.goal_pos_visualizers = {}
                for i, agent in enumerate(self.cfg.possible_agents):
                    marker_cfg = CUBOID_MARKER_CFG.copy()
                    marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                    marker_cfg.markers["cuboid"].visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))
                    marker_cfg.prim_path = f"/Visuals/Command/goal_{i}"
                    self.goal_pos_visualizers[agent] = VisualizationMarkers(marker_cfg)
                    self.goal_pos_visualizers[agent].set_visibility(True)
            else:
                if hasattr(self, "goal_pos_visualizers"):
                    for agent in self.cfg.possible_agents:
                        if agent in self.goal_pos_visualizers:
                            self.goal_pos_visualizers[agent].set_visibility(False)

            if self.cfg.debug_vis_action:
                # 为每个智能体创建路径点可视化器
                self.waypoint_visualizers = {}
                self.tailpoint_visualizers = {}
                for i, agent in enumerate(self.cfg.possible_agents):
                    # 路径点可视化器
                    marker_cfg = VisualizationMarkersCfg(
                        prim_path=f"/Visuals/Command/waypoints_{i}",
                        markers={
                            "sphere": sim_utils.SphereCfg(
                                radius=0.008,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                            ),
                        },
                    )
                    self.waypoint_visualizers[agent] = VisualizationMarkers(marker_cfg)
                    self.waypoint_visualizers[agent].set_visibility(True)

                    # 终点可视化器
                    marker_cfg = VisualizationMarkersCfg(
                        prim_path=f"/Visuals/Command/tailpoint_{i}",
                        markers={
                            "sphere": sim_utils.SphereCfg(
                                radius=0.015,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
                            ),
                        },
                    )
                    self.tailpoint_visualizers[agent] = VisualizationMarkers(marker_cfg)
                    self.tailpoint_visualizers[agent].set_visibility(True)
            else:
                if hasattr(self, "waypoint_visualizers"):
                    for agent in self.cfg.possible_agents:
                        if agent in self.waypoint_visualizers:
                            self.waypoint_visualizers[agent].set_visibility(False)
                if hasattr(self, "tailpoint_visualizers"):
                    for agent in self.cfg.possible_agents:
                        if agent in self.tailpoint_visualizers:
                            self.tailpoint_visualizers[agent].set_visibility(False)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        head_pva_all, inner_pts_all, tail_pva_all, durations_all = [], [], [], []
        for agent in self.possible_agents:
            # Action parametrization: waypoints in body frame
            self.waypoints[agent] = actions[agent].clone().clamp(-self.cfg.clip_action, self.cfg.clip_action) / self.cfg.clip_action

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
                inner_pts[:, :, i] = quat_rotate(q_odom, self.waypoints[agent][:, 3 * i : 3 * (i + 1)] * self.cfg.p_max[agent]) + p_odom
            inner_pts_all.append(inner_pts)

            # Tail states, transformed to world frame
            p_tail = quat_rotate(q_odom, self.waypoints[agent][:, 3 * (self.cfg.num_pieces - 1) : 3 * (self.cfg.num_pieces + 0)] * self.cfg.p_max[agent]) + p_odom
            v_tail = quat_rotate(q_odom, self.waypoints[agent][:, 3 * (self.cfg.num_pieces + 0) : 3 * (self.cfg.num_pieces + 1)] * self.cfg.v_max[agent])
            a_tail = quat_rotate(q_odom, self.waypoints[agent][:, 3 * (self.cfg.num_pieces + 1) : 3 * (self.cfg.num_pieces + 2)] * self.cfg.a_max[agent])
            tail_pva = torch.stack([p_tail, v_tail, a_tail], dim=2)
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
        logger.debug(f"Local trajectories generation takes {end - start:.6f}s")

        traj_all = MJO.get_traj()
        self.trajs = {agent: traj_all[i * self.num_envs : (i + 1) * self.num_envs] for i, agent in enumerate(self.possible_agents)}
        self.execution_time = torch.zeros(self.num_envs, device=self.device)
        self.has_prev_traj.fill_(True)
        
        # 标记需要可视化新的指令
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
            logger.debug(f"get_control for all drones takes {end - start:.6f}s")

            self.control_counter = 0
        self.control_counter += 1

        for agent in self.possible_agents:
            self.robots[agent].set_external_force_and_torque(self._thrust_desired[agent].unsqueeze(1), self.m_desired[agent].unsqueeze(1), body_ids=self.body_ids[agent])
        self.execution_time += self.physics_dt

    def _get_observations(self) -> dict[str, torch.Tensor]:
        observations = {}
        for agent in self.possible_agents:
            # 计算目标在机体坐标系下的相对位置（类似于quadcopter_env.py的实现）
            goal_in_body_frame = quat_rotate(quat_inv(self.robots[agent].data.root_quat_w), 
                                            self.desired_pos_w[agent] - self.robots[agent].data.root_pos_w)
            
            # 将目标相对位置、四元数姿态和世界坐标系下的速度拼接成观察向量
            obs = torch.cat(
                [
                    goal_in_body_frame,                   # 目标在机体坐标系下的相对位置 (3)
                    self.robots[agent].data.root_quat_w,  # 四元数姿态 (4)
                    self.robots[agent].data.root_vel_w,   # 世界坐标系下的线速度和角速度 (6)
                ],
                dim=-1,
            )
            observations[agent] = obs
            
        return observations

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rewards = {}
        
        # 获取死亡状态信息用于计算died奖励
        died_agents, _ = self._get_dones()
        
        
        for agent in self.possible_agents:
            # Calculate raw components
            components = {
                "lin_vel": torch.sum(torch.square(self.robots[agent].data.root_lin_vel_b), dim=1),
                "ang_vel": torch.sum(torch.square(self.robots[agent].data.root_ang_vel_b), dim=1),
                "distance_to_goal": torch.linalg.norm(self.desired_pos_w[agent] - self.robots[agent].data.root_pos_w, dim=1),
                "survival": torch.ones(self.num_envs, device=self.device),  # 添加生存奖励组件，每一步都给予固定奖励
                "died": died_agents[agent].float()  # 添加死亡标志，用于计算死亡惩罚
            }
            
            # 记录原始距离，方便调试
            raw_distance = components["distance_to_goal"].clone()
            self.extras["log"][f"Raw_Components/{agent}/raw_distance"] = raw_distance
            
            # 使用tanh函数计算距离奖励，参考quadcopter_env.py的实现
            # 1 - tanh(distance/scale)，距离为0时奖励为1，随距离增加而平滑减少
            components["distance_to_goal"] = 1.0 - torch.tanh(components["distance_to_goal"] / 3)
            
            # Calculate final rewards with scaling factors（取消每步的时间因子self.step_dt）
            agent_rewards = {
                "lin_vel": components["lin_vel"] * self.cfg.lin_vel_reward_scale,
                "ang_vel": components["ang_vel"] * self.cfg.ang_vel_reward_scale,
                "distance_to_goal": components["distance_to_goal"] * self.cfg.distance_to_goal_reward_scale,
                "survival": components["survival"] * self.cfg.survival_reward_scale,
                "died": components["died"] * self.cfg.died_reward_scale
            }
            
            # Calculate total reward for this agent
            agent_total_reward = torch.sum(torch.stack(list(agent_rewards.values())), dim=0)
            
            # Log individual reward components
            for key, value in agent_rewards.items():
                self.episode_sums[f"{agent}_{key}"] = self.episode_sums.get(f"{agent}_{key}", torch.zeros_like(value)) + value
                # Update the extras dictionary with the current values
                self.extras["log"][f"Rewards/{agent}/{key}"] = value
                
                # 添加原始值记录
                self.extras["log"][f"Raw_Components/{agent}/{key}"] = components[key]
            
            # 添加总奖励记录
            self.extras["log"][f"Total_Reward/{agent}"] = agent_total_reward
            
            rewards[agent] = agent_total_reward
            
        return rewards

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        died = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        for agent in self.possible_agents:
            # 检查高度是否超出范围：低于0.1m或高于10m
            z_exceed_bounds = torch.logical_or(self.robots[agent].data.root_link_pos_w[:, 2] < 0.1, self.robots[agent].data.root_link_pos_w[:, 2] > 10.0)
            # if z_exceed_bounds.any():
            #     print("z_exceed_bounds")
            # 检查倾角是否过大：z轴与世界z轴夹角超过60度
            ang_between_z_body_and_z_world = torch.rad2deg(quat_to_ang_between_z_body_and_z_world(self.robots[agent].data.root_link_quat_w))
            # if (ang_between_z_body_and_z_world > 60.0).any():
            #     print("ang_between_z_body_and_z_world")
            _died = torch.logical_or(z_exceed_bounds, ang_between_z_body_and_z_world > 60.0)
            # 只要有一个智能体死亡，整个环境就会终止
            died = torch.logical_or(died, _died)


        # 检查是否达到最大步数
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 返回两个字典：一个表示"死亡"终止，一个表示"时间到"终止
        return {agent: died for agent in self.cfg.possible_agents}, {agent: time_out for agent in self.cfg.possible_agents}

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self.robots["drone_0"]._ALL_INDICES

        # Logging episode rewards
        for key in self.episode_sums.keys():
            episodic_sum_avg = torch.mean(self.episode_sums[key][env_ids])
            self.extras["log"]["Episode_Reward/" + key] = torch.full((self.num_envs,), episodic_sum_avg.item() / self.max_episode_length_s, device=self.device)
            
            # 计算平均每步奖励
            self.extras["log"]["Episode_Reward_PerStep/" + key] = torch.full((self.num_envs,), episodic_sum_avg.item() / self.max_episode_length, device=self.device)
            
            # 清零episode累积值
            self.episode_sums[key][env_ids] = 0.0
        
        # Update termination states
        terminated, time_out = self._get_dones()
        self.extras["log"]["Episode_Termination/died"] = torch.full((self.num_envs,), torch.count_nonzero(terminated["drone_0"][env_ids]).item(), device=self.device)
        self.extras["log"]["Episode_Termination/time_out"] = torch.full((self.num_envs,), torch.count_nonzero(time_out["drone_0"][env_ids]).item(), device=self.device)
        
        # Update final distance to goal for each agent
        for agent in self.possible_agents:
            final_distance_to_goal = torch.linalg.norm(
                self.desired_pos_w[agent][env_ids] - self.robots[agent].data.root_pos_w[env_ids], 
                dim=1
            ).mean()
            self.extras["log"][f"Metrics/final_distance_to_goal/{agent}"] = torch.full((self.num_envs,), final_distance_to_goal.item(), device=self.device)
        
        for agent in self.possible_agents:
            self.robots[agent].reset(env_ids)

        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self.has_prev_traj[env_ids].fill_(False)
        
        for agent in self.possible_agents:
            # 目标点采样 - 使用均匀分布在最大距离范围内采样
            # 水平方向距离
            self.desired_pos_w[agent][env_ids, :2] = torch.zeros_like(self.desired_pos_w[agent][env_ids, :2]).uniform_(-5.0, 5.0)
            self.desired_pos_w[agent][env_ids, :2] += self.terrain.env_origins[env_ids, :2]
            self.desired_pos_w[agent][env_ids, 2] = torch.zeros_like(self.desired_pos_w[agent][env_ids, 2]).uniform_(0.5, 3.0)
        
        # Reset robot state
        for agent in self.possible_agents:
            joint_pos = self.robots[agent].data.default_joint_pos[env_ids]
            joint_vel = self.robots[agent].data.default_joint_vel[env_ids]
            default_root_state = self.robots[agent].data.default_root_state[env_ids]
            default_root_state[:, :3] += self.terrain.env_origins[env_ids]
            self.robots[agent].write_root_pose_to_sim(default_root_state[:, :7], env_ids)
            self.robots[agent].write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
            self.robots[agent].write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_pos_visualizers"):
            for agent in self.cfg.possible_agents:
                if agent in self.goal_pos_visualizers:
                    self.goal_pos_visualizers[agent].visualize(translations=self.desired_pos_w[agent])

        if self.visualize_new_cmd and hasattr(self, "waypoint_visualizers") and hasattr(self, "tailpoint_visualizers"):
            for agent in self.cfg.possible_agents:
                if agent in self.waypoint_visualizers and agent in self.tailpoint_visualizers:
                    p_odom = self.robots[agent].data.root_state_w[:, :3]
                    q_odom = self.robots[agent].data.root_state_w[:, 3:7]

                    inner_pts_world = torch.zeros((self.num_envs, self.cfg.num_pieces - 1, 3), device=self.device)
                    for i in range(self.cfg.num_pieces - 1):
                        inner_pts_world[:, i] = quat_rotate(q_odom, self.waypoints[agent][:, 3 * i : 3 * (i + 1)] * self.cfg.p_max[agent]) + p_odom

                    inner_pts_flat = inner_pts_world.reshape(-1, 3)
                    self.waypoint_visualizers[agent].visualize(translations=inner_pts_flat)
                    
                    p_tail = quat_rotate(q_odom, self.waypoints[agent][:, 3 * (self.cfg.num_pieces - 1) : 3 * (self.cfg.num_pieces + 0)] * self.cfg.p_max[agent]) + p_odom
                    self.tailpoint_visualizers[agent].visualize(translations=p_tail)

            self.visualize_new_cmd = False

from config import agents

gym.register(
    id="FAST-Quadcopter-Swarm-Direct-v0",
    entry_point=QuadcopterSwarmEnv,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": QuadcopterSwarmEnvCfg,
            "sb3_cfg_entry_point": f"{agents.__name__}:quadcopter_swarm_sb3_ppo_cfg.yaml",
        },
)
