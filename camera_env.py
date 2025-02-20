from __future__ import annotations

import gymnasium as gym
from loguru import logger
import math
import time
import torch

import isaaclab.sim as sim_utils
import isaacsim.core.utils.prims as prim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg, ViewerCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg, save_images_to_file
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from quadcopter import CRAZYFLIE_CFG, DJI_FPV_CFG  # isort: skip
from utils import quat_to_ang_between_z_body_and_z_world
from controller import Controller


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment"""

    def __init__(self, env: QuadcopterCameraEnv, window_name: str = "IsaacLab"):
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
class QuadcopterRGBCameraEnvCfg(DirectRLEnvCfg):
    # Change viewer settings
    viewer = ViewerCfg(eye=(-5.0, -5.0, 2.5))

    # Env
    episode_length_s = 60.0
    decimation = 2
    action_space = 14
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=3, replicate_physics=True)

    # Robot
    robot: ArticulationCfg = DJI_FPV_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    p_max = 500.0
    v_max = 5.0
    yaw_dot_max = math.pi / 2

    # Camera
    # Hi there, Isaac Sim does not currently provide independent cameras that don’t see other environments.
    # One way to workaround it is to build walls around the environments,
    # which would just be large rectangle prims that block the views of other environments.
    # Another alternative would be to place the environments far apart, or on different height levels.
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.05), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 20.0),
        ),
        width=320,
        height=240,
    )
    observation_space = [tiled_camera.height, tiled_camera.width, 3]
    write_image_to_file = False

    # Reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


@configclass
class QuadcopterDepthCameraEnvCfg(QuadcopterRGBCameraEnvCfg):
    # Camera
    max_depth = 10.0
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.1, 0.0, 0.05), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        data_types=["depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, max_depth),
        ),
        width=320,
        height=240,
    )
    observation_space = [tiled_camera.height, tiled_camera.width, 1]


class QuadcopterCameraEnv(DirectRLEnv):
    cfg: QuadcopterRGBCameraEnvCfg | QuadcopterDepthCameraEnvCfg

    def __init__(self, cfg: QuadcopterRGBCameraEnvCfg | QuadcopterDepthCameraEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # Goal position
        self.desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self.episode_sums = {key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in ["lin_vel", "ang_vel", "distance_to_goal"]}

        # Get specific indices
        self.body_id = self.robot.find_bodies("body")[0]
        self.joint_id = self.robot.find_joints(".*joint")[0]

        self.robot_mass = self.robot.root_physx_view.get_masses()[0, 0]
        self.robot_inertia = self.robot.root_physx_view.get_inertias()[0, 0]
        self.gravity = torch.tensor(self.sim.cfg.gravity, device=self.device)

        # Controller
        self.controller = Controller(self.step_dt, self.gravity, self.robot_mass.to(self.device), self.robot_inertia.to(self.device))

        # Add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

        if len(self.cfg.tiled_camera.data_types) != 1:
            raise ValueError(
                "Currently, the camera environment only supports one image type at a time but the following were"
                f" provided: {self.cfg.tiled_camera.data_types}"
            )

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot

        self.tiled_camera = TiledCamera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self.tiled_camera

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

        prim_utils.create_prim("/World/Objects", "Xform")

        cfg_cone_rigid = sim_utils.ConeCfg(
            radius=0.25,
            height=1.618 / 2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.77, 0.23, 0.23)),
        )
        cfg_cone_rigid.func(
            "/World/Objects/ConeRigid",
            cfg_cone_rigid,
            translation=(4.0, 1.0, 0.5),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

        cfg_cylinder_rigid = sim_utils.CylinderCfg(
            radius=0.25,
            height=1.618 / 2,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.23, 0.23, 0.77)),
        )
        cfg_cylinder_rigid.func(
            "/World/Objects/CylinderRigid",
            cfg_cylinder_rigid,
            translation=(4.0, 2.0, 0.5),
            orientation=(1.0, 0.0, 0.0, 0.0),
        )

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

    def _apply_action(self):
        self.robot.set_external_force_and_torque(self._thrust_desired.unsqueeze(1), self.m_desired.unsqueeze(1), body_ids=self.body_id)

        # TODO: only for visualization 0_0 And it's not working due to unknown reason
        self.robot.set_joint_velocity_target(self.robot.data.default_joint_vel, joint_ids=self.joint_id, env_ids=self.robot._ALL_INDICES)

    def _get_observations(self) -> dict:
        data_type = "rgb" if "rgb" in self.cfg.tiled_camera.data_types else "depth"
        if "rgb" in self.cfg.tiled_camera.data_types:
            camera_data = self.tiled_camera.data.output[data_type] / 255.0

            # Normalize the camera data for better training results
            # mean_tensor = torch.mean(camera_data, dim=(1, 2), keepdim=True)
            # camera_data -= mean_tensor

        elif "depth" in self.cfg.tiled_camera.data_types:
            camera_data = self.tiled_camera.data.output[data_type]
            camera_data[camera_data == float("inf")] = 0
            camera_data /= self.cfg.max_depth

        odom = self.robot.data.root_link_state_w.clone()
        odom[:, :3] -= self.terrain.env_origins

        observations = {"image": camera_data.clone(), "policy": odom}

        if self.cfg.write_image_to_file:
            save_images_to_file(observations["image"], f"quadcopter_{data_type}.png")

        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self.robot.data.root_com_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self.robot.data.root_com_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self.desired_pos_w - self.robot.data.root_link_pos_w, dim=1)
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
        final_distance_to_goal = torch.linalg.norm(self.desired_pos_w[env_ids] - self.robot.data.root_link_pos_w[env_ids], dim=1).mean()
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
        self.desired_pos_w[env_ids, :2] = torch.zeros_like(self.desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self.desired_pos_w[env_ids, :2] += self.terrain.env_origins[env_ids, :2]
        self.desired_pos_w[env_ids, 2] = torch.zeros_like(self.desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

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
        self.goal_pos_visualizer.visualize(self.desired_pos_w)


gym.register(
    id="FAST-Quadcopter-RGB-Camera-Direct-v0",
    entry_point=QuadcopterCameraEnv,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": QuadcopterRGBCameraEnvCfg},
)

gym.register(
    id="FAST-Quadcopter-Depth-Camera-Direct-v0",
    entry_point=QuadcopterCameraEnv,
    disable_env_checker=True,
    kwargs={"env_cfg_entry_point": QuadcopterDepthCameraEnvCfg},
)
