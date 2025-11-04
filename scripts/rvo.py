"""Script to run Reciprocal Collision Avoidance (RVO) in swarm velocity environment"""

import argparse
import os
import sys
from isaaclab.app import AppLauncher


# Add argparse arguments
parser = argparse.ArgumentParser(description="Run RVO in swarm velocity environment.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--verbosity", type=str, default="INFO", choices=["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"], help="Verbosity level of the custom logger."
)

# Append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# Parse the arguments
args_cli = parser.parse_args()

# Launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# TODO: Improve import modality
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

import gymnasium as gym
from loguru import logger
import math
import rclpy
import torch
from typing import Dict, List

from envs import swarm_vel_env
from isaaclab_tasks.utils import parse_env_cfg


# =========================
# RVO / ORCA (agents-only)
# =========================


@torch.no_grad()
def rvo(state: torch.Tensor, env_cfg, params: Dict = None) -> Dict[str, torch.Tensor]:
    """
    Args:
        state: Tensor[ num_envs, state_dim ], concatenated in the order:
               [ a0_action(2), a0_pos(3), a0_goal_rel(3), a0_quat(4), a0_root_vel(6),
                 a1_action(2), a1_pos(3), ... concatenated for all agents ]
        env_cfg: uses env_cfg.possible_agents (same order as in `state`)
        params:
            {
              "dt": simulation time step (seconds),
              "time_horizon": human-human time horizon T,
              "neighbor_dist": neighborhood radius (meters),
              "radius": agent radius (meters) or Tensor[num_agents] / Tensor[num_envs, num_agents],
              "max_speed": speed limit (m/s) or Tensor[num_agents] / Tensor[num_envs, num_agents],
              # Optional: if you want to be more conservative, set "inflation": extra safety margin (meters), default 0
            }
    Returns:
        actions: { agent_id: Tensor[num_envs, 2] }, 2D velocity commands in the world frame
    """

    if params is None:
        params = {}
    dt = float(params.get("dt", 0.01))  # seconds
    T_h = float(params.get("time_horizon", 5.0))  # seconds
    neighbor_dist = float(params.get("neighbor_dist", 5.0))  # meters
    radius = params.get("radius", 0.25)  # meters
    max_speed = params.get("max_speed", 5.0)  # m/s
    success_threshold = float(params.get("success_threshold", 0.1))  # meters
    inflation = float(params.get("inflation", 0.0))

    agent_ids: List[str] = list(env_cfg.possible_agents)
    num_envs = state.shape[0]
    num_agents = len(agent_ids)

    # Parse slices of state
    d_action = 2
    d_pos = 3
    d_goal = 3
    d_quat = 4
    d_rvel = 6  # linear vel (3) + angular vel (3)
    stride = d_action + d_pos + d_goal + d_quat + d_rvel

    assert state.shape[1] == stride * num_agents, f"state_dim={state.shape[1]} does not match num_agents={num_agents}, expected {stride * num_agents} #^#"

    def take_chunk(i, off, dim):
        s = i * stride + off
        return state[:, s : s + dim]

    # For each agent, take xy position, xy linear velocity, and the xy components of the goal-relative vector
    pos_xy = torch.empty(num_envs, num_agents, 2, device=state.device, dtype=state.dtype)
    vel_xy = torch.empty_like(pos_xy)
    goal_rel_xy = torch.empty_like(pos_xy)

    for i in range(num_agents):
        pos = take_chunk(i, d_action, d_pos)  # (N,3)
        goal_rel = take_chunk(i, d_action + d_pos, d_goal)  # (N,3)
        root_vel = take_chunk(i, d_action + d_pos + d_goal + d_quat, d_rvel)  # (N,6)
        pos_xy[:, i] = pos[:, :2]
        vel_xy[:, i] = root_vel[:, :2]
        goal_rel_xy[:, i] = goal_rel[:, :2]

    # Broadcast parameters to (N,A)
    def to_tensor_like(x, ref):
        if isinstance(x, (int, float)):
            t = torch.full((num_envs, num_agents, 1), float(x), device=ref.device, dtype=ref.dtype)
        elif isinstance(x, torch.Tensor):
            x = x.to(ref)
            if x.ndim == 0:
                t = torch.full((num_envs, num_agents, 1), float(x.item()), device=ref.device, dtype=ref.dtype)
            elif x.ndim == 1:  # (N)
                assert x.shape[0] == num_agents, "Length of 1D parameter should be num_agents #^#"
                t = x.view(1, num_agents, 1).expand(num_envs, num_agents, 1)
            elif x.ndim == 2:  # (N,A)
                assert x.shape == (num_envs, num_agents), "Shape of 2D parameter should be (num_envs, num_agents) #^#"
                t = x.unsqueeze(-1)
            elif x.ndim == 3:
                assert x.shape == (num_envs, num_agents, 1), "Shape of 3D parameter should be (num_envs, num_agents, 1) #^#"
                t = x
            else:
                raise ValueError("Unsupported parameter dimensionality #^#")
        else:
            raise TypeError("Parameters must be of type float/int/torch.Tensor #^#")
        return t

    R = to_tensor_like(radius + inflation, pos_xy)  # (N,A,1)
    Vmax = to_tensor_like(max_speed, pos_xy)  # (N,A,1)
    invT = torch.as_tensor(1.0 / max(T_h, 1e-6), device=state.device, dtype=state.dtype)

    # Preferred velocity: toward the goal; magnitude = Vmax
    goal_norm = torch.norm(goal_rel_xy, dim=-1, keepdim=True).clamp_min(1e-12)
    pref_dir = goal_rel_xy / goal_norm
    pref_speed = Vmax
    pref_vel = pref_dir * pref_speed  # (N,A,2)
    success_mask = goal_norm < success_threshold  # (N,A,1)
    pref_vel = torch.where(success_mask, torch.zeros_like(pref_vel), pref_vel)

    # Perturb a little to avoid deadlocks due to perfect symmetry
    angle = 2.0 * math.pi * torch.rand(pref_vel.shape[0], pref_vel.shape[1], 1, device=pref_vel.device, dtype=pref_vel.dtype)
    dist = 1.0 * Vmax * torch.rand(pref_vel.shape[0], pref_vel.shape[1], 1, device=pref_vel.device, dtype=pref_vel.dtype)
    noise = dist * torch.cat([torch.cos(angle), torch.sin(angle)], dim=-1)  # (N,A,2)
    pref_vel += noise

    # Neighbor mask (other agents in the same env, distance < neighbor_dist)
    diff = pos_xy.unsqueeze(2) - pos_xy.unsqueeze(1)  # (N,A,A,2): i-j
    dist2 = torch.sum(diff * diff, dim=-1)  # (N,A,A)
    eye = torch.eye(num_agents, dtype=torch.bool, device=state.device).unsqueeze(0)
    neighbor_mask = (dist2 < neighbor_dist * neighbor_dist) & (~eye)

    # Utility functions (2D)
    def det2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def normalize(v, eps=1e-8):
        n = torch.linalg.norm(v, dim=-1, keepdim=True)
        return v / torch.clamp(n, min=eps)

    new_vel = torch.empty_like(pref_vel)  # (N,A,2)

    # ========= Main loop: solve ORCA per env and per agent =========
    for e in range(num_envs):
        # Snapshot for env_e
        P = pos_xy[e]  # (A,2)
        V = vel_xy[e]  # (A,2)
        PV = pref_vel[e]  # (A,2)
        Re = R[e].squeeze(-1)  # (A,)
        Vmax_e = Vmax[e].squeeze(-1)  # (A,)
        neigh = neighbor_mask[e]  # (A,A)

        for i in range(num_agents):
            # Build ORCA lines for agent_i
            points: List[torch.Tensor] = []
            dirs: List[torch.Tensor] = []

            js = torch.nonzero(neigh[i], as_tuple=False).squeeze(-1).tolist()
            for j in js:
                rel_pos = P[j] - P[i]  # (2,)
                rel_vel = V[i] - V[j]  # (2,)
                dist2_ij = torch.dot(rel_pos, rel_pos).item()
                R_ij = Re[i].item() + Re[j].item()
                R2_ij = R_ij * R_ij

                if dist2_ij > R2_ij:
                    # No collision: use time horizon
                    w = rel_vel - rel_pos * invT  # (2,)
                    w2 = torch.dot(w, w).item()
                    rp_dot_w = torch.dot(rel_pos, w).item()

                    if (rp_dot_w < 0.0) and (rp_dot_w * rp_dot_w > R2_ij * w2):
                        # Project to truncated circle
                        w_len = math.sqrt(max(w2, 0.0)) + 1e-12
                        unitW = w / w_len
                        line_dir = torch.tensor([unitW[1].item(), -unitW[0].item()], device=state.device, dtype=state.dtype)
                        u = (R_ij * invT - w_len) * unitW
                    else:
                        # Project to left/right legs
                        leg = math.sqrt(max(dist2_ij - R2_ij, 0.0)) + 1e-12
                        if det2(rel_pos, w).item() > 0.0:
                            # Left leg
                            line_dir = torch.tensor(
                                [(rel_pos[0] * leg - rel_pos[1] * R_ij) / dist2_ij, (rel_pos[0] * R_ij + rel_pos[1] * leg) / dist2_ij],
                                device=state.device,
                                dtype=state.dtype,
                            )
                        else:
                            # Right leg
                            line_dir = -torch.tensor(
                                [(rel_pos[0] * leg + rel_pos[1] * R_ij) / dist2_ij, (-rel_pos[0] * R_ij + rel_pos[1] * leg) / dist2_ij],
                                device=state.device,
                                dtype=state.dtype,
                            )

                        dot2 = torch.dot(rel_vel, line_dir).item()
                        u = dot2 * line_dir - rel_vel
                else:
                    # Already penetrating: stronger correction using dt
                    inv_dt = 1.0 / max(dt, 1e-6)
                    w = rel_vel - rel_pos * inv_dt
                    w_len = torch.linalg.norm(w).item() + 1e-12
                    unitW = w / w_len
                    line_dir = torch.tensor([unitW[1].item(), -unitW[0].item()], device=state.device, dtype=state.dtype)
                    u = (R_ij * inv_dt - w_len) * unitW

                line_point = V[i] + 0.5 * u
                line_dir = normalize(line_dir.unsqueeze(0)).squeeze(0)

                points.append(line_point)
                dirs.append(line_dir)

            # ====== Linear programming: in the intersection of half-planes ∩ velocity ball, closest to preferred velocity ======

            # Half-plane feasibility
            def violates(k, v):
                # RVO2/C++: violate if det(dir_k, point_k - v) > 0
                # Equivalent: det(dir_k, v - point_k) < 0 indicates violation
                return det2(dirs[k], v - points[k]).item() < 0.0

            def lp1(lineNo, cur_res, radius, opt_v, direction_opt):
                p = points[lineNo]
                d = dirs[lineNo]
                dot_pd = torch.dot(p, d).item()
                disc = dot_pd * dot_pd + radius * radius - torch.dot(p, p).item()
                if disc < 0.0:
                    return False, cur_res

                sqrt_disc = math.sqrt(disc)
                tL, tR = -dot_pd - sqrt_disc, -dot_pd + sqrt_disc

                # Intersect feasible interval with previous lines
                for i_line in range(lineNo):
                    denom = det2(d, dirs[i_line]).item()
                    numer = det2(dirs[i_line], p - points[i_line]).item()
                    if abs(denom) <= 1e-8:
                        # Parallel
                        if numer < 0.0:
                            return False, cur_res
                        else:
                            continue
                    t = numer / denom
                    if denom >= 0.0:
                        tR = min(tR, t)
                    else:
                        tL = max(tL, t)
                    if tL > tR:
                        return False, cur_res

                if direction_opt:
                    # Direction optimality: choose left/right endpoint based on dot(opt_v, d)
                    if torch.dot(opt_v, d).item() > 0.0:
                        t_star = tR
                    else:
                        t_star = tL
                else:
                    # Closest point optimality
                    t_star = torch.dot(d, (opt_v - p)).item()
                    if t_star < tL:
                        t_star = tL
                    elif t_star > tR:
                        t_star = tR

                return True, (p + t_star * d)

            def lp2(points, dirs, radius, opt_v, direction_opt=False):
                # Initial value
                if direction_opt:
                    # In direction optimization mode, opt_v is treated as a unit direction
                    res = opt_v * radius
                else:
                    # Closest point mode
                    if torch.dot(opt_v, opt_v).item() > radius * radius:
                        res = normalize(opt_v.unsqueeze(0)).squeeze(0) * radius
                    else:
                        res = opt_v.clone()

                for k in range(len(points)):
                    if violates(k, res):
                        ok, res_new = lp1(k, res, radius, opt_v, direction_opt)
                        if not ok:
                            return k, res
                        res = res_new
                return len(points), res

            def lp3(points, dirs, begin_idx, radius, cur_res):
                distance = 0.0
                for i_line in range(begin_idx, len(points)):
                    val = det2(dirs[i_line], points[i_line] - cur_res).item()
                    if val > distance:
                        # Build the projection line set (only from begin_idx to i_line-1)
                        proj_points = []
                        proj_dirs = []
                        for j_line in range(begin_idx, i_line):
                            det_ij = det2(dirs[i_line], dirs[j_line]).item()
                            if abs(det_ij) <= 1e-8:
                                if torch.dot(dirs[i_line], dirs[j_line]).item() > 0:
                                    continue
                                else:
                                    proj_p = 0.5 * (points[i_line] + points[j_line])
                            else:
                                proj_p = points[i_line] + (det2(dirs[j_line], points[i_line] - points[j_line]).item() / det_ij) * dirs[i_line]
                            proj_d = normalize((dirs[j_line] - dirs[i_line]).unsqueeze(0)).squeeze(0)
                            proj_points.append(proj_p)
                            proj_dirs.append(proj_d)

                        # Key: direction_opt=True, target direction is the perpendicular of the current line
                        perp = torch.tensor([-dirs[i_line][1].item(), dirs[i_line][0].item()], device=cur_res.device, dtype=cur_res.dtype)
                        fail_k, cur_res_new = lp2(proj_points, proj_dirs, radius, perp, direction_opt=True)

                        # In theory it should succeed; if it fails (numerical issues), keep the old solution
                        cur_res = cur_res if fail_k < len(proj_points) else cur_res_new
                        distance = det2(dirs[i_line], points[i_line] - cur_res).item()
                return cur_res

            vmax = float(Vmax_e[i].item())

            # First LP2 (closest point mode)
            fail_idx, res_i = lp2(points, dirs, vmax, PV[i], direction_opt=False)
            if fail_idx < len(points):
                # If it fails, run LP3 (direction optimization)
                res_i = lp3(points, dirs, fail_idx, vmax, res_i)

            new_vel[e, i] = res_i

    speed = torch.norm(new_vel, dim=-1, keepdim=True)
    new_vel = torch.where(speed > Vmax, new_vel * (Vmax / (speed + 1e-12)), new_vel)
    new_vel /= Vmax

    actions: Dict[str, torch.Tensor] = {agent_ids[i]: new_vel[:, i, :] for i in range(num_agents)}
    return actions


def main():
    # Create environment configuration
    env_cfg = parse_env_cfg("FAST-Swarm-Vel", device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.fix_range = True
    env_cfg.goal_reset_delay /= 1.5
    env_cfg.episode_length_s /= 10.0
    # Create environment
    env = gym.make("FAST-Swarm-Vel", cfg=env_cfg)
    env.reset()
    state = env.unwrapped.state()

    rvo_params = dict(
        dt=env.unwrapped.step_dt,
        time_horizon=5.0,
        neighbor_dist=5.0,
        radius=env_cfg.collide_dist / 2,
        max_speed=env_cfg.v_max[env_cfg.possible_agents[0]],
    )

    while simulation_app.is_running():
        with torch.inference_mode():
            actions = rvo(state, env_cfg, rvo_params)
            env.step(actions)
            state = env.unwrapped.state()


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stdout, level=args_cli.verbosity)

    rclpy.init()
    main()
    rclpy.shutdown()

    simulation_app.close()
