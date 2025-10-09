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
        state: Tensor[ num_envs, state_dim ], 拼接顺序为
               [ a0_action(2), a0_pos(3), a0_goal_rel(3), a0_quat(4), a0_root_vel(6),
                 a1_action(2), a1_pos(3), ... 依次拼接所有 agents ]
        env_cfg: 需要用到 env_cfg.possible_agents (按 state 拼接同顺序)
        params:
            {
              "dt": 仿真步长(秒),
              "time_horizon": 人-人时间视界 T,
              "neighbor_dist": 邻域半径 (米),
              "radius": agent 半径 (米) 或 Tensor[num_agents] / Tensor[num_envs,num_agents],
              "max_speed": 速度上限 (米/秒) 或 Tensor[num_agents] / Tensor[num_envs,num_agents],
              # 可选：若想更保守，可设 "inflation": 额外安全余量(米)，默认 0
            }
    Returns:
        actions: { agent_id: Tensor[num_envs, 2] }，世界系下二维速度指令
    """

    # --------- 默认参数 ----------
    if params is None:
        params = {}
    dt = float(params.get("dt", 0.02))  # 例如 50Hz
    T_h = float(params.get("time_horizon", 2.0))  # seconds
    neighbor_dist = float(params.get("neighbor_dist", 5.0))  # meters
    radius = params.get("radius", 0.3)  # meters
    max_speed = params.get("max_speed", 1.5)  # m/s
    inflation = float(params.get("inflation", 0.0))

    agent_ids: List[str] = list(env_cfg.possible_agents)
    num_envs = state.shape[0]
    num_agents = len(agent_ids)

    # ------- 解析 state 的切片（按你给定的拼接顺序与常见维度） -------
    d_action = 2
    d_pos = 3
    d_goal = 3
    d_quat = 4
    d_rvel = 6  # 线速度(3)+角速度(3)
    stride = d_action + d_pos + d_goal + d_quat + d_rvel

    assert state.shape[1] == stride * num_agents, (
        f"state_dim={state.shape[1]} 与 num_agents={num_agents} 不匹配，期望 {stride * num_agents}"
    )

    def take_chunk(i, off, dim):
        s = i * stride + off
        return state[:, s : s + dim]

    # 逐 agent 取 xy 位置、xy 线速度、目标相对向量的 xy 分量
    pos_xy = torch.empty(num_envs, num_agents, 2, device=state.device, dtype=state.dtype)
    vel_xy = torch.empty_like(pos_xy)
    goal_rel_xy = torch.empty_like(pos_xy)

    for i in range(num_agents):
        pos = take_chunk(i, d_action, d_pos)  # (N,3)
        goal_rel = take_chunk(i, d_action + d_pos, d_goal)  # (N,3)
        root_vel = take_chunk(i, d_action + d_pos + d_goal + d_quat, d_rvel)  # (N,6)
        pos_xy[:, i] = pos[:, :2]
        vel_xy[:, i] = root_vel[:, :2]  # 仅用线速度的 xy
        goal_rel_xy[:, i] = goal_rel[:, :2]

    # ------- 参数广播到 [num_envs, num_agents] -------
    def to_tensor_like(x, ref):
        if isinstance(x, (int, float)):
            t = torch.full((num_envs, num_agents, 1), float(x), device=ref.device, dtype=ref.dtype)
        elif isinstance(x, torch.Tensor):
            x = x.to(ref)
            if x.ndim == 0:
                t = torch.full((num_envs, num_agents, 1), float(x.item()), device=ref.device, dtype=ref.dtype)
            elif x.ndim == 1:  # [num_agents]
                assert x.shape[0] == num_agents, "1D 参数长度应为 num_agents"
                t = x.view(1, num_agents, 1).expand(num_envs, num_agents, 1)
            elif x.ndim == 2:  # [num_envs, num_agents]
                assert x.shape == (num_envs, num_agents), "2D 参数应为 [num_envs, num_agents]"
                t = x.unsqueeze(-1)
            elif x.ndim == 3:
                assert x.shape == (num_envs, num_agents, 1), "3D 参数应为 [num_envs, num_agents, 1]"
                t = x
            else:
                raise ValueError("不支持的参数维度")
        else:
            raise TypeError("param must be float/int/Tensor")
        return t

    R = to_tensor_like(radius + inflation, pos_xy)  # (N,A,1)
    Vmax = to_tensor_like(max_speed, pos_xy)  # (N,A,1)
    invT = torch.as_tensor(1.0 / max(T_h, 1e-6), device=state.device, dtype=state.dtype)

    # ------- 偏好速度：指向目标；模长为 Vmax（你也可以改成靠近目标减速） -------
    # 数值稳定：对范数做 clamp_min，而不是给向量加 eps
    goal_norm = torch.norm(goal_rel_xy, dim=-1, keepdim=True).clamp_min(1e-12)
    pref_dir = goal_rel_xy / goal_norm
    pref_speed = Vmax
    pref_vel = pref_dir * pref_speed  # (N,A,2)

    # ------- 邻居掩码（同环境内的其它 agent，且距离小于 neighbor_dist） -------
    diff = pos_xy.unsqueeze(2) - pos_xy.unsqueeze(1)  # (N,A,A,2): i-j
    dist2 = torch.sum(diff * diff, dim=-1)           # (N,A,A)
    eye = torch.eye(num_agents, dtype=torch.bool, device=state.device).unsqueeze(0)
    neighbor_mask = (dist2 < neighbor_dist * neighbor_dist) & (~eye)

    # ------- 工具函数（2D） -------
    def det2(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    def normalize(v, eps=1e-8):
        n = torch.linalg.norm(v, dim=-1, keepdim=True)
        return v / torch.clamp(n, min=eps)

    # 结果缓冲
    new_vel = torch.empty_like(pref_vel)  # (N,A,2)

    # ========= 主循环：逐 env、逐 agent 解 ORCA =========
    for e in range(num_envs):
        # 该环境快照
        P = pos_xy[e]          # (A,2)
        V = vel_xy[e]          # (A,2)
        PV = pref_vel[e]       # (A,2)
        Re = R[e].squeeze(-1)  # (A,)
        Vmax_e = Vmax[e].squeeze(-1)  # (A,)
        neigh = neighbor_mask[e]      # (A,A)

        for i in range(num_agents):
            # ------- 构造 agent i 的 ORCA 线 -------
            points: List[torch.Tensor] = []
            dirs:   List[torch.Tensor] = []

            js = torch.nonzero(neigh[i], as_tuple=False).squeeze(-1).tolist()
            for j in js:
                rel_pos = P[j] - P[i]      # (2,)
                rel_vel = V[i] - V[j]      # (2,)
                dist2_ij = torch.dot(rel_pos, rel_pos).item()
                R_ij = Re[i].item() + Re[j].item()
                R2_ij = R_ij * R_ij

                if dist2_ij > R2_ij:
                    # --- 无碰撞：按时间视界 T_h ---
                    w = rel_vel - rel_pos * invT  # (2,)
                    w2 = torch.dot(w, w).item()
                    rp_dot_w = torch.dot(rel_pos, w).item()

                    if (rp_dot_w < 0.0) and (rp_dot_w * rp_dot_w > R2_ij * w2):
                        # 投影到截断圆
                        w_len = math.sqrt(max(w2, 0.0)) + 1e-12
                        unitW = w / w_len
                        line_dir = torch.tensor([unitW[1].item(), -unitW[0].item()],
                                                device=state.device, dtype=state.dtype)
                        u = (R_ij * invT - w_len) * unitW
                    else:
                        # 投影到左右腿
                        leg = math.sqrt(max(dist2_ij - R2_ij, 0.0)) + 1e-12
                        if det2(rel_pos, w).item() > 0.0:
                            # 左腿
                            line_dir = torch.tensor(
                                [(rel_pos[0] * leg - rel_pos[1] * R_ij) / dist2_ij,
                                 (rel_pos[0] * R_ij + rel_pos[1] * leg) / dist2_ij],
                                device=state.device, dtype=state.dtype
                            )
                        else:
                            # 右腿
                            line_dir = -torch.tensor(
                                [(rel_pos[0] * leg + rel_pos[1] * R_ij) / dist2_ij,
                                 (-rel_pos[0] * R_ij + rel_pos[1] * leg) / dist2_ij],
                                device=state.device, dtype=state.dtype
                            )

                        dot2 = torch.dot(rel_vel, line_dir).item()
                        u = dot2 * line_dir - rel_vel
                else:
                    # --- 已经穿透：用 timeStep (dt) 的更强制修正 ---
                    inv_dt = 1.0 / max(dt, 1e-6)
                    w = rel_vel - rel_pos * inv_dt
                    w_len = torch.linalg.norm(w).item() + 1e-12
                    unitW = w / w_len
                    line_dir = torch.tensor([unitW[1].item(), -unitW[0].item()],
                                            device=state.device, dtype=state.dtype)
                    u = (R_ij * inv_dt - w_len) * unitW

                # “各退一半”
                line_point = V[i] + 0.5 * u
                # 归一化方向（Line.direction）
                line_dir = normalize(line_dir.unsqueeze(0)).squeeze(0)

                points.append(line_point)
                dirs.append(line_dir)

            # ====== 线性规划：在半平面交集 ∩ 速度圆内，最接近偏好速度 ======

            # 半平面可行性：与 RVO2/C++ 一致的符号——注意这里是 "< 0.0" 代表违反
            def violates(k, v):
                # C++: violate if det(dir_k, point_k - v) > 0
                # 等价：det(dir_k, v - point_k) < 0 才违反
                return det2(dirs[k], v - points[k]).item() < 0.0

            # 带 direction-opt 的 LP1（等价 C++ linearProgram1）
            def lp1(lineNo, cur_res, radius, opt_v, direction_opt):
                p = points[lineNo]
                d = dirs[lineNo]
                dot_pd = torch.dot(p, d).item()
                disc = dot_pd * dot_pd + radius * radius - torch.dot(p, p).item()
                if disc < 0.0:
                    return False, cur_res

                sqrt_disc = math.sqrt(disc)
                tL, tR = -dot_pd - sqrt_disc, -dot_pd + sqrt_disc

                # 与之前线的可行区间取交
                for i_line in range(lineNo):
                    denom = det2(d, dirs[i_line]).item()
                    numer = det2(dirs[i_line], p - points[i_line]).item()
                    if abs(denom) <= 1e-8:
                        # 平行
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
                    # 方向最优：根据 opt_v 与 d 的点积决定取左右端点
                    if torch.dot(opt_v, d).item() > 0.0:
                        t_star = tR
                    else:
                        t_star = tL
                else:
                    # 最近点最优
                    t_star = torch.dot(d, (opt_v - p)).item()
                    if t_star < tL:
                        t_star = tL
                    elif t_star > tR:
                        t_star = tR

                return True, (p + t_star * d)

            # 带 direction-opt 的 LP2（等价 C++ linearProgram2）
            def lp2(points, dirs, radius, opt_v, direction_opt=False):
                # 初值
                if direction_opt:
                    # 方向优化模式：opt_v 视为单位方向
                    res = opt_v * radius
                else:
                    # 最近点模式
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

            # LP3（等价 C++ linearProgram3）
            def lp3(points, dirs, begin_idx, radius, cur_res):
                distance = 0.0
                for i_line in range(begin_idx, len(points)):
                    val = det2(dirs[i_line], points[i_line] - cur_res).item()
                    if val > distance:
                        # 构造投影线集（仅从 begin_idx 到 i_line-1）
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
                                proj_p = points[i_line] + (
                                    det2(dirs[j_line], points[i_line] - points[j_line]).item() / det_ij
                                ) * dirs[i_line]
                            proj_d = normalize((dirs[j_line] - dirs[i_line]).unsqueeze(0)).squeeze(0)
                            proj_points.append(proj_p)
                            proj_dirs.append(proj_d)

                        # 关键：direction_opt=True，目标方向取当前线的垂向
                        perp = torch.tensor(
                            [-dirs[i_line][1].item(), dirs[i_line][0].item()],
                            device=cur_res.device, dtype=cur_res.dtype
                        )
                        fail_k, cur_res_new = lp2(proj_points, proj_dirs, radius, perp, direction_opt=True)

                        # 理论上应成功；若失败（数值问题），保留旧解
                        cur_res = cur_res if fail_k < len(proj_points) else cur_res_new
                        distance = det2(dirs[i_line], points[i_line] - cur_res).item()
                return cur_res

            # 半径与上限
            vmax = float(Vmax_e[i].item())

            # 先 LP2（最近点模式）
            fail_idx, res_i = lp2(points, dirs, vmax, PV[i], direction_opt=False)
            if fail_idx < len(points):
                # 若失败，再 LP3（方向优化）
                res_i = lp3(points, dirs, fail_idx, vmax, res_i)

            new_vel[e, i] = res_i

    # 限幅（保险）
    speed = torch.norm(new_vel, dim=-1, keepdim=True)
    new_vel = torch.where(speed > Vmax, new_vel * (Vmax / (speed + 1e-12)), new_vel)

    # 组织为 env.step 需要的 dict
    actions: Dict[str, torch.Tensor] = {agent_ids[i]: new_vel[:, i, :] for i in range(num_agents)}
    return actions


def main():
    # Create environment configuration
    env_cfg = parse_env_cfg("FAST-Swarm-Vel", device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric)
    env_cfg.goal_reset_delay /= 1.5
    # Create environment
    env = gym.make("FAST-Swarm-Vel", cfg=env_cfg)
    env.reset()
    state = env.unwrapped.state()

    rvo_params = dict(
        dt=env.unwrapped.step_dt,
        time_horizon=0.5,
        neighbor_dist=5.0,
        radius=0.25,
        max_speed=10.0,
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
