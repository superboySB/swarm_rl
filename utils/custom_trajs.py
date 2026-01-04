import time
import torch
from loguru import logger
import matplotlib.pyplot as plt

from utils.minco import MinJerkOpt
from isaaclab.utils import configclass


@configclass
class LissajousConfig:
    A_range = [2.5, 5.0]  # X-axis amplitude
    B_range = [2.5, 5.0]  # Y-axis amplitude

    ratio = [
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 3),
        (3, 4),
        (2, 1),
        (3, 1),
        (3, 2),
        (4, 3),
    ]  # List of (a, b) frequency ratios
    delta_range = [0, 2 * torch.pi]

    num_pieces = 128  # Number of pieces (segments) in the trajectory
    max_exec_speed = 6.0  # Desired max execution speed (m/s)


def generate_custom_trajs(type_id="lissajous", p_odom=None, v_odom=None, a_odom=None, p_init=None, custom_cfg=None, is_plotting=False):
    if type_id == "lissajous":
        return generate_lissajous_trajs(
            p_odom=p_odom,
            v_odom=v_odom,
            a_odom=a_odom,
            p_init=p_init,
            custom_cfg=custom_cfg,
            is_plotting=is_plotting,
        )
    elif type_id == "eight":
        return generate_eight_trajs(
            p_odom=p_odom,
            v_odom=v_odom,
            a_odom=a_odom,
            p_init=p_init,
            custom_cfg=custom_cfg,
            is_plotting=is_plotting,
        )
    else:
        raise NotImplementedError(f"Trajectory type '{type_id}' is not implemented.")


def generate_lissajous_trajs(p_odom, v_odom, a_odom, p_init, custom_cfg: LissajousConfig | None = None, is_plotting=None):
    if custom_cfg is None:
        custom_cfg = LissajousConfig()

    device = p_odom.device
    head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)
    tail_pva = torch.stack([p_init, torch.zeros_like(p_init), torch.zeros_like(p_init)], dim=2)

    ratio_indices = torch.randint(0, len(custom_cfg.ratio), (p_odom.shape[0],), device=device)
    ratio_tensor = torch.tensor(custom_cfg.ratio, device=device, dtype=torch.float32)
    a_params = ratio_tensor[ratio_indices][:, 0]
    b_params = ratio_tensor[ratio_indices][:, 1]
    delta_params = torch.rand(p_odom.shape[0], device=device) * (custom_cfg.delta_range[1] - custom_cfg.delta_range[0]) + custom_cfg.delta_range[0]
    A_params = torch.rand(p_odom.shape[0], device=device) * (custom_cfg.A_range[1] - custom_cfg.A_range[0]) + custom_cfg.A_range[0]
    B_params = torch.rand(p_odom.shape[0], device=device) * (custom_cfg.B_range[1] - custom_cfg.B_range[0]) + custom_cfg.B_range[0]

    # Estimate curve lengths and adjust duration
    inner_pts, execution_durations, traj_info = compute_lissajous_inner_pts(
        p_odom=p_odom,
        p_init=p_init,
        A_params=A_params,
        B_params=B_params,
        a_params=a_params,
        b_params=b_params,
        delta_params=delta_params,
        num_pieces=custom_cfg.num_pieces,
        max_exec_speed=custom_cfg.max_exec_speed,
    )

    MJO = MinJerkOpt(head_pva, tail_pva, custom_cfg.num_pieces)
    start = time.perf_counter()
    MJO.generate(inner_pts, execution_durations)
    end = time.perf_counter()

    # Log trajectory information
    if is_plotting is True:
        for idx in range(p_odom.shape[0]):
            plot_inner_pts(inner_pts=inner_pts[idx], env_idx=idx, traj_info=traj_info[idx], save_path=None)

    return MJO.get_traj()


def generate_eight_trajs(p_odom, v_odom, a_odom, p_init, custom_cfg=None, is_plotting=None):
    num_pieces = 6

    head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)
    tail_pva = torch.stack([p_init, torch.zeros_like(p_init), torch.zeros_like(p_init)], dim=2)

    inner_pts = torch.zeros((p_odom.shape[0], 3, num_pieces - 1), device=p_odom.device)
    inner_pts[:, :, 0] = p_init + torch.tensor([3.0, -3.0, 0.0], device=p_odom.device)
    inner_pts[:, :, 1] = p_init + torch.tensor([3.0, 3.0, 0.0], device=p_odom.device)
    inner_pts[:, :, 2] = p_init + torch.tensor([0.0, 0.0, 0.0], device=p_odom.device)
    inner_pts[:, :, 3] = p_init + torch.tensor([-3.0, -3.0, 0.0], device=p_odom.device)
    inner_pts[:, :, 4] = p_init + torch.tensor([-3.0, 3.0, 0.0], device=p_odom.device)

    durations = torch.full((p_odom.shape[0], num_pieces), 2.0, device=p_odom.device)

    MJO = MinJerkOpt(head_pva, tail_pva, num_pieces)
    start = time.perf_counter()
    MJO.generate(inner_pts, durations)
    end = time.perf_counter()
    logger.trace(f"Eight trajectory generation takes {end - start:.5f}s")

    return MJO.get_traj()


def compute_lissajous_inner_pts(p_odom, p_init, A_params=None, B_params=None, a_params=None, b_params=None, delta_params=None, num_pieces=64, max_exec_speed=2.0):
    device = p_odom.device
    K = num_pieces - 1
    total_lengths, origin_max_speeds, origin_max_accels = compute_curve_properties(A_params, B_params, a_params, b_params, delta_params)

    # Uniformly sample t
    execution_times = 2 * torch.pi * origin_max_speeds / max_exec_speed  # [num_trajs]
    t = torch.linspace(0.0, 2 * torch.pi, steps=num_pieces + 1, device=device, dtype=p_init.dtype)
    t_expanded = t.unsqueeze(0).expand(p_odom.shape[0], -1)  # [num_trajs, num_pieces + 1]
    # Compute execution durations for each segment
    execution_durations = execution_times.unsqueeze(1).expand(-1, num_pieces) / num_pieces

    # Compute inner points
    A_expanded = A_params.unsqueeze(1)
    B_expanded = B_params.unsqueeze(1)
    a_expanded = a_params.unsqueeze(1)
    b_expanded = b_params.unsqueeze(1)
    delta_expanded = delta_params.unsqueeze(1)
    x = A_expanded * torch.sin(a_expanded * t_expanded + delta_expanded)
    y = B_expanded * torch.sin(b_expanded * t_expanded)

    center_x = torch.mean(x, dim=1, keepdim=True)
    center_y = torch.mean(y, dim=1, keepdim=True)
    x = x - center_x
    y = y - center_y
    z = torch.zeros_like(x)

    # [num_trajs, num_pieces + 1, 3] -> [num_trajs, 3, K]
    inner_pts = p_init.unsqueeze(2) + torch.stack([x, y, z], dim=2).transpose(1, 2)
    inner_pts = inner_pts[:, :, 1:-1]  # Remove start and end points

    first_sample_point = inner_pts[:, :, 0]
    last_sample_point = inner_pts[:, :, -1]
    distance_to_first = torch.linalg.norm(first_sample_point - p_init, dim=1)
    distance_from_last = torch.linalg.norm(p_init - last_sample_point, dim=1)
    time_to_first = distance_to_first / max_exec_speed * 3
    time_from_last = distance_from_last / max_exec_speed * 3

    # Set the first and last durations to a custom value
    execution_durations[:, 0] = time_to_first
    execution_durations[:, -1] = time_from_last

    # Generate trajectory information
    traj_info = []
    for i in range(p_odom.shape[0]):
        traj_info.append(
            {
                "total_length": total_lengths[i].item(),
                "execution_time": execution_times[i].item(),
                "num_inner_points": K,
                "origin_max_speed": origin_max_speeds[i].item(),
                "origin_max_accel": origin_max_accels[i].item(),
                "max_speed": max_exec_speed,
                "a": a_params[i].item(),
                "b": b_params[i].item(),
                "delta": delta_params[i].item(),
                "A": A_params[i].item(),
                "B": B_params[i].item(),
                "distance_to_first": distance_to_first[i].item(),
                "distance_from_last": distance_from_last[i].item(),
                "time_to_first": time_to_first[i].item(),
                "time_from_last": time_from_last[i].item(),
            }
        )

    return inner_pts, execution_durations, traj_info


def compute_curve_properties(A_params, B_params, a_params, b_params, delta_params, num_integration_points=10000):
    device = a_params.device

    t_high_res = torch.linspace(0, 2 * torch.pi, num_integration_points, device=device)  # [num_integration_points]
    t_high_res_expanded = t_high_res.unsqueeze(0).expand(a_params.shape[0], -1)
    A_expanded = A_params.unsqueeze(1)
    B_expanded = B_params.unsqueeze(1)
    a_expanded = a_params.unsqueeze(1)
    b_expanded = b_params.unsqueeze(1)
    delta_expanded = delta_params.unsqueeze(1)
    # velocity
    dx_dt = A_expanded * a_expanded * torch.cos(a_expanded * t_high_res_expanded + delta_expanded)
    dy_dt = B_expanded * b_expanded * torch.cos(b_expanded * t_high_res_expanded)
    speed = torch.sqrt(dx_dt**2 + dy_dt**2)
    # acceleration
    ddx_ddt = -A_expanded * a_expanded**2 * torch.sin(a_expanded * t_high_res_expanded + delta_expanded)
    ddy_ddt = -B_expanded * b_expanded**2 * torch.sin(b_expanded * t_high_res_expanded)
    accel = torch.sqrt(ddx_ddt**2 + ddy_ddt**2)

    dt = 2 * torch.pi / (num_integration_points - 1)
    cumulative_arc_lengths = torch.cumsum(speed * dt, dim=1)
    total_lengths = cumulative_arc_lengths[:, -1].unsqueeze(1)  # [num_trajs, 1]
    max_speeds = torch.max(speed, dim=1).values  # [num_trajs]
    max_accels = torch.max(accel, dim=1).values  # [num_trajs]

    return total_lengths, max_speeds, max_accels


def plot_inner_pts(inner_pts, traj_info, env_idx, save_path=None):
    pts = inner_pts.detach().cpu().numpy()  # (3, K)
    x, y, z = pts[0], pts[1], pts[2]
    K = len(x)

    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(121)
    ax1.plot(x, y, "-", linewidth=1, alpha=0.7, color='blue')
    ax1.scatter(x, y, s=30, c='red', zorder=5)

    for i in range(K):
        ax1.annotate(f'{i}', (x[i], y[i]), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, color='black', weight='bold')

    ax1.scatter(x[0], y[0], s=100, c='green', marker='s', label='Start', zorder=6)
    ax1.scatter(x[-1], y[-1], s=100, c='orange', marker='^', label='End', zorder=6)

    ax1.set_aspect("equal", "box")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_title("Lissajous (XY)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(122, projection="3d")
    ax2.plot(x, y, z, "-", linewidth=1, alpha=0.7, color='blue')
    ax2.scatter(x, y, z, s=20, c='red', zorder=5)

    step = max(1, K // 8) 
    for i in range(0, K, step):
        ax2.text(x[i], y[i], z[i], f'{i}', fontsize=7)

    ax2.scatter(x[0], y[0], z[0], s=60, c='green', marker='s', label='Start')
    ax2.scatter(x[-1], y[-1], z[-1], s=60, c='orange', marker='^', label='End')
    ax2.set_xlabel("x [m]")
    ax2.set_ylabel("y [m]")
    ax2.set_zlabel("z [m]")
    ax2.set_title(f"Trajectory {env_idx} | 3D View with Waypoints")
    ax2.legend()
    ax2.grid(True)

    # Add trajectory info text
    info_text = (
        f"Total Length: {traj_info['total_length']:.2f} m\n"
        f"Max Speed: {traj_info['max_speed']:.2f} m/s\n"
        f"Origin Max Speed: {traj_info['origin_max_speed']:.2f} m/s\n"
        f"Origin Max Accel: {traj_info['origin_max_accel']:.2f} m/s²\n"
        f"Lissajous Params: a={traj_info['a']:.2f}, b={traj_info['b']:.2f}, delta={traj_info['delta']:.2f}\n"
        f"Distance to First Point: {traj_info['distance_to_first']:.2f} m, Time: {traj_info['time_to_first']:.2f} s\n"
        f"Distance from Last Point: {traj_info['distance_from_last']:.2f} m, Time: {traj_info['time_from_last']:.2f} s\n"
    )
    fig.text(0.5, 0.01, info_text, ha="center", fontsize=6, wrap=True)

    plt.tight_layout(rect=[0, 0.1, 1, 0.95])
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    return fig, (ax1, ax2)
