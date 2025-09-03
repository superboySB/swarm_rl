import time
import torch
from loguru import logger
import matplotlib.pyplot as plt

from utils.minco import MinJerkOpt
from isaaclab.utils import configclass


@configclass
class LissajousConfig:
    A = 3.0  # X-axis amplitude
    B = 3.0  # Y-axis amplitude

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
    delta = [
        0,
        torch.pi / 4,
        torch.pi / 2,
        3 * torch.pi / 4,
        torch.pi,
        5 * torch.pi / 4,
        3 * torch.pi / 2,
        7 * torch.pi / 4,
    ]  # Phase shifts

    num_pieces = 128  # Number of pieces (segments) in the trajectory
    max_exec_speed = 2  # Desired max execution speed (m/s)
    head_tail_buffer = 1.0  # Time buffer at start and end of trajectory (s)


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
        pass
    else:
        raise NotImplementedError(f"Trajectory type '{type_id}' is not implemented.")


def generate_lissajous_trajs(p_odom, v_odom, a_odom, p_init, custom_cfg: LissajousConfig | None = None, is_plotting=None):
    if custom_cfg is None:
        custom_cfg = LissajousConfig()

    device = p_odom.device
    head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)
    tail_pva = torch.stack([p_init, torch.zeros_like(p_init), torch.zeros_like(p_init)], dim=2)

    ratio_indices = torch.randint(0, len(custom_cfg.ratio), (p_odom.shape[0],), device=device)
    delta_indices = torch.randint(0, len(custom_cfg.delta), (p_odom.shape[0],), device=device)
    ratio_tensor = torch.tensor(custom_cfg.ratio, device=device, dtype=torch.float32)
    delta_tensor = torch.tensor(custom_cfg.delta, device=device, dtype=torch.float32)
    a_params = ratio_tensor[ratio_indices][:, 0]
    b_params = ratio_tensor[ratio_indices][:, 1]
    delta_params = delta_tensor[delta_indices]

    # Estimate curve lengths and adjust duration
    inner_pts, execution_durations, traj_info = compute_lissajous_inner_pts(
        p_odom=p_odom,
        p_init=p_init,
        A=custom_cfg.A,
        B=custom_cfg.B,
        a_params=a_params,
        b_params=b_params,
        delta_params=delta_params,
        num_pieces=custom_cfg.num_pieces,
        max_exec_speed=custom_cfg.max_exec_speed,
        head_tail_buffer=custom_cfg.head_tail_buffer,
    )

    MJO = MinJerkOpt(head_pva, tail_pva, custom_cfg.num_pieces)
    start = time.perf_counter()
    MJO.generate(inner_pts, execution_durations)
    end = time.perf_counter()

    # Log trajectory information
    if is_plotting is True:
        for idx in range(p_odom.shape[0]):
            for i, info in enumerate(traj_info):
                logger.info(
                    f"Trajectory {i}: "
                    f"Length={info['total_length']:.2f}m, "
                    f"ExecTime={info['execution_time']:.2f}s, "
                    f"MaxSpeed={info['max_speed']:.2f}m/s, "
                    f"InnerPts={info['num_inner_points']}, "
                    f"a={info['a']:.1f}, b={info['b']:.1f}, delta={info['delta']:.2f}"
                )
            plot_inner_pts(inner_pts, env_idx=idx)

    return MJO.get_traj()


def compute_lissajous_inner_pts(p_odom, p_init, A=3.0, B=3.0, a_params=None, b_params=None, delta_params=None, num_pieces=64, max_exec_speed=2.0, head_tail_buffer=0.8):
    device = p_odom.device
    K = num_pieces - 1
    total_lengths, max_speeds = compute_curve_properties(A, B, a_params, b_params, delta_params)

    # Uniformly sample t
    execution_times = 2 * torch.pi * max_speeds / max_exec_speed  # [num_trajs]
    t = torch.linspace(0.0, 2 * torch.pi, steps=num_pieces + 1, device=device, dtype=p_init.dtype)
    t_expanded = t.unsqueeze(0).expand(p_odom.shape[0], -1)  # [num_trajs, num_pieces + 1]
    # Compute execution durations for each segment
    execution_durations = execution_times.unsqueeze(1).expand(-1, num_pieces) / num_pieces
    # Set the first and last durations to a custom value
    execution_durations[:, 0] = execution_durations[:, -1] = head_tail_buffer

    # Compute inner points
    a_expanded = a_params.unsqueeze(1)
    b_expanded = b_params.unsqueeze(1)
    delta_expanded = delta_params.unsqueeze(1)
    x = A * torch.sin(a_expanded * t_expanded + delta_expanded)
    y = B * torch.sin(b_expanded * t_expanded)
    z = torch.zeros_like(x)
    start_x = A * torch.sin(delta_expanded)
    start_y = torch.zeros_like(start_x)
    x = x - start_x
    y = y - start_y

    # [num_trajs, num_pieces + 1, 3] -> [num_trajs, 3, K]
    inner_pts = p_init.unsqueeze(2) + torch.stack([x, y, z], dim=2).transpose(1, 2)
    inner_pts = inner_pts[:, :, 1:-1]  # Remove start and end points

    # Generate trajectory information
    traj_info = []
    for i in range(p_odom.shape[0]):
        traj_info.append(
            {
                "total_length": total_lengths[i].item(),
                "execution_time": execution_times[i].item(),
                "num_inner_points": K,
                "max_speed": max_speeds[i].item(),
                "a": a_params[i].item(),
                "b": b_params[i].item(),
                "delta": delta_params[i].item(),
            }
        )

    return inner_pts, execution_durations, traj_info


def compute_curve_properties(A, B, a_params, b_params, delta_params, num_integration_points=10000):
    device = a_params.device

    t_high_res = torch.linspace(0, 2 * torch.pi, num_integration_points, device=device)  # [num_integration_points]
    t_high_res_expanded = t_high_res.unsqueeze(0).expand(a_params.shape[0], -1)
    a_expanded = a_params.unsqueeze(1)
    b_expanded = b_params.unsqueeze(1)
    delta_expanded = delta_params.unsqueeze(1)
    dx_dt = A * a_expanded * torch.cos(a_expanded * t_high_res_expanded + delta_expanded)
    dy_dt = B * b_expanded * torch.cos(b_expanded * t_high_res_expanded)
    speed = torch.sqrt(dx_dt**2 + dy_dt**2)

    dt = 2 * torch.pi / (num_integration_points - 1)
    cumulative_arc_lengths = torch.cumsum(speed * dt, dim=1)
    total_lengths = cumulative_arc_lengths[:, -1].unsqueeze(1)  # [num_trajs, 1]
    max_speeds = torch.max(speed, dim=1).values  # [num_trajs]

    return total_lengths, max_speeds


def plot_inner_pts(inner_pts: torch.Tensor, env_idx: int = 0, figsize=(10, 5), save_path: str | None = None):
    pts = inner_pts[env_idx].detach().cpu().numpy()  # (3, K)
    x, y, z = pts[0], pts[1], pts[2]
    K = len(x)

    fig = plt.figure(figsize=figsize)

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

    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("z")
    ax2.set_title("Lissajous (3D)")
    ax2.legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()

    return fig, (ax1, ax2)


# return generate_eight_trajs(
#     p_odom=p_odom,
#     v_odom=v_odom,
#     a_odom=a_odom,
#     p_init=p_init,
#     custom_cfg=custom_cfg,
#     plot=is_plotting,
# )

# def generate_eight_trajs(p_odom, v_odom, a_odom, p_init, custom_cfg=None, plot=None):
#     head_pva = torch.stack([p_odom, v_odom, a_odom], dim=2)
#     tail_pva = torch.stack([p_init, torch.zeros_like(p_init), torch.zeros_like(p_init)], dim=2)

#     inner_pts = torch.zeros((p_odom.shape[0], 3, custom_cfg.num_samples - 1), device=p_odom.device)
#     inner_pts[:, :, 0] = p_init + torch.tensor([3.0, -3.0, 0.0], device=p_odom.device)
#     inner_pts[:, :, 1] = p_init + torch.tensor([3.0, 3.0, 0.0], device=p_odom.device)
#     inner_pts[:, :, 2] = p_init + torch.tensor([0.0, 0.0, 0.0], device=p_odom.device)
#     inner_pts[:, :, 3] = p_init + torch.tensor([-3.0, -3.0, 0.0], device=p_odom.device)
#     inner_pts[:, :, 4] = p_init + torch.tensor([-3.0, 3.0, 0.0], device=p_odom.device)

#     durations = torch.full((p_odom.shape[0], custom_cfg.num_samples), 2.0, device=p_odom.device)

#     MJO = MinJerkOpt(head_pva, tail_pva, custom_cfg.num_samples)
#     start = time.perf_counter()
#     MJO.generate(inner_pts, durations)
#     end = time.perf_counter()
#     logger.trace(f"Eight trajectory generation takes {end - start:.5f}s")

#     return MJO.get_traj()
