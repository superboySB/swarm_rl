import torch


@torch.jit.script
def quat_to_ang_between_z_body_and_z_world(quat: torch.Tensor) -> torch.Tensor:
    x, y = quat[:, 1], quat[:, 2]
    z_body_z = 1 - 2 * (x**2 + y**2)
    return torch.acos(torch.clamp(z_body_z, -1.0, 1.0))


@torch.jit.script
def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
