import torch


@torch.jit.script
def quat_inv(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((quat[:, 0:1], -quat[:, 1:]), dim=1)


@torch.jit.script
def quat_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return torch.stack([w, x, y, z], dim=-1)


@torch.jit.script
def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    w = quat[:, 0:1]
    xyz = quat[:, 1:]
    t = 2 * torch.cross(xyz, vec, dim=1)
    return vec + w * t + torch.cross(xyz, t, dim=1)


@torch.jit.script
def quat_to_ang_between_z_body_and_z_world(quat: torch.Tensor) -> torch.Tensor:
    x, y = quat[:, 1], quat[:, 2]
    z_body_z = 1 - 2 * (x**2 + y**2)
    return torch.acos(torch.clamp(z_body_z, -1.0, 1.0))


@torch.jit.script
def quat_to_rot_mat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    R11 = w * w + x * x - y * y - z * z
    R12 = 2 * (x * y - w * z)
    R13 = 2 * (x * z + w * y)
    R21 = 2 * (x * y + w * z)
    R22 = w * w - x * x + y * y - z * z
    R23 = 2 * (y * z - w * x)
    R31 = 2 * (x * z - w * y)
    R32 = 2 * (y * z + w * x)
    R33 = w * w - x * x - y * y + z * z
    return torch.stack(
        [
            torch.stack((R11, R12, R13), dim=-1),
            torch.stack((R21, R22, R23), dim=-1),
            torch.stack((R31, R32, R33), dim=-1),
        ],
        dim=1,
    )


@torch.jit.script
def quat_to_yaw(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))
