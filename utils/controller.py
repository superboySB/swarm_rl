from loguru import logger
import math
import torch

from isaaclab.utils.math import quat_inv, quat_mul, quat_apply, matrix_from_quat, axis_angle_from_quat


class Controller:
    def __init__(self, step_dt: float, gravity: torch.Tensor, mass: torch.Tensor, inertia: torch.Tensor, num_envs: int):
        # Params
        self.kPp = torch.tensor([3.0, 3.0, 10.0], dtype=torch.float32, device=gravity.device)
        self.kPv = torch.tensor([7.5, 7.5, 50.0], dtype=torch.float32, device=gravity.device)
        self.kPR = torch.tensor([20.0, 20.0, 20.0], dtype=torch.float32, device=gravity.device)
        self.kPw = torch.tensor([0.03, 0.02, 0.03], dtype=torch.float32, device=gravity.device)
        self.kIw = torch.tensor([0.0, 0.0, 0.0], device=gravity.device)
        self.kDw = torch.tensor([0.0, 0.0, 0.0], device=gravity.device)

        self.w_error_integral_max = torch.tensor([0.1, 0.1, 0.1], device=gravity.device)
        self.w_error_integral = torch.zeros(num_envs, 3, device=gravity.device)
        self.w_error_prev = torch.zeros(num_envs, 3, device=gravity.device)

        self.K_min_norm_collec_acc = 3
        self.K_max_ang = 30
        self.K_max_bodyrates_feedback = 13
        self.K_max_angular_acc = 200

        self.step_dt = step_dt
        self.gravity = gravity.to(dtype=torch.float32)
        self.mass = mass.to(dtype=torch.float32)
        self.inertia = inertia.to(dtype=torch.float32)

        self.w_old = torch.zeros(num_envs, 3, dtype=torch.float32, device=self.gravity.device)
        self.thrust_old = torch.full((num_envs,), self.mass * self.gravity.norm(), dtype=torch.float32, device=self.gravity.device)
        self.w_last = torch.zeros(num_envs, 3, dtype=torch.float32, device=self.gravity.device)
        self.num_envs = num_envs

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self.w_old = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.gravity.device)
            self.thrust_old = torch.full((self.num_envs,), self.mass * self.gravity.norm(), dtype=torch.float32, device=self.gravity.device)
            self.w_last = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.gravity.device)
            self.w_error_integral = torch.zeros(self.num_envs, 3, device=self.gravity.device)
            self.w_error_prev = torch.zeros(self.num_envs, 3, device=self.gravity.device)
        else:
            self.w_old[env_ids] = torch.zeros(len(env_ids), 3, dtype=torch.float32, device=self.gravity.device)
            self.thrust_old[env_ids] = torch.full((len(env_ids),), self.mass * self.gravity.norm(), dtype=torch.float32, device=self.gravity.device)
            self.w_last[env_ids] = torch.zeros(len(env_ids), 3, dtype=torch.float32, device=self.gravity.device)
            self.w_error_integral[env_ids] = torch.zeros(len(env_ids), 3, device=self.gravity.device)
            self.w_error_prev[env_ids] = torch.zeros(len(env_ids), 3, device=self.gravity.device)

    def get_control(
        self, state_: torch.Tensor, action_: torch.Tensor, env_ids: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        state = state_.clone().to(dtype=torch.float32)
        action = action_.clone().to(dtype=torch.float32)

        p_odom = state[:, :3]
        q_odom = state[:, 3:7]
        v_odom = state[:, 7:10]
        w_odom_w = state[:, 10:13]
        w_odom = quat_apply(quat_inv(q_odom), w_odom_w)

        p_desired = action[:, :3]
        v_desired = action[:, 3:6]
        a_desired = action[:, 6:9]
        j_desired = action[:, 9:12]
        yaw_desired = action[:, 12]
        yaw_dot_desired = action[:, 13]

        pid_error_accelerations = compute_pid_error_acc(p_odom, v_odom, p_desired, v_desired, self.kPp, self.kPv)
        translational_acc = pid_error_accelerations + a_desired
        translational_acc = self.gravity + compute_limited_total_acc_from_thrust_force(
            translational_acc - self.gravity,
            torch.tensor(1.0, dtype=torch.float32, device=translational_acc.device),
            self.K_min_norm_collec_acc,
            self.K_max_ang,
        )

        if env_ids is None:
            thrust_desired, q_desired, w_desired = self.minimum_singularity_flat_with_drag(q_odom, v_desired, translational_acc, j_desired, yaw_desired, yaw_dot_desired)
        else:
            thrust_desired, q_desired, w_desired = self.minimum_singularity_flat_with_drag(
                q_odom, v_desired, translational_acc, j_desired, yaw_desired, yaw_dot_desired, env_ids
            )

        thrustforce = quat_apply(q_desired, thrust_desired.unsqueeze(1) * torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=thrust_desired.device))
        total_des_acc = compute_limited_total_acc_from_thrust_force(thrustforce, self.mass, self.K_min_norm_collec_acc, self.K_max_ang)
        force_desired = total_des_acc * self.mass

        w_desired = quat_apply(quat_inv(q_odom), quat_apply(q_desired, w_desired))
        feedback_bodyrates = compute_feedback_control_bodyrates(q_odom, q_desired, self.kPR, self.K_max_bodyrates_feedback)

        if env_ids is None:
            w_desired = compute_limited_angular_acc(w_desired + feedback_bodyrates, self.w_last, self.K_max_angular_acc, self.step_dt)
            self.w_last = w_desired
            thrust_desired, torque_desired = self.bodyrate_control(q_odom, w_odom, force_desired, w_desired)
        else:
            w_desired = compute_limited_angular_acc(w_desired + feedback_bodyrates, self.w_last[env_ids], self.K_max_angular_acc, self.step_dt)
            self.w_last[env_ids] = w_desired
            thrust_desired, torque_desired = self.bodyrate_control(q_odom, w_odom, force_desired, w_desired, env_ids)

        return total_des_acc, thrust_desired, q_desired, w_desired, torque_desired

    def minimum_singularity_flat_with_drag(
        self,
        q_odom: torch.Tensor,
        v_desired: torch.Tensor,
        a_desired: torch.Tensor,
        j_desired: torch.Tensor,
        yaw_desired: torch.Tensor,
        yaw_dot_desired: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ):
        # Drag effect parameters (Drag may cause larger tracking error in aggressive flight during our tests)
        # dv >= dh is required
        # dv is the rotor drag effect in vertical direction, typical value is 0.35
        # dh is the rotor drag effect in horizontal direction, typical value is 0.25
        # cp is the second-order drag effect, typical value is 0.01
        dh = 0.0
        dv = 0.0
        cp = 0.0

        # veps is a smoothing constant, do not change it
        veps = 0.02

        success, thrust_desired, q_desired, w_desired = flatness_with_drag(
            v_desired, a_desired, j_desired, yaw_desired, yaw_dot_desired, self.mass, self.gravity, cp, dv, dh, veps
        )

        if env_ids is None:
            thrust_desired = torch.where(success, thrust_desired, self.thrust_old)
            q_desired = torch.where(success.unsqueeze(1), q_desired, q_odom)
            w_desired = torch.where(success.unsqueeze(1), w_desired, self.w_old)

            self.thrust_old = torch.where(success, thrust_desired, self.thrust_old)
            self.w_old = torch.where(success.unsqueeze(1), w_desired, self.w_old)
        else:
            thrust_desired = torch.where(success, thrust_desired, self.thrust_old[env_ids])
            q_desired = torch.where(success.unsqueeze(1), q_desired, q_odom)
            w_desired = torch.where(success.unsqueeze(1), w_desired, self.w_old[env_ids])

            self.thrust_old[env_ids] = torch.where(success, thrust_desired, self.thrust_old[env_ids])
            self.w_old[env_ids] = torch.where(success.unsqueeze(1), w_desired, self.w_old[env_ids])

        failed_indices = torch.nonzero(success == False).squeeze()
        for i in failed_indices:
            logger.warning(f"Coner case in environment[{i}]: exactly inverted flight or unactuated falling 0_0")

        return thrust_desired, q_desired, w_desired

    def bodyrate_control(
        self,
        q_odom: torch.Tensor,
        w_odom: torch.Tensor,
        force_desired: torch.Tensor,
        w_desired: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        R = matrix_from_quat(q_odom)
        thrust_desired = (force_desired * R[:, :, 2]).sum(dim=1)

        I = self.inertia.view(3, 3)
        I_w_odom = torch.matmul(I, w_odom.transpose(0, 1))
        feedforward = torch.stack(
            [
                w_odom[:, 1] * I_w_odom[2, :] - w_odom[:, 2] * I_w_odom[1, :],
                w_odom[:, 2] * I_w_odom[0, :] - w_odom[:, 0] * I_w_odom[2, :],
                w_odom[:, 0] * I_w_odom[1, :] - w_odom[:, 1] * I_w_odom[0, :],
            ],
            dim=1,
        )
        w_err = w_desired - w_odom

        if env_ids is None:
            self.w_error_integral += w_err * self.step_dt
            self.w_error_integral = torch.max(torch.min(self.w_error_integral, self.w_error_integral_max), -self.w_error_integral_max)
            w_err_deriv = (w_err - self.w_error_prev) / self.step_dt
            self.w_error_prev = w_err.clone()

            torque_desired = feedforward + self.kPw * w_err + self.kIw * self.w_error_integral + self.kDw * w_err_deriv
        else:
            self.w_error_integral[env_ids] += w_err * self.step_dt
            self.w_error_integral[env_ids] = torch.max(torch.min(self.w_error_integral[env_ids], self.w_error_integral_max), -self.w_error_integral_max)
            w_err_deriv = (w_err - self.w_error_prev[env_ids]) / self.step_dt
            self.w_error_prev[env_ids] = w_err.clone()

            torque_desired = feedforward + self.kPw * w_err + self.kIw * self.w_error_integral[env_ids] + self.kDw * w_err_deriv

        return thrust_desired, torque_desired


@torch.jit.script
def compute_pid_error_acc(
    p_odom: torch.Tensor, v_odom: torch.Tensor, p_desired: torch.Tensor, v_desired: torch.Tensor, kPp: torch.Tensor, kPv: torch.Tensor
) -> torch.Tensor:

    # pos_error = torch.where(torch.isnan(p_desired), torch.zeros_like(p_desired, dtype=torch.float32), torch.clamp(p_desired - p_odom, -1.0, 1.0))
    pos_error = torch.where(torch.isnan(p_desired), torch.zeros_like(p_desired, dtype=torch.float32), p_desired - p_odom)

    # vel_error = torch.clamp((v_desired + kPp * pos_error) - v_odom, -1.0, 1.0)
    vel_error = (v_desired + kPp * pos_error) - v_odom

    acc_error = kPv * vel_error

    return acc_error


@torch.jit.script
def compute_limited_total_acc_from_thrust_force(thrustforce: torch.Tensor, mass: torch.Tensor, K_min_norm_collec_acc: float, K_max_ang: float) -> torch.Tensor:

    total_acc = thrustforce / mass

    # Limit magnitude
    norms = total_acc.norm(p=2, dim=1, keepdim=True) + 1e-6
    total_acc = torch.where(norms < K_min_norm_collec_acc, total_acc / norms * K_min_norm_collec_acc, total_acc)

    # Limit angle
    if K_max_ang > 0:
        z_acc = total_acc[:, 2]
        # Not allow too small z-force when angle limit is enabled
        z_acc = torch.where(z_acc < K_min_norm_collec_acc, K_min_norm_collec_acc, z_acc)

        z_B = total_acc / (total_acc.norm(p=2, dim=1, keepdim=True) + 1e-6)
        unit_z = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32, device=total_acc.device).expand(z_B.shape[0], 3)
        rot_axis = torch.cross(unit_z, z_B, dim=1)
        rot_axis = rot_axis / (rot_axis.norm(p=2, dim=1, keepdim=True) + 1e-6)

        rot_ang = torch.acos(z_B[:, 2].clamp(-1.0, 1.0))
        K_max_ang = math.radians(K_max_ang)
        mask = rot_ang > K_max_ang
        # Exceed the angle limit
        if mask.any():
            limited_z_B = unit_z * math.cos(K_max_ang) + torch.cross(rot_axis, unit_z, dim=1) * math.sin(K_max_ang)

            new_total_acc = (z_acc / math.cos(K_max_ang)).unsqueeze(1) * limited_z_B
            total_acc = torch.where(mask.unsqueeze(1), new_total_acc, total_acc)

    return total_acc


@torch.jit.script
def flatness_with_drag(
    v_desired: torch.Tensor,
    a_desired: torch.Tensor,
    j_desired: torch.Tensor,
    yaw_desired: torch.Tensor,
    yaw_dot_desired: torch.Tensor,
    mass: torch.Tensor,
    gravity: torch.Tensor,
    cp: float,
    dv: float,
    dh: float,
    veps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    almost_zero = 1e-6

    v0, v1, v2 = v_desired[:, 0], v_desired[:, 1], v_desired[:, 2]
    a0, a1, a2 = a_desired[:, 0], a_desired[:, 1], a_desired[:, 2]
    cp_term = torch.sqrt(v0**2 + v1**2 + v2**2 + veps)
    w_term = 1.0 + cp * cp_term
    w0 = w_term * v0
    w1 = w_term * v1
    w2 = w_term * v2
    dh_over_m = dh / mass
    zu0 = a0 + dh_over_m * w0
    zu1 = a1 + dh_over_m * w1
    zu2 = a2 + dh_over_m * w2 - gravity[2]
    zu_sqr0 = zu0 * zu0
    zu_sqr1 = zu1 * zu1
    zu_sqr2 = zu2 * zu2
    zu01 = zu0 * zu1
    zu12 = zu1 * zu2
    zu02 = zu0 * zu2
    zu_sqr_norm = zu_sqr0 + zu_sqr1 + zu_sqr2
    zu_norm = torch.sqrt(zu_sqr_norm)
    check1 = zu_norm < almost_zero
    safe_zu_norm = torch.where(check1, torch.ones_like(zu_norm), zu_norm)
    z0 = zu0 / safe_zu_norm
    z1 = zu1 / safe_zu_norm
    z2 = zu2 / safe_zu_norm
    ng_den = zu_sqr_norm * safe_zu_norm
    ng00 = (zu_sqr1 + zu_sqr2) / ng_den
    ng01 = -zu01 / ng_den
    ng02 = -zu02 / ng_den
    ng11 = (zu_sqr0 + zu_sqr2) / ng_den
    ng12 = -zu12 / ng_den
    ng22 = (zu_sqr0 + zu_sqr1) / ng_den
    v_dot_a = v0 * a0 + v1 * a1 + v2 * a2
    dw_term = cp * v_dot_a / cp_term
    dw0 = w_term * a0 + dw_term * v0
    dw1 = w_term * a1 + dw_term * v1
    dw2 = w_term * a2 + dw_term * v2
    dz_term0 = j_desired[:, 0] + dh_over_m * dw0
    dz_term1 = j_desired[:, 1] + dh_over_m * dw1
    dz_term2 = j_desired[:, 2] + dh_over_m * dw2
    dz0 = ng00 * dz_term0 + ng01 * dz_term1 + ng02 * dz_term2
    dz1 = ng01 * dz_term0 + ng11 * dz_term1 + ng12 * dz_term2
    dz2 = ng02 * dz_term0 + ng12 * dz_term1 + ng22 * dz_term2
    f_term0 = mass * a0 + dv * w0
    f_term1 = mass * a1 + dv * w1
    f_term2 = mass * (a2 - gravity[2]) + dv * w2
    thr = z0 * f_term0 + z1 * f_term1 + z2 * f_term2
    check2 = (1.0 + z2) < almost_zero
    success = torch.logical_not(torch.logical_or(check1, check2))
    safe_tilt_den = torch.where((1.0 + z2) < almost_zero, torch.ones_like(z2), torch.sqrt(2.0 * (1.0 + z2)))
    tilt0 = 0.5 * safe_tilt_den
    tilt1 = -z1 / safe_tilt_den
    tilt2 = z0 / safe_tilt_den
    c_half_psi = torch.cos(0.5 * yaw_desired)
    s_half_psi = torch.sin(0.5 * yaw_desired)
    quat = torch.zeros(v_desired.size(0), 4, dtype=torch.float32, device=v_desired.device)
    quat[:, 0] = tilt0 * c_half_psi
    quat[:, 1] = tilt1 * c_half_psi + tilt2 * s_half_psi
    quat[:, 2] = tilt2 * c_half_psi - tilt1 * s_half_psi
    quat[:, 3] = tilt0 * s_half_psi
    c_psi = torch.cos(yaw_desired)
    s_psi = torch.sin(yaw_desired)
    omg_den = z2 + 1.0
    safe_omg_den = torch.where(torch.abs(omg_den) < almost_zero, torch.ones_like(omg_den), omg_den)
    omg_term = dz2 / safe_omg_den
    omg = torch.zeros_like(v_desired, dtype=torch.float32)
    omg[:, 0] = dz0 * s_psi - dz1 * c_psi - (z0 * s_psi - z1 * c_psi) * omg_term
    omg[:, 1] = dz0 * c_psi + dz1 * s_psi - (z0 * c_psi + z1 * s_psi) * omg_term
    omg[:, 2] = (z1 * dz0 - z0 * dz1) / omg_den + yaw_dot_desired

    thr = torch.where(success, thr, torch.zeros_like(thr))
    quat = torch.where(success.unsqueeze(1), quat, torch.zeros_like(quat))
    omg = torch.where(success.unsqueeze(1), omg, torch.zeros_like(omg))

    return success, thr, quat, omg


@torch.jit.script
def compute_feedback_control_bodyrates(q_odom: torch.Tensor, q_desired: torch.Tensor, kPR: torch.Tensor, K_max_bodyrates_feedback: float):

    q_err = quat_mul(quat_inv(q_odom), q_desired)
    axis_angle_err = axis_angle_from_quat(q_err)
    bodyrates = kPR * axis_angle_err
    bodyrates = torch.clamp(bodyrates, min=-K_max_bodyrates_feedback, max=K_max_bodyrates_feedback)

    return bodyrates


@torch.jit.script
def compute_limited_angular_acc(candidate_bodyrate: torch.Tensor, w_last: torch.Tensor, K_max_angular_acc: float, step_dt: float):
    max_delta_bodyrate = K_max_angular_acc * step_dt
    w_delta = candidate_bodyrate - w_last

    bodyrate_out = torch.where(
        w_delta > max_delta_bodyrate,
        w_last + max_delta_bodyrate,
        torch.where(
            w_delta < -max_delta_bodyrate,
            w_last - max_delta_bodyrate,
            candidate_bodyrate,
        ),
    )
    return bodyrate_out


@torch.jit.script
def bodyrate_control_without_thrust(
    w_odom: torch.Tensor,
    w_desired: torch.Tensor,
    inertia: torch.Tensor,
    kPw: torch.Tensor,
) -> torch.Tensor:

    I = inertia.view(3, 3)
    I_w_odom = torch.matmul(I, w_odom.transpose(0, 1))
    feedforward = torch.stack(
        [
            w_odom[:, 1] * I_w_odom[2, :] - w_odom[:, 2] * I_w_odom[1, :],
            w_odom[:, 2] * I_w_odom[0, :] - w_odom[:, 0] * I_w_odom[2, :],
            w_odom[:, 0] * I_w_odom[1, :] - w_odom[:, 1] * I_w_odom[0, :],
        ],
        dim=1,
    )
    torque_desired = feedforward + kPw * (w_desired - w_odom)

    return torque_desired
