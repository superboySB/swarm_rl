from loguru import logger
import torch


class Trajectory:
    def __init__(self, durs=None, coeff_mats_=None):
        self.num_env, self.N = durs.shape
        self.durations = durs
        self.coeff_mats = coeff_mats_

    def __setitem__(self, index, traj_new):
        self.durations[index] = traj_new.durations
        self.coeff_mats[index] = traj_new.coeff_mats

    def get_total_duration(self):
        return self.durations.sum(dim=1)

    def locate_piece_idx(self, t):
        cum_durations = torch.cat([torch.zeros(self.num_env, 1, device=self.durations.device), self.durations.cumsum(dim=1)], dim=1)
        idx = torch.searchsorted(cum_durations, t.unsqueeze(1)) - 1
        idx = idx.clamp(0, self.N - 1).squeeze(1)
        t_local = t - cum_durations[range(self.num_env), idx]

        mask = t_local > self.durations[range(self.num_env), idx] + 1e-10
        if mask.any():
            for env in torch.where(mask)[0]:
                logger.error(f"Retrieved timestamp out of trajectory duration in environment {env.item()} #^#")

        return idx, t_local

    def get_pos(self, t):
        idx, t_local = self.locate_piece_idx(t)
        pos = torch.zeros(self.num_env, 3, dtype=torch.float32, device=self.coeff_mats.device)
        tn = torch.ones(self.num_env, dtype=torch.float32, device=self.coeff_mats.device)

        coeff = self.coeff_mats[range(self.num_env), idx]
        for i in range(5, -1, -1):
            pos += tn.unsqueeze(1) * coeff[:, :, i]
            tn *= t_local
        return pos

    def get_vel(self, t):
        idx, t_local = self.locate_piece_idx(t)
        vel = torch.zeros(self.num_env, 3, dtype=torch.float32, device=self.coeff_mats.device)
        tn = torch.ones(self.num_env, dtype=torch.float32, device=self.coeff_mats.device)
        n = 1

        coeff = self.coeff_mats[range(self.num_env), idx]
        for i in range(4, -1, -1):
            vel += n * tn.unsqueeze(1) * coeff[:, :, i]
            tn *= t_local
            n += 1
        return vel

    def get_acc(self, t):
        idx, t_local = self.locate_piece_idx(t)
        acc = torch.zeros(self.num_env, 3, dtype=torch.float32, device=self.coeff_mats.device)
        tn = torch.ones(self.num_env, dtype=torch.float32, device=self.coeff_mats.device)
        m = 1
        n = 2

        coeff = self.coeff_mats[range(self.num_env), idx]
        for i in range(3, -1, -1):
            acc += m * n * tn.unsqueeze(1) * coeff[:, :, i]
            tn *= t_local
            m += 1
            n += 1
        return acc

    def get_jer(self, t):
        idx, t_local = self.locate_piece_idx(t)
        jer = torch.zeros(self.num_env, 3, dtype=torch.float32, device=self.coeff_mats.device)
        tn = torch.ones(self.num_env, dtype=torch.float32, device=self.coeff_mats.device)
        l = 1
        m = 2
        n = 3

        coeff = self.coeff_mats[range(self.num_env), idx]
        for i in range(2, -1, -1):
            jer += l * m * n * tn.unsqueeze(1) * coeff[:, :, i]
            tn *= t_local
            l += 1
            m += 1
            n += 1
        return jer


class BandedSystem:
    def __init__(self, num_env, n, p, q, device):
        self.N = n
        self.lower_bw = p
        self.upper_bw = q
        self.mat_data = torch.zeros(num_env, self.N * (self.lower_bw + self.upper_bw + 1), dtype=torch.float32, device=device)

    def __call__(self, i, j):
        return self.mat_data[:, (i - j + self.upper_bw) * self.N + j]

    def factorizeLU(self):
        eps = 1e-10
        for k in range(self.N - 1):
            iM = min(k + self.lower_bw, self.N - 1)
            cVl = self(k, k)
            cVl[torch.abs(cVl) < eps] += eps
            for i in range(k + 1, iM + 1):
                mask = self(i, k) != 0.0
                if mask.any():
                    self(i, k)[mask] /= cVl[mask]

            jM = min(k + self.upper_bw, self.N - 1)
            for j in range(k + 1, jM + 1):
                cVl = self(k, j)
                mask = cVl != 0.0
                if mask.any():
                    for i in range(k + 1, iM + 1):
                        mask_ = (self(i, k) != 0.0) & mask
                        if mask_.any():
                            self(i, j)[mask_] -= self(i, k)[mask_] * cVl[mask_]

    def solve(self, b):
        for j in range(self.N):
            iM = min(j + self.lower_bw, self.N - 1)
            for i in range(j + 1, iM + 1):
                mask = self(i, j) != 0.0
                if mask.any():
                    b[mask, i] -= self(i, j)[mask].unsqueeze(1) * b[mask, j]

        eps = 1e-10
        for j in range(self.N - 1, -1, -1):
            self(j, j)[torch.abs(self(j, j)) < eps] += eps
            b[:, j] /= self(j, j).unsqueeze(1)
            iM = max(0, j - self.upper_bw)
            for i in range(iM, j):
                mask = self(i, j) != 0.0
                if mask.any():
                    b[mask, i] -= self(i, j)[mask].unsqueeze(1) * b[mask, j]


class MinJerkOpt:
    def __init__(self, head_pva_, tail_pva_, num_pieces):
        self.num_env = head_pva_.shape[0]
        self.device = head_pva_.device

        self.N = num_pieces
        self.head_pva = head_pva_.clone().to(dtype=torch.float32)
        self.tail_pva = tail_pva_.clone().to(dtype=torch.float32)

    def generate(self, inner_pts_, durations):
        inner_pts = inner_pts_.clone().to(dtype=torch.float32)
        self.t1 = durations.clone().to(dtype=torch.float32)

        if inner_pts.shape[1] == 0:
            t1_inv = 1.0 / self.t1
            t2_inv = t1_inv * t1_inv
            t3_inv = t2_inv * t1_inv
            t4_inv = t2_inv * t2_inv
            t5_inv = t4_inv * t1_inv
            coeff_mat_reversed = torch.zeros(self.num_env, 3, 6, dtype=torch.float32, device=self.device)
            coeff_mat_reversed[:, :, 5] = (
                0.5 * (self.tail_pva[:, :, 2] - self.head_pva[:, :, 2]) * t3_inv
                - 3.0 * (self.head_pva[:, :, 1] + self.tail_pva[:, :, 1]) * t4_inv
                + 6.0 * (self.tail_pva[:, :, 0] - self.head_pva[:, :, 0]) * t5_inv
            )
            coeff_mat_reversed[:, :, 4] = (
                (-self.tail_pva[:, :, 2] + 1.5 * self.head_pva[:, :, 2]) * t2_inv
                + (8.0 * self.head_pva[:, :, 1] + 7.0 * self.tail_pva[:, :, 1]) * t3_inv
                + 15.0 * (-self.tail_pva[:, :, 0] + self.head_pva[:, :, 0]) * t4_inv
            )
            coeff_mat_reversed[:, :, 3] = (
                (0.5 * self.tail_pva[:, :, 2] - 1.5 * self.head_pva[:, :, 2]) * t1_inv
                - (6.0 * self.head_pva[:, :, 1] + 4.0 * self.tail_pva[:, :, 1]) * t2_inv
                + 10.0 * (self.tail_pva[:, :, 0] - self.head_pva[:, :, 0]) * t3_inv
            )
            coeff_mat_reversed[:, :, 2] = 0.5 * self.head_pva[:, :, 2]
            coeff_mat_reversed[:, :, 1] = self.head_pva[:, :, 1]
            coeff_mat_reversed[:, :, 0] = self.head_pva[:, :, 0]
            self.b = coeff_mat_reversed.transpose(1, 2)
        else:
            t2 = self.t1 * self.t1
            t3 = t2 * self.t1
            t4 = t2 * t2
            t5 = t4 * self.t1

            A = BandedSystem(self.num_env, 6 * self.N, 6, 6, device=self.device)
            self.b = torch.zeros(self.num_env, 6 * self.N, 3, dtype=torch.float32, device=self.device)

            A(0, 0)[:] = 1.0
            A(1, 1)[:] = 1.0
            A(2, 2)[:] = 2.0
            self.b[:, 0] = self.head_pva[:, :, 0]
            self.b[:, 1] = self.head_pva[:, :, 1]
            self.b[:, 2] = self.head_pva[:, :, 2]

            for i in range(self.N - 1):
                A(6 * i + 3, 6 * i + 3)[:] = 6.0
                A(6 * i + 3, 6 * i + 4)[:] = 24.0 * self.t1[:, i]
                A(6 * i + 3, 6 * i + 5)[:] = 60.0 * t2[:, i]
                A(6 * i + 3, 6 * i + 9)[:] = -6.0
                A(6 * i + 4, 6 * i + 4)[:] = 24.0
                A(6 * i + 4, 6 * i + 5)[:] = 120.0 * self.t1[:, i]
                A(6 * i + 4, 6 * i + 10)[:] = -24.0
                A(6 * i + 5, 6 * i)[:] = 1.0
                A(6 * i + 5, 6 * i + 1)[:] = self.t1[:, i]
                A(6 * i + 5, 6 * i + 2)[:] = t2[:, i]
                A(6 * i + 5, 6 * i + 3)[:] = t3[:, i]
                A(6 * i + 5, 6 * i + 4)[:] = t4[:, i]
                A(6 * i + 5, 6 * i + 5)[:] = t5[:, i]
                A(6 * i + 6, 6 * i)[:] = 1.0
                A(6 * i + 6, 6 * i + 1)[:] = self.t1[:, i]
                A(6 * i + 6, 6 * i + 2)[:] = t2[:, i]
                A(6 * i + 6, 6 * i + 3)[:] = t3[:, i]
                A(6 * i + 6, 6 * i + 4)[:] = t4[:, i]
                A(6 * i + 6, 6 * i + 5)[:] = t5[:, i]
                A(6 * i + 6, 6 * i + 6)[:] = -1.0
                A(6 * i + 7, 6 * i + 1)[:] = 1.0
                A(6 * i + 7, 6 * i + 2)[:] = 2 * self.t1[:, i]
                A(6 * i + 7, 6 * i + 3)[:] = 3 * t2[:, i]
                A(6 * i + 7, 6 * i + 4)[:] = 4 * t3[:, i]
                A(6 * i + 7, 6 * i + 5)[:] = 5 * t4[:, i]
                A(6 * i + 7, 6 * i + 7)[:] = -1.0
                A(6 * i + 8, 6 * i + 2)[:] = 2.0
                A(6 * i + 8, 6 * i + 3)[:] = 6 * self.t1[:, i]
                A(6 * i + 8, 6 * i + 4)[:] = 12 * t2[:, i]
                A(6 * i + 8, 6 * i + 5)[:] = 20 * t3[:, i]
                A(6 * i + 8, 6 * i + 8)[:] = -2.0
                self.b[:, 6 * i + 5] = inner_pts[:, :, i]

            A(6 * self.N - 3, 6 * self.N - 6)[:] = 1.0
            A(6 * self.N - 3, 6 * self.N - 5)[:] = self.t1[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 4)[:] = t2[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 3)[:] = t3[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 2)[:] = t4[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 1)[:] = t5[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 5)[:] = 1.0
            A(6 * self.N - 2, 6 * self.N - 4)[:] = 2 * self.t1[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 3)[:] = 3 * t2[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 2)[:] = 4 * t3[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 1)[:] = 5 * t4[:, self.N - 1]
            A(6 * self.N - 1, 6 * self.N - 4)[:] = 2.0
            A(6 * self.N - 1, 6 * self.N - 3)[:] = 6 * self.t1[:, self.N - 1]
            A(6 * self.N - 1, 6 * self.N - 2)[:] = 12 * t2[:, self.N - 1]
            A(6 * self.N - 1, 6 * self.N - 1)[:] = 20 * t3[:, self.N - 1]

            self.b[:, 6 * self.N - 3] = self.tail_pva[:, :, 0]
            self.b[:, 6 * self.N - 2] = self.tail_pva[:, :, 1]
            self.b[:, 6 * self.N - 1] = self.tail_pva[:, :, 2]

            A.factorizeLU()
            A.solve(self.b)

    def get_traj(self):
        blocks = self.b.view(self.num_env, self.N, 6, 3)
        coeff_mats = blocks.permute(0, 1, 3, 2).flip(3)
        return Trajectory(self.t1, coeff_mats)
