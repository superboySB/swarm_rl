from loguru import logger
import torch


class Trajectory:
    def __init__(self, durs=None, cMats=None):
        self.num_env, self.N = durs.shape
        self.durations = durs
        self.coeffMats = cMats

    def getTotalDuration(self):
        return self.durations.sum(dim=1)

    def __setitem__(self, index, traj_new):
        self.durations[index] = traj_new.durations
        self.coeffMats[index] = traj_new.coeffMats

    def locatePieceIdx(self, t):
        cum_durations = torch.cat([torch.zeros(self.num_env, 1, device=self.durations.device), self.durations.cumsum(dim=1)], dim=1)  # (num_env, N+1)
        idx = torch.searchsorted(cum_durations, t.unsqueeze(1)) - 1
        idx = idx.clamp(0, self.N - 1).squeeze(1)
        t_local = t - cum_durations[range(self.num_env), idx]

        mask = t_local > self.durations[range(self.num_env), idx] + 1e-10
        if mask.any():
            for env in torch.where(mask)[0]:
                logger.error(f"Retrieved timestamp out of trajectory duration in environment {env.item()} #^#")

        return idx, t_local

    def getPos(self, t):
        idx, t_local = self.locatePieceIdx(t)
        pos = torch.zeros(self.num_env, 3, dtype=torch.float32, device=self.coeffMats.device)
        tn = torch.ones(self.num_env, dtype=torch.float32, device=self.coeffMats.device)

        coeff = self.coeffMats[range(self.num_env), idx]
        for i in range(5, -1, -1):
            pos += tn.unsqueeze(1) * coeff[:, :, i]
            tn *= t_local
        return pos

    def getVel(self, t):
        idx, t_local = self.locatePieceIdx(t)
        vel = torch.zeros(self.num_env, 3, dtype=torch.float32, device=self.coeffMats.device)
        tn = torch.ones(self.num_env, dtype=torch.float32, device=self.coeffMats.device)
        n = 1

        coeff = self.coeffMats[range(self.num_env), idx]
        for i in range(4, -1, -1):
            vel += n * tn.unsqueeze(1) * coeff[:, :, i]
            tn *= t_local
            n += 1
        return vel

    def getAcc(self, t):
        idx, t_local = self.locatePieceIdx(t)
        acc = torch.zeros(self.num_env, 3, dtype=torch.float32, device=self.coeffMats.device)
        tn = torch.ones(self.num_env, dtype=torch.float32, device=self.coeffMats.device)
        m = 1
        n = 2

        coeff = self.coeffMats[range(self.num_env), idx]
        for i in range(3, -1, -1):
            acc += m * n * tn.unsqueeze(1) * coeff[:, :, i]
            tn *= t_local
            m += 1
            n += 1
        return acc

    def getJer(self, t):
        idx, t_local = self.locatePieceIdx(t)
        jer = torch.zeros(self.num_env, 3, dtype=torch.float32, device=self.coeffMats.device)
        tn = torch.ones(self.num_env, dtype=torch.float32, device=self.coeffMats.device)
        l = 1
        m = 2
        n = 3

        coeff = self.coeffMats[range(self.num_env), idx]
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
        self.lowerBw = p
        self.upperBw = q
        self.MatData = torch.zeros(num_env, self.N * (self.lowerBw + self.upperBw + 1), dtype=torch.float32, device=device)

    def __call__(self, i, j):
        return self.MatData[:, (i - j + self.upperBw) * self.N + j]
 
    def factorizeLU(self):
        eps = 1e-10
        for k in range(self.N - 1):
            iM = min(k + self.lowerBw, self.N - 1)
            cVl = self(k, k)
            cVl[torch.abs(cVl) < eps] += eps
            for i in range(k + 1, iM + 1):
                mask = self(i, k) != 0.0
                if mask.any():
                    self(i, k)[mask] /= cVl[mask]

            jM = min(k + self.upperBw, self.N - 1)
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
            iM = min(j + self.lowerBw, self.N - 1)
            for i in range(j + 1, iM + 1):
                mask = self(i, j) != 0.0
                if mask.any():
                    b[mask, i] -= self(i, j)[mask].unsqueeze(1) * b[mask, j]

        eps = 1e-10
        for j in range(self.N - 1, -1, -1):
            self(j, j)[torch.abs(self(j, j)) < eps] += eps
            b[:, j] /= self(j, j).unsqueeze(1)
            iM = max(0, j - self.upperBw)
            for i in range(iM, j):
                mask = self(i, j) != 0.0
                if mask.any():
                    b[mask, i] -= self(i, j)[mask].unsqueeze(1) * b[mask, j]


class MinJerkOpt:
    def __init__(self, headState, tailState, pieceNum):
        self.num_env = headState.shape[0]
        self.device = headState.device

        self.N = pieceNum
        self.headPVA = headState
        self.tailPVA = tailState

    def generate(self, inPs, ts):
        if inPs.shape[1] == 0:
            self.T1 = ts
            t1_inv = 1.0 / self.T1
            t2_inv = t1_inv * t1_inv
            t3_inv = t2_inv * t1_inv
            t4_inv = t2_inv * t2_inv
            t5_inv = t4_inv * t1_inv
            coeffMatReversed = torch.zeros(self.num_env, 3, 6, dtype=torch.float32, device=self.device)
            coeffMatReversed[:, :, 5] = (
                0.5 * (self.tailPVA[:, :, 2] - self.headPVA[:, :, 2]) * t3_inv
                - 3.0 * (self.headPVA[:, :, 1] + self.tailPVA[:, :, 1]) * t4_inv
                + 6.0 * (self.tailPVA[:, :, 0] - self.headPVA[:, :, 0]) * t5_inv
            )
            coeffMatReversed[:, :, 4] = (
                (-self.tailPVA[:, :, 2] + 1.5 * self.headPVA[:, :, 2]) * t2_inv
                + (8.0 * self.headPVA[:, :, 1] + 7.0 * self.tailPVA[:, :, 1]) * t3_inv
                + 15.0 * (-self.tailPVA[:, :, 0] + self.headPVA[:, :, 0]) * t4_inv
            )
            coeffMatReversed[:, :, 3] = (
                (0.5 * self.tailPVA[:, :, 2] - 1.5 * self.headPVA[:, :, 2]) * t1_inv
                - (6.0 * self.headPVA[:, :, 1] + 4.0 * self.tailPVA[:, :, 1]) * t2_inv
                + 10.0 * (self.tailPVA[:, :, 0] - self.headPVA[:, :, 0]) * t3_inv
            )
            coeffMatReversed[:, :, 2] = 0.5 * self.headPVA[:, :, 2]
            coeffMatReversed[:, :, 1] = self.headPVA[:, :, 1]
            coeffMatReversed[:, :, 0] = self.headPVA[:, :, 0]
            self.b = coeffMatReversed.transpose(1, 2)
        else:
            self.T1 = ts
            T2 = self.T1 * self.T1
            T3 = T2 * self.T1
            T4 = T2 * T2
            T5 = T4 * self.T1

            A = BandedSystem(self.num_env, 6 * self.N, 6, 6, device=self.device)
            self.b = torch.zeros(self.num_env, 6 * self.N, 3, dtype=torch.float32, device=self.device)

            A(0, 0)[:] = 1.0
            A(1, 1)[:] = 1.0
            A(2, 2)[:] = 2.0
            self.b[:, 0] = self.headPVA[:, :, 0]
            self.b[:, 1] = self.headPVA[:, :, 1]
            self.b[:, 2] = self.headPVA[:, :, 2]

            for i in range(self.N - 1):
                A(6 * i + 3, 6 * i + 3)[:] = 6.0
                A(6 * i + 3, 6 * i + 4)[:] = 24.0 * self.T1[:, i]
                A(6 * i + 3, 6 * i + 5)[:] = 60.0 * T2[:, i]
                A(6 * i + 3, 6 * i + 9)[:] = -6.0
                A(6 * i + 4, 6 * i + 4)[:] = 24.0
                A(6 * i + 4, 6 * i + 5)[:] = 120.0 * self.T1[:, i]
                A(6 * i + 4, 6 * i + 10)[:] = -24.0
                A(6 * i + 5, 6 * i)[:] = 1.0
                A(6 * i + 5, 6 * i + 1)[:] = self.T1[:, i]
                A(6 * i + 5, 6 * i + 2)[:] = T2[:, i]
                A(6 * i + 5, 6 * i + 3)[:] = T3[:, i]
                A(6 * i + 5, 6 * i + 4)[:] = T4[:, i]
                A(6 * i + 5, 6 * i + 5)[:] = T5[:, i]
                A(6 * i + 6, 6 * i)[:] = 1.0
                A(6 * i + 6, 6 * i + 1)[:] = self.T1[:, i]
                A(6 * i + 6, 6 * i + 2)[:] = T2[:, i]
                A(6 * i + 6, 6 * i + 3)[:] = T3[:, i]
                A(6 * i + 6, 6 * i + 4)[:] = T4[:, i]
                A(6 * i + 6, 6 * i + 5)[:] = T5[:, i]
                A(6 * i + 6, 6 * i + 6)[:] = -1.0
                A(6 * i + 7, 6 * i + 1)[:] = 1.0
                A(6 * i + 7, 6 * i + 2)[:] = 2 * self.T1[:, i]
                A(6 * i + 7, 6 * i + 3)[:] = 3 * T2[:, i]
                A(6 * i + 7, 6 * i + 4)[:] = 4 * T3[:, i]
                A(6 * i + 7, 6 * i + 5)[:] = 5 * T4[:, i]
                A(6 * i + 7, 6 * i + 7)[:] = -1.0
                A(6 * i + 8, 6 * i + 2)[:] = 2.0
                A(6 * i + 8, 6 * i + 3)[:] = 6 * self.T1[:, i]
                A(6 * i + 8, 6 * i + 4)[:] = 12 * T2[:, i]
                A(6 * i + 8, 6 * i + 5)[:] = 20 * T3[:, i]
                A(6 * i + 8, 6 * i + 8)[:] = -2.0
                self.b[:, 6 * i + 5] = inPs[:, :, i]

            A(6 * self.N - 3, 6 * self.N - 6)[:] = 1.0
            A(6 * self.N - 3, 6 * self.N - 5)[:] = self.T1[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 4)[:] = T2[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 3)[:] = T3[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 2)[:] = T4[:, self.N - 1]
            A(6 * self.N - 3, 6 * self.N - 1)[:] = T5[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 5)[:] = 1.0
            A(6 * self.N - 2, 6 * self.N - 4)[:] = 2 * self.T1[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 3)[:] = 3 * T2[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 2)[:] = 4 * T3[:, self.N - 1]
            A(6 * self.N - 2, 6 * self.N - 1)[:] = 5 * T4[:, self.N - 1]
            A(6 * self.N - 1, 6 * self.N - 4)[:] = 2.0
            A(6 * self.N - 1, 6 * self.N - 3)[:] = 6 * self.T1[:, self.N - 1]
            A(6 * self.N - 1, 6 * self.N - 2)[:] = 12 * T2[:, self.N - 1]
            A(6 * self.N - 1, 6 * self.N - 1)[:] = 20 * T3[:, self.N - 1]

            self.b[:, 6 * self.N - 3] = self.tailPVA[:, :, 0]
            self.b[:, 6 * self.N - 2] = self.tailPVA[:, :, 1]
            self.b[:, 6 * self.N - 1] = self.tailPVA[:, :, 2]

            A.factorizeLU()
            A.solve(self.b)

    def getTraj(self):
        blocks = self.b.view(self.num_env, self.N, 6, 3)
        coeffMats = blocks.permute(0, 1, 3, 2).flip(3)
        return Trajectory(self.T1, coeffMats)
