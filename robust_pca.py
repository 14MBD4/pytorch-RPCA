import torch


class RPCA:
    def __init__(self, matrix, mu=None, lmd=None):
        self.__M = torch.tensor(matrix, dtype=torch.float)
        self.L = torch.zeros_like(self.__M)
        self.S = torch.zeros_like(self.__M)
        self.__Y = torch.zeros_like(self.__M)
        self.__mu = (
            mu
            if mu
            else (
                torch.prod(torch.tensor(self.__M.shape))
                / (torch.tensor(4) * torch.norm(self.__M, p=1))
            )
        )
        # Cited from: Understanding Robust Principal Component Analysis (RPCA)
        # Author: AneetKumard
        # Link: https://medium.com/@aneetkumard8/understanding-robust-principal-component-analysis-rpca-d722aab80202
        self.__lmd = (
            lmd
            if lmd
            else torch.tensor(1) / torch.sqrt(torch.max(torch.tensor(self.__M.shape)))
        )
        self.__delta = torch.tensor(1e-7)
        self.__lower_limit_value = self.__delta * torch.norm(self.__M, p=2)
        # Based on the original text
        # Link: https://arxiv.org/abs/0912.3599(Robust Principal Component Analysis?)

    def __singular_value_threshold(self, tau, matrix):
        u, sigma, v = torch.linalg.svd(matrix, full_matrices=False)
        sigma = torch.diag(
            self.__shrinkage(tau, sigma)
        )  # Thresholding of singular values and construction of diagonal matrices
        return torch.matmul(torch.matmul(u, sigma), v)

    def __shrinkage(self, tau, matrix):
        return torch.sign(matrix) * torch.max(
            torch.abs(matrix) - tau,
            torch.zeros(matrix.shape),
        )

    def alternating_directions(self, max_iter_pass=500):
        L2norm_value = torch.tensor(
            1
        )  # The initialized 1 is large enough for 'lowerlimitvalue' to enter the judgment
        for _ in range(max_iter_pass):
            if L2norm_value > self.__lower_limit_value:
                self.L = self.__singular_value_threshold(
                    1 / self.__mu, self.__M - self.S + (1 / self.__mu) * self.__Y
                )
                self.S = self.__shrinkage(
                    self.__lmd / self.__mu,
                    self.__M - self.L + (1 / self.__mu) * self.__Y,
                )
                self.__Y = self.__Y + self.__mu * (self.__M - self.L - self.S)

                L2norm_value = torch.norm(self.__M - self.L - self.S, p=2)
                print("||M-L-S||:{}".format(L2norm_value))
            else:
                break
        return self.L, self.S
