from torch import Tensor
import torch


def _shrinkage(tau, matrix):
    return torch.sign(matrix) * torch.max(
        torch.abs(matrix) - tau, torch.zeros(matrix.shape)
    )


def _singular_value_threshold(tau, matrix):
    u, sigma, v = torch.linalg.svd(matrix, full_matrices=False)

    # Thresholding of singular values and construction of diagonal matrices
    sigma = torch.diag(_shrinkage(tau, sigma))

    return torch.matmul(torch.matmul(u, sigma), v)


def robust_pca(M: Tensor, mu: float, lmd: float, max_iter_pass: int = 500):
    L = torch.zeros_like(M)
    S = torch.zeros_like(M)
    Y = torch.zeros_like(M)

    # Cited from: Understanding Robust Principal Component Analysis (RPCA)
    # Author: AneetKumard
    # Link: https://medium.com/@aneetkumard8/understanding-robust-principal-component-analysis-rpca-d722aab80202
    if not mu:
        mu = torch.prod(torch.tensor(M.shape)) / (4 * torch.norm(M, p=1))

    if not lmd:
        lmd = 1 / torch.sqrt(torch.max(torch.tensor(M.shape)))

    # Based on the original text
    # Link: https://arxiv.org/abs/0912.3599(Robust Principal Component Analysis?)
    delta = 1e-7
    lower_limit_value = delta * torch.norm(M, p=2)

    # The initialized 1 is large enough for 'lowerlimitvalue' to enter the judgment
    L2norm_value = torch.tensor(1)

    for _ in range(max_iter_pass):
        if L2norm_value > lower_limit_value:
            L = _singular_value_threshold(1 / mu, M - S + (1 / mu) * Y)
            S = _shrinkage(
                lmd / mu,
                M - L + (1 / mu) * Y,
            )
            Y = Y + mu * (M - L - S)
            L2norm_value = torch.norm(M - L - S, p=2)
            print("||M-L-S||:{}".format(L2norm_value))
        else:
            break
    return L, S
