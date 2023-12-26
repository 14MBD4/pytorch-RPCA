from torch import Tensor
import torch


def _shrinkage(tau: Tensor, M: Tensor) -> Tensor:
    return M.sign() * torch.maximum(M.abs() - tau, torch.zeros_like(M))


def _singular_value_threshold(tau: Tensor, M: Tensor) -> Tensor:
    u, sigma, v = torch.linalg.svd(M, full_matrices=False)
    return u @ _shrinkage(tau, sigma.diag()) @ v


def robust_pca(M: Tensor, mu: float = None, lmd: float = None, delta: float = 1e-7, max_iter_pass: int = 500) -> tuple[Tensor, Tensor]:  # type: ignore
    """Robust PCA

    Args:
        M: The data to be processed.
        mu: Parameter controlling the balance between low-rank and sparse components.
        lmd: Regularization parameter for the sparse component.
        delta: Convergence threshold. Defaults to 1e-7.
        max_iter_pass: Maximum number of iterations. Defaults to 500.

    Raises:
        ValueError: If mu is not positive.

    Note:
        The default parameter of this function is set according to the original paper.
        See the References for more details.

    References:
        - "Robust Principal Component Analysis?" by Emmanuel J. Candès, et al.

    Returns:
        Tuple containing the low-rank component (L) and the sparse component (S).
    """
    mu: Tensor
    if not mu:
        mu = torch.prod(torch.tensor(M.shape)) / (4 * torch.linalg.norm(M, ord=1))
    elif mu <= 0:
        raise ValueError("mu must be a positive number")
    else:
        mu = torch.tensor(mu)

    lmd: Tensor
    if not lmd:
        lmd = 1 / torch.sqrt(torch.max(torch.tensor(M.shape)))
    else:
        lmd = torch.tensor(lmd)

    L = torch.zeros_like(M)
    S = torch.zeros_like(M)
    Y = torch.zeros_like(M)

    stop_boundary = delta * torch.linalg.norm(M, ord="fro")
    frobenius_norm_value = 0xDEADBEEF  # This vaule is big enough to enter the iteration step at the beginning

    for _ in range(max_iter_pass):
        if frobenius_norm_value > stop_boundary:
            L = _singular_value_threshold(1 / mu, M - S + (1 / mu) * Y)
            S = _shrinkage(lmd / mu, M - L + (1 / mu) * Y)
            Y = Y + mu * (M - L - S)
            frobenius_norm_value = torch.linalg.norm(M - L - S, ord="fro")
            print(f"Current frobenius norm value: {frobenius_norm_value}")
        else:
            break

    return L, S


if __name__ == "__main__":
    L, S = robust_pca(torch.randn(4, 6))
    print(L)
