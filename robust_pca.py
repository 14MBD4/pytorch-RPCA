from torch import Tensor
import torch


def _shrinkage(tau: float, M: Tensor) -> Tensor:
    return M.sign() * torch.maximum(M.abs() - tau, torch.zeros_like(M))


def _singular_value_threshold(tau: float, M: Tensor) -> Tensor:
    u, sigma, v = torch.linalg.svd(M, full_matrices=False)
    return u @ _shrinkage(tau, sigma).diag() @ v


def robust_pca(M: Tensor, mu: float = None, lmd: float = None, delta: float = 1e-7, max_iter_pass: int = 500, devices="cpu") -> tuple[Tensor, Tensor]:  # type: ignore
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
    if not mu:
        mu = (
            torch.prod(torch.tensor(M.shape))
            / (torch.tensor(4) * torch.linalg.norm(M, ord=1))
        ).item()

    if mu <= 0:
        raise ValueError("mu must be a positive number")

    if not lmd:
        lmd = (
            torch.tensor(1) / torch.sqrt(torch.max(torch.tensor(M.shape))).item()
        ).item()

    L = torch.zeros_like(M, device=devices)
    S = torch.zeros_like(M, device=devices)
    Y = torch.zeros_like(M, device=devices)

    stop_boundary = delta * torch.linalg.norm(M, ord="fro")
    frobenius_norm_value = torch.inf

    current_pass = 0
    while current_pass <= max_iter_pass and frobenius_norm_value > stop_boundary:
        current_pass += 1
        L = _singular_value_threshold(1 / mu, M - S + (1 / mu) * Y)
        S = _shrinkage(lmd / mu, M - L + (1 / mu) * Y)
        tmp = M - L - S
        Y = Y + mu * tmp
        frobenius_norm_value = torch.linalg.norm(tmp, ord="fro")
        print(f"Current frobenius norm value at {current_pass}: {frobenius_norm_value}")

    return L, S
