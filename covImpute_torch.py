import torch
import numpy as np
from torch import Tensor
from typing import Optional, Union
from torch.special import log_ndtr  # For numerically stable log(Phi(x))

STANDARD_NORMAL = torch.distributions.Normal(0.0, 1.0)


def f1_torch(
    U: Tensor, M1: Tensor, W: Tensor, Q: Tensor, weighted: bool = False
) -> Tensor:
    """
    Compute the cross-entropy loss for U and W.
    """
    W_minus_Q = W - Q
    log_Phi = log_ndtr(W_minus_Q)
    log_one_minus_Phi = log_ndtr(-W_minus_Q)

    if weighted:
        P = STANDARD_NORMAL.cdf(Q)
        return -torch.sum(
            M1 * (U * P * log_Phi + (1 - U) * (1 - P) * log_one_minus_Phi)
        )
    return -torch.sum(M1 * (U * log_Phi + (1 - U) * log_one_minus_Phi))


def df1_torch(
    U: Tensor, M1: Tensor, W: Tensor, Q: Tensor, weighted: bool = False
) -> Tensor:
    """
    Compute the gradient of f1 with respect to W.
    """
    W_minus_Q = W - Q
    log_pdf = STANDARD_NORMAL.log_prob(W_minus_Q)
    log_cdf = log_ndtr(W_minus_Q)
    log_one_minus_cdf = log_ndtr(-W_minus_Q)

    first_term = torch.exp(log_pdf - log_cdf)
    second_term = torch.exp(log_pdf - log_one_minus_cdf)

    if weighted:
        P = STANDARD_NORMAL.cdf(Q)
        return -M1 * (U * P * first_term - (1 - U) * (1 - P) * second_term)

    return -M1 * (U * first_term - (1 - U) * second_term)


def f2_torch(V: Tensor, M2: Tensor, Y: Tensor) -> Tensor:
    """
    Compute the squared error loss.
    """
    diff = M2 * (V - Y)
    return 0.5 * torch.sum(diff * diff)


def f3_torch(Z: Tensor, E_inv_sqrt: Tensor, mu: float) -> Tensor:
    """
    Compute the covariance regularization term.
    """
    return 0.5 * mu * torch.norm(Z @ E_inv_sqrt, p="fro") ** 2


def to_tensor(x, device):
    if x is None:
        return None
    return torch.as_tensor(x, device=device, dtype=torch.float32)


def covImpute_torch(
    V: Union[Tensor, np.ndarray],
    E: Union[Tensor, np.ndarray],
    mu: float,
    *,
    U: Optional[Union[Tensor, np.ndarray]] = None,
    Q: Optional[Union[Tensor, np.ndarray]] = None,
    maxit: int = 1000,
    tol: float = 1e-3,
    patience: int = 3,
    verbose: bool = False,
    weighted: bool = False,
    device: Optional[str] = None,
) -> tuple[Tensor, Tensor]:
    """
    Covariance-regularized imputation with optional binary part.
    Accepts NumPy arrays or PyTorch tensors.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert all inputs to torch.Tensor on device
    V = to_tensor(V, device)
    E = to_tensor(E, device)
    U = to_tensor(U, device)
    Q = to_tensor(Q, device)

    eigvals, eigvecs = torch.linalg.eigh(E)
    gamma_min, gamma_max = eigvals.min(), eigvals.max()

    eta = mu / gamma_max
    L = 1 + mu / gamma_min
    step_size = 1 / L
    beta = (1 - torch.sqrt(eta / L)) / (1 + torch.sqrt(eta / L))

    E_inv = eigvecs @ torch.diag(1 / eigvals) @ eigvecs.T
    E_inv_sqrt = eigvecs @ torch.diag(1 / eigvals.sqrt()) @ eigvecs.T

    obj_old = torch.tensor(float("inf"), device=device)
    patience_counter = 0

    if U is None:
        M = (~torch.isnan(V)).float()
        V = torch.nan_to_num(V)
        Z = V.clone()
        tilde_Z = Z.clone()

        for it in range(1, maxit + 1):
            Z_old = Z.clone()
            grad = -M * (V - tilde_Z) + mu * (tilde_Z @ E_inv)
            Z = tilde_Z - step_size * grad

            obj = f2_torch(V, M, Z) + f3_torch(Z, E_inv_sqrt, mu)
            change = torch.abs(obj_old - obj)

            if verbose:
                print(
                    f"Iter {it}, Objective: {obj.item():.4e}, Change: {change.item():.4e}"
                )

            if change < tol:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= patience:
                break

            tilde_Z = Z + beta * (Z - Z_old)
            obj_old = obj

        X_hat = torch.where(M.bool(), V, Z)
        return X_hat, Z

    else:
        if Q is None:
            raise ValueError("Q must be provided when U is not None.")

        M1 = (~torch.isnan(U)).float()
        M2 = (~torch.isnan(V)).float()
        U = torch.nan_to_num(U)
        V = torch.nan_to_num(V)

        m1 = U.shape[1]
        M = torch.cat((M1, M2), dim=1)
        X = torch.cat((U, V), dim=1)
        Z = torch.cat((M1 * Q, V), dim=1)
        tilde_Z = Z.clone()

        for it in range(1, maxit + 1):
            Z_old = Z.clone()
            W = tilde_Z[:, :m1]
            Y = tilde_Z[:, m1:]

            grad_W = df1_torch(U, M1, W, Q, weighted)
            grad_V = -M2 * (V - Y)
            grad_obj = torch.cat((grad_W, grad_V), dim=1)

            grad_penalty = mu * (tilde_Z @ E_inv)
            Z = tilde_Z - step_size * (grad_obj + grad_penalty)

            obj = (
                f1_torch(U, M1, Z[:, :m1], Q, weighted)
                + f2_torch(V, M2, Z[:, m1:])
                + f3_torch(Z, E_inv_sqrt, mu)
            )
            change = torch.abs(obj_old - obj)

            if verbose:
                print(
                    f"Iter {it}, Objective: {obj.item():.4e}, Change: {change.item():.4e}"
                )

            if change < tol:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= patience:
                break

            tilde_Z = Z + beta * (Z - Z_old)
            obj_old = obj

        W_hat = (Z[:, :m1] > Q).float()
        Y_hat = Z[:, m1:]
        X_hat = torch.where(M.bool(), X, torch.cat((W_hat, Y_hat), dim=1))

        return X_hat, Z
