import torch
import math
import warnings
import numpy as np
from torch import Tensor
from typing import Optional, Union
from torch.special import log_ndtr  # For numerically stable log(Phi(x))

STANDARD_NORMAL = torch.distributions.Normal(0.0, 1.0)


def f1_torch(
    U: Tensor,
    M1: Tensor,
    W: Tensor,
    Q: Tensor,
    mu: float = 1.0,
) -> Tensor:
    """
    Compute the cross-entropy loss for U and W.
    """
    inv_sqrt_mu = 1 / math.sqrt(mu)
    W_minus_Q = (W - Q) * inv_sqrt_mu
    log_Phi = log_ndtr(W_minus_Q)  # pylint: disable=not-callable
    log_one_minus_Phi = log_ndtr(-W_minus_Q)  # pylint: disable=not-callable

    return -torch.sum(M1 * (U * log_Phi + (1 - U) * log_one_minus_Phi))


def df1_torch(U: Tensor, M1: Tensor, W: Tensor, Q: Tensor, mu: float = 1.0) -> Tensor:
    """
    Compute the gradient of f1 with respect to W.
    """
    inv_sqrt_mu = 1 / math.sqrt(mu)
    W_minus_Q = (W - Q) * inv_sqrt_mu
    log_pdf = STANDARD_NORMAL.log_prob(W_minus_Q)
    log_cdf = log_ndtr(W_minus_Q)  # pylint: disable=not-callable
    log_one_minus_cdf = log_ndtr(-W_minus_Q)  # pylint: disable=not-callable

    first_term = torch.exp(log_pdf - log_cdf)
    second_term = torch.exp(log_pdf - log_one_minus_cdf)

    return -M1 * (U * first_term - (1 - U) * second_term) * inv_sqrt_mu


def f2_torch(
    V: Tensor,
    M2: Tensor,
    Y: Tensor,
    mu: float = 1.0,
) -> Tensor:
    """
    Compute the squared error loss.
    """
    diff = M2 * (V - Y)
    return 0.5 * torch.sum(diff * diff) / mu


def f3_torch(Z: Tensor, E_inv_sqrt: Tensor) -> Tensor:
    """
    Compute the covariance regularization term.
    """
    Z_E_inv_sqrt = Z @ E_inv_sqrt

    return 0.5 * torch.sum(Z_E_inv_sqrt * Z_E_inv_sqrt)


def to_tensor(x, device):
    if x is None:
        return None
    return torch.as_tensor(x, device=device, dtype=torch.float64)


def covImpute_torch(
    G: Union[Tensor, np.ndarray],
    mu: float,
    *,
    V: Optional[Union[Tensor, np.ndarray]] = None,
    U: Optional[Union[Tensor, np.ndarray]] = None,
    Q: Optional[Union[Tensor, np.ndarray]] = None,
    maxit: int = 10000,
    tol: float = 1e-4,
    patience: int = 3,
    verbose: bool = False,
    device: Optional[str] = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Covariance-regularized imputation with optional binary part.
    Accepts NumPy arrays or PyTorch tensors.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert all inputs to torch.Tensor on device
    V = to_tensor(V, device)
    G = to_tensor(G, device)
    U = to_tensor(U, device)
    Q = to_tensor(Q, device)

    eigvals, eigvecs = torch.linalg.eigh(G)  # pylint: disable=not-callable
    gamma_min, gamma_max = eigvals.min(), eigvals.max()

    eta = 1 / gamma_max
    L = 1 / mu + 1 / gamma_min
    step_size = 1 / L
    if verbose:
        print(f"Step size 1/L = {step_size:.4e}")
    sqrt_q_f = torch.sqrt(eta / L)
    beta = (1 - sqrt_q_f) / (1 + sqrt_q_f)

    E_inv = eigvecs @ torch.diag(1 / eigvals) @ eigvecs.T
    E_inv_sqrt = eigvecs @ torch.diag(1 / eigvals.sqrt()) @ eigvecs.T

    obj_old = torch.tensor(float("inf"), device=device)
    patience_counter = 0

    if U is None:
        if verbose:
            warnings.warn(
                "No binary part provided. Performing continuous-only imputation."
            )
        # Continuous-only imputation
        M = (~torch.isnan(V)).float()
        V = torch.nan_to_num(V)
        Z = V.clone()
        tilde_Z = Z.clone()

        for it in range(1, maxit + 1):
            Z_old = Z.clone()
            grad = -M * (V - tilde_Z) / mu + (tilde_Z @ E_inv)
            Z = tilde_Z - step_size * grad

            obj = f2_torch(V, M, Z, mu=mu) + f3_torch(Z, E_inv_sqrt)
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

        if V is None:
            if verbose:
                warnings.warn(
                    "No continuous part provided. Performing binary-only imputation."
                )
            # Binary-only imputation
            M = (~torch.isnan(U)).float()
            U = torch.nan_to_num(U)
            Z = U.clone()
            tilde_Z = Z.clone()

            for it in range(1, maxit + 1):
                Z_old = Z.clone()
                grad = df1_torch(U, M, Z, Q, mu=mu) + (Z @ E_inv)
                Z = tilde_Z - step_size * grad
                obj = f1_torch(U, M, Z, Q, mu=mu) + f3_torch(Z, E_inv_sqrt)
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

            X_hat = torch.where(M.bool(), U, (Z > Q).float())

            return X_hat, Z

        else:
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

                grad_W = df1_torch(U, M1, W, Q, mu=mu)
                grad_V = -M2 * (V - Y) / mu
                grad_obj = torch.cat((grad_W, grad_V), dim=1)

                grad_penalty = tilde_Z @ E_inv
                Z = tilde_Z - step_size * (grad_obj + grad_penalty)

                obj = (
                    f1_torch(U, M1, Z[:, :m1], Q, mu=mu)
                    + f2_torch(V, M2, Z[:, m1:], mu=mu)
                    + f3_torch(Z, E_inv_sqrt)
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


def nearest_mu(
    E: Union[Tensor, np.ndarray],
    norm: str = "fro",
    device: Optional[str] = None,
) -> Tensor:
    """
    Find the mu such that mu I is the nearest for the (environemental) covariance matrix E.

    E: torch.Tensor or Numpy array (The environmental covariance matrix)
    norm: str (norm type, either 'fro', 'spectral', or 'nuclear')
    device: device to use for computation (e.g., 'cuda' or 'cpu')

    Returns:
    torch.Tensor: The mu value that leads to the nearest mu I matrix to E under the specified norm.
    """
    E = to_tensor(E, device=device)
    if norm == "fro":
        return torch.trace(E) / E.shape[0]
    if norm == "spectral" or norm == "nuclear":
        eigvals, _ = torch.linalg.eigh(E)  # pylint: disable=not-callable
        if norm == "spectral":
            gamma_min, gamma_max = eigvals.min(), eigvals.max()
            return (gamma_min + gamma_max) / 2.0
        else:
            return torch.median(eigvals)
    raise ValueError("Unsupported norm type. Use 'fro', 'spectral', or 'nuclear'.")


def higham_torch(
    X: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-6,
    kappa: float = 1e3,
    eig_tol: float = 1e-6,
    maxit: int = 1000,
    verbose: bool = False,
    device: Optional[str] = None,
) -> Tensor:
    """
    Higham's algorithm to find the nearest positive semidefinite matrix (torch version).

    Parameters:
    X: torch.Tensor or Numpy array (2D square matrix, symmetric but possibly not PSD)
    eps: convergence tolerance
    kappa: target condition number (lambda_max / lambda_min)
    eig_tol: relative eigenvalue threshold
    maxit: maximum number of iterations
    verbose: whether to print convergence info
    device: device to use for computation (e.g., 'cuda' or 'cpu')

    Returns:
    torch.Tensor: PSD-adjusted version of X
    """
    X = to_tensor(X, device)
    if not torch.allclose(X, X.T, atol=1e-10):
        raise ValueError("Input matrix must be symmetric.")

    D_s = torch.zeros_like(X)
    conv = float("inf")

    for it in range(1, maxit + 1):
        Y = X
        R = Y - D_s
        d, Q = torch.linalg.eigh(R)  # pylint: disable=not-callable
        max_d = d[-1]
        p = d > (eig_tol * max_d)

        if not torch.any(p):
            raise ValueError("Matrix seems to be negative semidefinite.")

        Q = Q[:, p]
        X = Q @ torch.diag(d[p]) @ Q.T
        D_s = X - R
        conv = torch.norm(Y - X, p="fro") / torch.norm(Y, p="fro")

        if verbose:
            print(f"Iteration {it}: Convergence criterion = {conv:.6f}")

        if conv < eps:
            break

    # Enforce condition number constraint
    d, Q = torch.linalg.eigh(X)  # pylint: disable=not-callable
    min_val = d[-1] / kappa
    d[d < min_val] = min_val

    o_diag = torch.diag(X)
    X = Q @ torch.diag(d) @ Q.T
    D = torch.sqrt(torch.maximum(min_val, o_diag) / torch.diag(X))
    X = D[:, None] * X * D[None, :]

    if conv >= eps:
        print(f"Warning: Higham's algorithm did not converge after {maxit} iterations.")

    return X
