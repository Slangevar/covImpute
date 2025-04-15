import torch
import warnings
import numpy as np
from torch import Tensor
from typing import Optional, Union, Callable
from torch.special import log_ndtr  # For numerically stable log(Phi(x))
from sklearn.model_selection import KFold

STANDARD_NORMAL = torch.distributions.Normal(0.0, 1.0)


def f1_torch(
    U: Tensor, M1: Tensor, W: Tensor, Q: Tensor, weighted: bool = False
) -> Tensor:
    """
    Compute the cross-entropy loss for U and W.
    """
    W_minus_Q = W - Q
    log_Phi = log_ndtr(W_minus_Q)  # pylint: disable=not-callable
    log_one_minus_Phi = log_ndtr(-W_minus_Q)  # pylint: disable=not-callable

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
    log_cdf = log_ndtr(W_minus_Q)  # pylint: disable=not-callable
    log_one_minus_cdf = log_ndtr(-W_minus_Q)  # pylint: disable=not-callable

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
    return torch.as_tensor(x, device=device, dtype=torch.float64)


def covImpute_torch(
    E: Union[Tensor, np.ndarray],
    mu: float,
    *,
    V: Optional[Union[Tensor, np.ndarray]] = None,
    U: Optional[Union[Tensor, np.ndarray]] = None,
    Q: Optional[Union[Tensor, np.ndarray]] = None,
    maxit: int = 1000,
    tol: float = 1e-3,
    patience: int = 3,
    verbose: bool = False,
    weighted: bool = False,
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
    E = to_tensor(E, device)
    U = to_tensor(U, device)
    Q = to_tensor(Q, device)

    eigvals, eigvecs = torch.linalg.eigh(E)  # pylint: disable=not-callable
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

        return X_hat, X_hat, Z

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
                grad = df1_torch(U, M, Z, Q, weighted) + mu * (Z @ E_inv)
                Z = tilde_Z - step_size * grad
                obj = f1_torch(U, M, Z, Q, weighted) + f3_torch(Z, E_inv_sqrt, mu)
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

            return X_hat, Z, Z

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
            V_hat = torch.where(M2.bool(), V, Y_hat)
            X_hat_partial = torch.cat((Z[:, :m1], V_hat), dim=1)
            X_hat = torch.where(M.bool(), X, torch.cat((W_hat, Y_hat), dim=1))

            return X_hat, X_hat_partial, Z


def cv_mu_torch(
    E: Union[Tensor, np.ndarray],
    mu_values: list[float],
    *,
    V: Optional[Union[Tensor, np.ndarray]] = None,
    U: Optional[Union[Tensor, np.ndarray]] = None,
    Q: Optional[Union[Tensor, np.ndarray]] = None,
    loss_fn: Optional[
        Callable[
            [
                Union[Tensor, np.ndarray],
                Union[Tensor, np.ndarray],
                Union[Tensor, np.ndarray],
            ],
            float,
        ]
    ] = None,
    nfolds: int = 5,
    maxit: int = 1000,
    tol: float = 1e-3,
    patience: int = 3,
    seed: int = 9727,
    verbose: int = 0,
    weighted: bool = False,
    device: Optional[str] = None,
) -> tuple[float, float]:
    """
    PyTorch cross-validation to select mu for covImpute_torch.
    Compatible with both torch and NumPy-based loss functions.
    """
    assert V is not None or U is not None, "At least one of V or U must be provided"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Convert inputs to torch tensors on the specified device
    E = to_tensor(E, device)
    V = to_tensor(V, device)
    U = to_tensor(U, device)
    Q = to_tensor(Q, device)

    if U is not None and V is not None:
        m1 = U.shape[1]
        obs_mask = torch.cat([~torch.isnan(U), ~torch.isnan(V)], dim=1)
        col_has_missing = torch.cat([torch.isnan(U), torch.isnan(V)], dim=1).any(dim=0)
        U = torch.nan_to_num(U)
        V = torch.nan_to_num(V)
    elif U is not None:
        obs_mask = ~torch.isnan(U)
        col_has_missing = torch.isnan(U).any(dim=0)
        U = torch.nan_to_num(U)
    else:
        obs_mask = ~torch.isnan(V)
        col_has_missing = torch.isnan(V).any(dim=0)
        V = torch.nan_to_num(V)

    all_obs_indices = obs_mask.nonzero(as_tuple=False).cpu().numpy()
    obs_indices = all_obs_indices[col_has_missing[all_obs_indices[:, 1]].cpu().numpy()]

    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)
    sum_loss = {mu: 0.0 for mu in mu_values}

    for fold_idx, (_, test_idx) in enumerate(kf.split(obs_indices)):
        if verbose > 0:
            print(f"Processing Fold {fold_idx + 1}/{nfolds}")

        test_mask = obs_indices[test_idx]

        V_train = V.clone() if V is not None else None
        U_train = U.clone() if U is not None else None

        if U is not None and V is not None:
            m1 = U.shape[1]
            test_U_idx = test_mask[test_mask[:, 1] < m1]
            test_V_idx = test_mask[test_mask[:, 1] >= m1]

            U_test_mask = torch.zeros_like(U, dtype=torch.bool)
            V_test_mask = torch.zeros_like(V, dtype=torch.bool)

            U_test_mask[test_U_idx[:, 0], test_U_idx[:, 1]] = True
            V_test_mask[test_V_idx[:, 0], test_V_idx[:, 1] - m1] = True

            U_train[U_test_mask] = torch.nan
            V_train[V_test_mask] = torch.nan

        elif U is not None:
            U_test_mask = torch.zeros_like(U, dtype=torch.bool)
            U_test_mask[test_mask[:, 0], test_mask[:, 1]] = True
            U_train[U_test_mask] = torch.nan

        else:
            V_test_mask = torch.zeros_like(V, dtype=torch.bool)
            V_test_mask[test_mask[:, 0], test_mask[:, 1]] = True
            V_train[V_test_mask] = torch.nan

        for mu in mu_values:
            if verbose:
                print(f"Evaluating mu = {mu:.2e}")

            X_hat, _, _ = covImpute_torch(
                E=E,
                mu=mu,
                V=V_train,
                U=U_train,
                Q=Q,
                maxit=maxit,
                tol=tol,
                patience=patience,
                verbose=(verbose == 2),
                weighted=weighted,
                device=device,
            )

            # Compute loss (try torch, fallback to numpy if needed)
            if loss_fn is not None:
                try:
                    if U is not None and V is not None:
                        X_true = torch.cat((U, V), dim=1)
                        mask = torch.zeros_like(X_true, dtype=torch.bool)
                        mask[test_mask[:, 0], test_mask[:, 1]] = True
                    elif U is not None:
                        X_true = U
                        mask = U_test_mask
                    else:
                        X_true = V
                        mask = V_test_mask

                    fold_loss = loss_fn(X_true, X_hat, mask)

                except Exception:
                    # Fall back to NumPy-based loss_fn
                    if U is not None and V is not None:
                        X_true = torch.cat((U, V), dim=1).cpu().numpy()
                        mask = np.zeros_like(X_true, dtype=bool)
                        mask[test_mask[:, 0], test_mask[:, 1]] = True
                    elif U is not None:
                        X_true = U.cpu().numpy()
                        mask = U_test_mask.cpu().numpy()
                    else:
                        X_true = V.cpu().numpy()
                        mask = V_test_mask.cpu().numpy()

                    X_hat = X_hat.cpu().numpy()

                    fold_loss = loss_fn(X_true, X_hat, mask)

            else:
                if U is not None and V is not None:
                    fold_loss = (
                        f1_torch(
                            U, U_test_mask.float(), X_hat[:, :m1], Q, weighted
                        ).item()
                        + f2_torch(V, V_test_mask.float(), X_hat[:, m1:]).item()
                    )
                elif U is not None:
                    fold_loss = f1_torch(
                        U, U_test_mask.float(), X_hat, Q, weighted
                    ).item()
                else:
                    fold_loss = f2_torch(V, V_test_mask.float(), X_hat).item()

            if verbose > 0:
                print(f"Fold {fold_idx + 1}, mu = {mu:.2e}, Loss: {fold_loss:.4e}")

            sum_loss[mu] += fold_loss

    best_mu = min(sum_loss, key=sum_loss.get)
    best_avg_loss = sum_loss[best_mu] / nfolds

    if verbose > 0:
        print(f"Best mu: {best_mu:.2e}, Average Loss: {best_avg_loss:.4e}")

    return best_mu, best_avg_loss


def higham_torch(
    X: Union[torch.Tensor, np.ndarray],
    eps: float = 1e-6,
    kappa: float = 1e3,
    eig_tol: float = 1e-6,
    maxit: int = 1000,
    verbose: bool = False,
    device: Optional[str] = None,
) -> torch.Tensor:
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
