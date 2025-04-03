import warnings
import numpy as np
from typing import Optional
from scipy.stats import norm


def f1(
    U: np.ndarray,
    M1: np.ndarray,
    W: np.ndarray,
    Q: np.ndarray,
    weighted: bool = False,
) -> float:
    """
    Compute the cross-entropy loss for U and W.

    U: the binary matrix of shape n x m_1
    M1: the mask matrix of shape n x m_1. M_{ij} = 1 if U_{ij} is observed, 0 otherwise.
    Q: the threshold matrix of shape n x m_1
    W: the continuous matrix of shape n x m_1
    weighted: if True, the loss is weighted by the probability of observing certain values of U_{ij}

    Return: the cross-entropy loss
    """
    W_minus_Q = W - Q
    log_Phi = norm.logcdf(W_minus_Q)
    log_one_minus_Phi = norm.logcdf(-W_minus_Q)

    if weighted:
        P = norm.cdf(Q)
        return -np.sum(M1 * (U * P * log_Phi + (1 - U) * (1 - P) * log_one_minus_Phi))
    return -np.sum(M1 * (U * log_Phi + (1 - U) * log_one_minus_Phi))


def df1(
    U: np.ndarray,
    M1: np.ndarray,
    W: np.ndarray,
    Q: np.ndarray,
    weighted: bool = False,
) -> np.ndarray:
    """
    Compute the gradient of f1 with respect to W. Using exponentials of log values to avoid numerical underflow.

    U: the binary matrix of shape n x m_1
    M1: the mask matrix of shape n x m_1. M_{ij} = 1 if U_{ij} is observed, 0 otherwise.
    W: the matrix of shape n x m_1
    Q: the matrix of shape n x m_1
    weighted: if True, the gradient is weighted by the probability of observing certain values of U_{ij}

    Return: the gradient of f1 with respect to W
    """
    W_minus_Q = W - Q
    log_dPhi = norm.logpdf(W_minus_Q)
    log_Phi = norm.logcdf(W_minus_Q)
    log_one_minus_Phi = norm.logcdf(-W_minus_Q)

    first_term = np.exp(log_dPhi - log_Phi)
    second_term = np.exp(log_dPhi - log_one_minus_Phi)

    if weighted:
        P = norm.cdf(Q)
        return -M1 * (U * P * first_term - (1 - U) * (1 - P) * second_term)

    return -M1 * (U * first_term - (1 - U) * second_term)


def f2(V: np.ndarray, M2: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute the squared error loss for V and Y, or equivalently,
    the squared Frobenius norm of the difference.

    V: the continuous matrix of shape n x m_2
    M2: the mask matrix of shape n x m_2. M_{ij} = 1 if V_{ij} is observed, 0 otherwise.
    Y: the imputed matrix of shape n x m_2

    Return: the squared error loss
    """
    diff = M2 * (V - Y)
    return 0.5 * np.sum(diff * diff)


def f3(Z: np.ndarray, E_inv_sqrt: np.ndarray, mu: float) -> float:
    """
    Compute the covariance regularization term.

    Z: the imputed matrix of shape n x m (m = m_1 + m_2)
    E_inv_sqrt: the inverse square root of the covariance matrix E, shape m x m
    mu: the regularization parameter

    Return: the covariance regularization term
    """
    return 0.5 * mu * np.linalg.norm(Z @ E_inv_sqrt, "fro") ** 2


def covImpute(
    V: np.ndarray,
    E: np.ndarray,
    mu: float,
    *,
    U: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    maxit: int = 1000,
    tol: float = 1e-3,
    patience: int = 3,
    verbose: bool = False,
    weighted: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform covariance-regularized matrix imputation with optional binary part.

    V: continuous matrix of shape n x m2 (may contain NaNs)
    E: covariance matrix of shape (m1 + m2) x (m1 + m2)
    mu: regularization parameter
    U: optional binary matrix of shape n x m1 (may contain NaNs)
    Q: threshold matrix of shape n x m1, required if U is provided
    maxit: maximum number of iterations
    tol: convergence tolerance (based on objective improvement)
    patience: number of small-change iterations before early stopping
    verbose: whether to print progress
    weighted: whether to apply observation-based weighting in binary loss

    Returns: (X_hat, Z) where X_hat is the imputed matrix and Z is the final latent variable estimate
    """
    eigvals, eigvecs = np.linalg.eigh(E)  # Symmetric E, use eigh
    gamma_min, gamma_max = np.min(eigvals), np.max(eigvals)

    eta = mu / gamma_max
    L = 1 + mu / gamma_min
    step_size = 1 / L
    beta = (1 - np.sqrt(eta / L)) / (1 + np.sqrt(eta / L))

    E_inv = eigvecs @ np.diag(1 / eigvals) @ eigvecs.T
    E_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    if verbose:
        print(f"Step size = {step_size:.4f}, Beta = {beta:.4f}")

    obj_old = np.inf
    patience_counter = 0

    if U is None:
        if verbose:
            warnings.warn(
                "No binary part provided. Performing continuous-only imputation."
            )

        M = 1 - np.isnan(V)
        V = np.nan_to_num(V)
        Z = V.copy()
        tilde_Z = Z.copy()

        for it in range(1, maxit + 1):
            Z_old = Z.copy()

            # Gradient step
            grad = -M * (V - tilde_Z) + mu * (tilde_Z @ E_inv)
            Z = tilde_Z - step_size * grad

            # Objective
            obj = f2(V, M, Z) + f3(Z, E_inv_sqrt, mu)
            change = np.abs(obj_old - obj)

            if verbose:
                print(f"Iter: {it}, Objective: {obj:.4e}, Change: {change:.4e}")

            if change < tol:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= patience:
                if verbose:
                    print(f"Converged after {it} iterations.")
                break

            tilde_Z = Z + beta * (Z - Z_old)
            obj_old = obj

        X_hat = V + (1 - M) * Z
        return X_hat, Z

    else:
        if Q is None:
            raise ValueError("Q must be provided when U is not None.")

        M1 = 1 - np.isnan(U)
        M2 = 1 - np.isnan(V)
        U = np.nan_to_num(U)
        V = np.nan_to_num(V)

        m1 = U.shape[1]
        M = np.hstack((M1, M2))
        X = np.hstack((U, V))

        # Initialize Z from available values
        Z = np.hstack((M1 * Q, V))
        tilde_Z = Z.copy()

        for it in range(1, maxit + 1):
            Z_old = Z.copy()

            W = tilde_Z[:, :m1]
            Y = tilde_Z[:, m1:]

            grad_W = df1(U, M1, W, Q, weighted)
            grad_V = -M2 * (V - Y)
            grad_obj = np.hstack((grad_W, grad_V))

            grad_penalty = mu * (tilde_Z @ E_inv)
            Z = tilde_Z - step_size * (grad_obj + grad_penalty)

            obj = (
                f1(U, M1, Z[:, :m1], Q, weighted)
                + f2(V, M2, Z[:, m1:])
                + f3(Z, E_inv_sqrt, mu)
            )
            change = np.abs(obj_old - obj)

            if verbose:
                print(f"Iter: {it}, Objective: {obj:.4e}, Change: {change:.4e}")

            if change < tol:
                patience_counter += 1
            else:
                patience_counter = 0

            if patience_counter >= patience:
                if verbose:
                    print(f"Converged after {it} iterations.")
                break

            tilde_Z = Z + beta * (Z - Z_old)
            obj_old = obj

        W_hat = (Z[:, :m1] > Q).astype(int)
        Y_hat = Z[:, m1:]
        X_hat = M * X + (1 - M) * np.hstack((W_hat, Y_hat))

        return X_hat, Z
