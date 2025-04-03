import warnings
import numpy as np
from typing import Optional
from scipy.stats import norm

# from scipy.special import log_ndtr
from sklearn.model_selection import KFold


def f1(
    U: np.ndarray,
    M1: np.ndarray,
    W: np.ndarray,
    Q: np.ndarray,
    weighted: bool = False,
) -> float:
    """
    Compute the cross-entropy loss for U and W

    U: the binary matrix of shape n x m_1
    M1: the mask matrix of shape n x m_1. M_{ij} = 1 if U_{ij} is observed, 0 otherwise.
    Phi: the matrix of shape n x m_1. Phi_{ij} = \Phi(W_{ij} - Q_{ij})
    weight: if True, the loss is weighted by the probability of observing certain values of U_{ij}.

    Return: the cross-entropy loss
    """
    log_Phi = norm.logcdf(W - Q)
    log_one_minus_Phi = norm.logcdf(-W + Q)

    if weighted:
        P = norm.cdf(Q)
        return -np.sum(M1 * (U * P * log_Phi + (1 - U) * (1 - P) * log_one_minus_Phi))
    return -np.sum(M1 * (U * log_Phi + (1 - U) * log_one_minus_Phi))


def f2(V: np.ndarray, M2: np.ndarray, Y: np.ndarray):
    """
    Compute the squared error loss for V and Y, or equivalently,
    the square Frobenius norm of the difference.

    V: the continuous matrix of shape n x m_2
    M2: the mask matrix of shape n x m_2. M_{ij} = 1 if V_{ij} is observed, 0 otherwise.
    Y: the imputed matrix of shape n x m_2

    Return: the squared error loss
    """

    return 0.5 * np.sum((M2 * (V - Y)) ** 2)


def f3(Z: np.ndarray, E_inv_sqrt: np.ndarray, mu: float) -> float:
    """
    Compute the covariance regularization term

    Z: the imputed matrix of shape n x m (m = m_1 + m_2)
    E_inv_sqrt: the inverse of the squared root of the covariance matrix E of shape m x m
    mu: the regularization parameter

    Return: the covariance regularization term
    """
    return 0.5 * mu * np.linalg.norm(Z @ E_inv_sqrt, "fro") ** 2


def df1(
    U: np.ndarray,
    M1: np.ndarray,
    W: np.ndarray,
    Q: np.ndarray,
    weighted: bool = False,
) -> np.ndarray:
    """
    Compute the gradient of f1 with respect to W. Using exponential of the log value to avoid numerical underflow.

    U: the binary matrix of shape n x m_1
    M1: the mask matrix of shape n x m_1. M_{ij} = 1 if U_{ij} is observed, 0 otherwise.
    W: the matrix of shape n x m_1
    Q: the matrix of shape n x m_1
    weighted: if True, the gradient is weighted by the probability of observing certain values of U_{ij}.

    Return: the gradient of f1 with respect to W
    """
    log_dPhi = norm.logpdf(W - Q)
    log_Phi = norm.logcdf(W - Q)
    log_one_minus_Phi = norm.logcdf(-W + Q)

    first_term = np.exp(log_dPhi - log_Phi)
    second_term = np.exp(log_dPhi - log_one_minus_Phi)

    if weighted:
        P = norm.cdf(Q)
        return -M1 * (U * P * first_term - (1 - U) * (1 - P) * second_term)

    return -M1 * (U * first_term - (1 - U) * second_term)


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
    TODO: Complete the docstring
    """
    # Compute eigen decomposition of E
    eigvals, eigvecs = np.linalg.eigh(E)  # Use eigh for symmetric matrices
    gamma_min, gamma_max = np.min(eigvals), np.max(eigvals)

    eta = mu / gamma_max
    L = 1 + mu / gamma_min
    step_size = 1 / L
    beta = (1 - np.sqrt(eta / L)) / (1 + np.sqrt(eta / L))

    E_inv = eigvecs @ np.diag(1 / eigvals) @ eigvecs.T
    E_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T

    if verbose:
        print(f"Step size = {step_size:.4f}, Beta = {beta:.4f}")

    # Convergence monitoring
    patience_counter = 0
    obj_old = np.inf

    if U is None:
        if verbose:
            warnings.warn("There is no binary part. Only continuous part is provided.")
        M = 1 - np.isnan(V)
        V = np.nan_to_num(V)  # Replace NaNs with 0
        Z = V.copy()
        tilde_Z = Z.copy()
        for it in range(1, maxit + 1):
            Z_old = Z.copy()

            Z = tilde_Z - step_size * (-M * (V - tilde_Z) + mu * (tilde_Z @ E_inv))

            obj = f2(V, M, Z) + f3(Z, E_inv_sqrt, mu)
            change_in_obj = np.abs(obj_old - obj)

            if verbose:
                print(f"Iter: {it}, Objective: {obj:.4e}, Change: {change_in_obj:.4e}")

            if change_in_obj < tol:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"Converged after {it} iterations.")
                return V + (1 - M) * Z  # This is the final imputed matrix X_hat
            else:
                # Update tilde Z (tilde W, tilde Y)
                tilde_Z = Z + beta * (Z - Z_old)
                obj_old = obj

        return V + (1 - M) * Z

    else:
        if Q is None:
            raise ValueError("When U is not None, Q must be provided.")
        M1 = 1 - np.isnan(U)
        M2 = 1 - np.isnan(V)
        U = np.nan_to_num(U)  # Replace NaNs with 0
        V = np.nan_to_num(V)
        X = np.hstack((U, V))
        M = np.hstack((M1, M2))

        Z = np.hstack((M1 * Q, V))
        tilde_Z = Z.copy()

        # Iterative optimization
        for it in range(1, maxit + 1):
            W = tilde_Z[:, : U.shape[1]]
            Y = tilde_Z[:, U.shape[1] :]

            # Using stabilized version of the gradient
            grad_W = df1(U, M1, W, Q, weighted)
            grad_V = -M2 * (V - Y)

            grad_obj = np.hstack((grad_W, grad_V))
            grad_penalty = mu * tilde_Z @ E_inv

            Z_old = Z.copy()
            Z = tilde_Z - step_size * (grad_obj + grad_penalty)

            # Compute objective function
            obj = (
                f1(U, M1, Z[:, : U.shape[1]], Q, weighted)
                + f2(V, M2, Z[:, U.shape[1] :])
                + f3(Z, E_inv_sqrt, mu)
            )

            change_in_obj = np.abs(obj_old - obj)

            if verbose:
                print(f"Iter: {it}, Objective: {obj:.4e}, Change: {change_in_obj:.4e}")

            # Check for convergence
            if change_in_obj < tol:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"Converged after {it} iterations.")
                return M * X + (1 - M) * np.hstack(
                    (W > Q, Y)
                )  # This is the final imputed matrix X_hat
            else:
                # Update tilde Z (tilde W, tilde Y)
                tilde_Z = Z + beta * (Z - Z_old)
                obj_old = obj

        return M * X + (1 - M) * np.hstack((W > Q, Y))
