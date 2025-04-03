import warnings
import numpy as np
from typing import Optional, Callable
from scipy.stats import norm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, roc_auc_score


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
    E: np.ndarray,
    mu: float,
    *,
    V: Optional[np.ndarray] = None,
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

        if V is None:
            if verbose:
                warnings.warn(
                    "No continuous part provided. Performing binary-only imputation."
                )

            M = 1 - np.isnan(U)
            U = np.nan_to_num(U)
            Z = U.copy()
            tilde_Z = Z.copy()

            for it in range(1, maxit + 1):
                Z_old = Z.copy()

                # Gradient step
                grad = df1(U, M, tilde_Z, Q, weighted) + mu * (tilde_Z @ E_inv)
                Z = tilde_Z - step_size * grad

                # Objective
                obj = f1(U, M, Z, Q, weighted) + f3(Z, E_inv_sqrt, mu)
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

            X_hat = U + (1 - M) * ((Z > Q).astype(int))

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


def cv_mu(
    E: np.ndarray,
    mu_values: list[float],
    *,
    V: Optional[np.ndarray] = None,
    U: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    loss_fn: Optional[Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = None,
    nfolds: int = 5,
    maxit: int = 1000,
    tol: float = 1e-3,
    patience: int = 3,
    seed: int = 9727,
    verbose: int = 0,
    weighted: bool = False,
) -> tuple[float, dict]:
    """
    Perform cross-validation to find the best mu for covariance-regularized matrix imputation.
    E: covariance matrix of shape (m1 + m2) x (m1 + m2)
    mu_values: list of mu values to evaluate
    V: continuous matrix of shape n x m2 (may contain NaNs)
    U: optional binary matrix of shape n x m1 (may contain NaNs)
    Q: threshold matrix of shape n x m1, required if U is provided
    loss_fn: optional custom loss function
    nfolds: number of folds for cross-validation
    maxit: maximum number of iterations for imputation
    tol: convergence tolerance (based on objective improvement)
    patience: number of small-change iterations before early stopping
    seed: random seed for reproducibility
    verbose: verbosity level (0, 1, or 2). If 1, print progress messages. If 2, print detailed messages.
    weighted: whether to apply observation-based weighting in binary loss

    Returns: (best_mu, best_avg_loss) where best_mu is the optimal mu and best_avg_loss is the average loss for that mu

    """
    assert V is not None or U is not None, "At least one of V or U must be provided"

    # Initialize KFold cross-validator
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)

    # Initialize objective sums
    sum_loss = {mu: 0 for mu in mu_values}

    # Determine observed entries and restrict to columns with missingness
    if U is not None and V is not None:
        X_concat = np.hstack((U, V))
        obs_mask = ~np.isnan(X_concat)
        col_has_missing = np.isnan(X_concat).any(axis=0)
        U = np.nan_to_num(U)
        V = np.nan_to_num(V)

    elif U is not None:
        obs_mask = ~np.isnan(U)
        col_has_missing = np.isnan(U).any(axis=0)
        U = np.nan_to_num(U)

    else:
        obs_mask = ~np.isnan(V)
        col_has_missing = np.isnan(V).any(axis=0)
        V = np.nan_to_num(V)

    all_obs_indices = np.argwhere(obs_mask)
    obs_indices = all_obs_indices[col_has_missing[all_obs_indices[:, 1]]]

    # Cross-validation loop
    for fold_idx, (_, test_idx) in enumerate(kf.split(obs_indices)):
        if verbose > 0:
            print(f"Processing Fold {fold_idx + 1}/{nfolds}")

        test_mask = obs_indices[test_idx]

        # Prepare training copies
        V_train = V.copy() if V is not None else None
        U_train = U.copy() if U is not None else None

        if U is not None and V is not None:
            m1 = U.shape[1]
            test_U_idx = test_mask[test_mask[:, 1] < m1]
            test_V_idx = test_mask[test_mask[:, 1] >= m1]

            U_test_mask = np.zeros_like(U, dtype=bool)
            V_test_mask = np.zeros_like(V, dtype=bool)

            U_test_mask[test_U_idx[:, 0], test_U_idx[:, 1]] = True
            V_test_mask[test_V_idx[:, 0], test_V_idx[:, 1] - m1] = True

            U_train[U_test_mask] = np.nan
            V_train[V_test_mask] = np.nan

        elif U is not None:
            U_test_mask = np.zeros_like(U, dtype=bool)
            U_test_mask[test_mask[:, 0], test_mask[:, 1]] = True
            U_train[U_test_mask] = np.nan

        else:  # V is not None
            V_test_mask = np.zeros_like(V, dtype=bool)
            V_test_mask[test_mask[:, 0], test_mask[:, 1]] = True
            V_train[V_test_mask] = np.nan

        # Evaluate each mu
        for mu in mu_values:
            if verbose:
                print(f"Evaluating mu = {mu: .2e}")

            # Perform imputation (assumes covImpute is defined elsewhere)
            X_hat, _ = covImpute(
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
            )

            if loss_fn is not None:
                if U is not None and V is not None:
                    X_true = np.hstack((U, V))
                    mask = np.zeros_like(X_true, dtype=bool)
                    mask[test_mask[:, 0], test_mask[:, 1]] = True
                elif U is not None:
                    X_true = U
                    mask = U_test_mask
                else:
                    X_true = V
                    mask = V_test_mask

                fold_loss = loss_fn(X_true, X_hat, mask)

            else:
                # Default to f1, f2, or f1+f2 depending on which blocks are present
                if U is not None and V is not None:
                    fold_loss = f1(U, U_test_mask, X_hat[:, :m1], Q, weighted) + f2(
                        V, V_test_mask, X_hat[:, m1:]
                    )
                elif U is not None:
                    fold_loss = f1(U, U_test_mask, X_hat, Q, weighted)
                else:
                    fold_loss = f2(V, V_test_mask, X_hat)

            if verbose > 0:
                print(f"Fold {fold_idx + 1}, mu = {mu}, Loss: {fold_loss:.4e}")

            sum_loss[mu] += fold_loss

    best_mu = min(sum_loss, key=sum_loss.get)
    best_avg_loss = sum_loss[best_mu] / nfolds

    if verbose > 0:
        print(f"Best mu: {best_mu:.2e}, Average Loss: {best_avg_loss:.4e}")

    return best_mu, best_avg_loss


def neg_f1_score(X_true, X_hat, test_mask):
    """
    Calculate the negative F1 score between the true and predicted values.

    X_true: True values (ground truth)
    X_hat: Predicted values
    test_mask: Mask indicating which values are involved in the test.
    """

    X_true_unobserved = X_true[test_mask].flatten()
    X_hat_unobserved = X_hat[test_mask].flatten()

    return -1 * f1_score(X_true_unobserved, X_hat_unobserved)


def neg_roc_auc(X_true, X_hat, test_mask):
    """
    Calculate the negative AUC-ROC score between the true and predicted values.

    X_true: True values (ground truth)
    X_hat: Predicted values
    test_mask: Mask indicating which values are involved in the test.
    """

    X_true_unobserved = X_true[test_mask].flatten()
    X_hat_unobserved = X_hat[test_mask].flatten()

    return -1 * roc_auc_score(X_true_unobserved, X_hat_unobserved)
