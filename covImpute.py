import numpy as np
from sklearn.model_selection import KFold


def obj_fun(Y, X_hat, mask):
    return 0.5 * np.sum(((Y - X_hat)[~mask]) ** 2)


def covImpute(
    X: np.ndarray,
    E: np.ndarray,
    mu: float,
    maxit: int = 1000,
    tol: float = 1e-3,
    patience: int = 3,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform matrix imputation using a fast gradient method with covariance regularization.

    Parameters:
    X (np.ndarray): Input data matrix with missing values (NaNs).
    E (np.ndarray): Covariance matrix.
    mu (float): Regularization parameter.
    maxit (int, optional): Maximum number of iterations. Default is 1000.
    tol (float, optional): Tolerance for convergence. Default is 1e-3.
    patience (int, optional): Number of iterations to wait for significant improvement before stopping. Default is 3.
    verbose (bool, optional): If True, print progress messages. Default is False.

    Returns:
    tuple[np.ndarray, np.ndarray]: Imputed matrix and matrix with observed entries restored.
    """

    # Create mask for missing entries
    mask = np.isnan(X)

    # Initialize variables
    Y = np.nan_to_num(X)  # Replace NaNs with 0
    X_hat = Y.copy()

    # Compute eigen decomposition of E
    eigvals, eigvecs = np.linalg.eigh(E)  # Use eigh for symmetric matrices
    gamma_min, gamma_max = np.min(eigvals), np.max(eigvals)

    # Precompute constants
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
    old_obj = np.inf

    # Iterative optimization
    for it in range(1, maxit + 1):
        X_hat_old = X_hat.copy()

        # Gradient step
        X_hat = Y.copy()
        X_hat[~mask] += step_size * (X[~mask] - Y[~mask])
        X_hat -= step_size * mu * (Y @ E_inv)

        # Momentum update
        Y = X_hat + beta * (X_hat - X_hat_old)

        # Compute objective function
        imputation_error = obj_fun(X, X_hat, mask)
        regularization = 0.5 * mu * np.linalg.norm(X_hat @ E_inv_sqrt, "fro") ** 2
        objective = imputation_error + regularization

        # Check for convergence
        change_in_obj = np.abs(old_obj - objective)
        if change_in_obj < tol:
            patience_counter += 1
        else:
            patience_counter = 0
        old_obj = objective

        # Verbose output
        if verbose:
            print(
                f"Iter: {it}, Objective: {objective:.4e}, Change: {change_in_obj:.4e}"
            )

        if patience_counter >= patience:
            if verbose:
                print(f"Converged after {it} iterations.")
            break
    else:
        if verbose:
            print(f"Did not converge after {maxit} iterations.")

    # Restore observed entries
    final_result = X_hat.copy()
    final_result[~mask] = X[~mask]

    return X_hat, final_result


def cv_mu(
    X: np.ndarray,
    E: np.ndarray,
    mu_values: list,
    nfolds: int = 5,
    maxit: int = 1000,
    tol: float = 1e-3,
    patience: int = 3,
    seed: int = 9727,
    verbose: bool = False,
):
    """
    Perform cross-validation to find the best choice of mu for covImpute.

    Parameters:
    X (np.ndarray): Input data matrix with missing values (NaNs).
    E (np.ndarray): Covariance matrix.
    mu_values (list): List of regularization parameters to evaluate.
    nfolds (int, optional): Number of folds for cross-validation. Default is 5.
    maxit (int, optional): Maximum number of iterations. Default is 1000.
    tol (float, optional): Tolerance for convergence. Default is 1e-3.
    patience (int, optional): Number of iterations to wait for significant improvement before stopping. Default is 3.
    seed (int, optional): Random seed for reproducibility. Default is 9727.
    verbose (bool, optional): If True, print progress messages. Default is False.

    Returns:
    tuple[float, dict]: Best mu value and dictionary of sum of objective values for each mu.
    """

    # Get observed (non-NaN) indices
    obs_indices = np.array(list(zip(*np.where(~np.isnan(X)))))

    # Initialize KFold cross-validator
    kf = KFold(n_splits=nfolds, shuffle=True, random_state=seed)

    # Initialize objective sums
    sum_obj = {mu: 0 for mu in mu_values}

    # Cross-validation loop
    for fold_idx, (_, test_idx) in enumerate(kf.split(obs_indices)):
        if verbose:
            print(f"Processing Fold {fold_idx + 1}/{nfolds}")

        # Create a copy of X for training
        X_train = X.copy()

        # Mask the test set
        test_indices = obs_indices[test_idx]
        test_mask = np.ones_like(X, dtype=bool)
        test_mask[test_indices[:, 0], test_indices[:, 1]] = False

        # Assign NaN to test set in X_train
        X_train[test_indices[:, 0], test_indices[:, 1]] = np.nan

        # Evaluate each mu
        for mu in mu_values:
            if verbose:
                print(f"Evaluating mu = {mu}")

            # Perform imputation (assumes covImpute is defined elsewhere)
            _, X_hat = covImpute(X_train, E, mu, maxit, tol, patience)

            # Compute objective function for this fold and mu
            obj = obj_fun(X, X_hat, test_mask)
            sum_obj[mu] += obj

    # Find the best mu (minimizing objective)
    best_mu = min(sum_obj, key=sum_obj.get)

    return best_mu, sum_obj
