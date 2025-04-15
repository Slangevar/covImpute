from sklearn.metrics import f1_score, roc_auc_score
import numpy as np


def higham(
    X,
    eps=1e-6,
    kappa=1e3,
    eig_tol=1e-6,
    maxit=1000,
    verbose=False,
):
    """
    Higham's algorithm to find the nearest positive semidefinite matrix.

    X: Input matrix
    eps: Tolerance for convergence of the higham algorithm
    kappa: The target condition number of the matrix
    eig_tol: Tolerance for eigenvalue convergence
    maxit: Maximum number of iterations
    verbose: If True, print the convergence criterion at each iteration
    """
    if not np.allclose(X, X.T, atol=1e-10):
        raise ValueError("Input matrix must be symmetric.")

    D_s = np.zeros_like(X)

    conv = np.inf

    for it in range(1, maxit + 1):
        Y = X
        R = Y - D_s
        d, Q = np.linalg.eigh(R)
        p = d > (eig_tol * d[-1])
        if np.all(p == False):
            raise ValueError("Matrix seems to be negative semidefinite.")

        # First part
        Q = Q[:, p]
        X = Q @ np.diag(d[p]) @ Q.T
        D_s = X - R
        conv = np.linalg.norm(Y - X, ord="fro") / np.linalg.norm(Y, ord="fro")

        if verbose:
            print(f"Iteration {it}: Convergence criterion = {conv:.6f}")

        if conv < eps:
            break

    # Second part for adjusting the condition number
    d, Q = np.linalg.eigh(X)
    min_val = d[-1] / kappa
    if d[0] < min_val:
        d[d < min_val] = min_val
    o_diag = np.diag(X)
    X = Q @ np.diag(d) @ Q.T
    D = np.sqrt(np.maximum(min_val, o_diag) / np.diag(X))
    X = D[:, None] * X * D[None, :]

    if conv >= eps:
        print(f"Warning: Higham's algorithm did not converge after {maxit} iterations.")
    return X


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
