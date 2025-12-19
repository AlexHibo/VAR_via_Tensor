from utils import *


### Ordinary Least Square Estimator ###


def ordinary_least_square(y,T_all,P):
  """
  Implementation of the ordinary least square algorithm
  We suppose y of shape(N*(T_all+P))
  """
  N,_=y.shape
  y_list=[]
  x_list=[]
  for t in range(P,T_all+P):
      y_t = y[:, t]
      X_t = y[:, t-P:t][:,::-1] 
      x_t = vec(X_t)
      y_list.append(y_t)
      x_list.append(x_t)
  y_stack = np.vstack(y_list)
  x_stack = np.vstack(x_list)
  A_OLM_1 = np.linalg.lstsq(x_stack, y_stack, rcond=None)[0]
  A_OLM = A_OLM_1.T.reshape((N,N,P), order='F')
  return A_OLM


### Reduced Rank Regression Estimator###


def reduced_rank_regression(y, T_all, P, rank):
    """
    Implementation of the Reduced-Rank Regression (RRR) estimator.

    It computes the OLS estimator first, then projects it onto
    the best rank-r approximation using SVD truncation.

    Args:
        y: Time series data
        T_all: Effective time steps
        P: Number of lags
        rank: The target rank (r1 in the paper context)

    Returns:
        A_RRR: The estimated transition tensor of shape (N, N, P)
    """
    N = y.shape[0]

    # OLS estimator
    A_ols_tensor = ordinary_least_square(y, T_all, P)
    A_ols_mat = tensor_mode(A_ols_tensor, 1)

    #SVD
    U, S, Vt = np.linalg.svd(A_ols_mat, full_matrices=False)

    S_truncated = np.zeros_like(S)
    S_truncated[:rank] = S[:rank]

    # final tensor
    A_rrr_mat = U @ np.diag(S_truncated) @ Vt

    A_rrr_tensor = np.zeros((N, N, P))
    for k in range(P):
        A_rrr_tensor[:, :, k] = A_rrr_mat[:, k*N : (k+1)*N]

    return A_rrr_tensor


### Functions for Lasso estimator ###


def soft_thresholding(x, lambda_):
    """
    Using of a thresholding
    S_lambda(x) = sign(x) * max(|x| - lambda, 0)
    """
    return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

def lasso_solver(X, Y, alpha, max_iter=2000, tol=1e-4):
    """
    Solve lasso with gradient descent
    Minimise: (1 / (2 * n_samples)) * ||Y - XW||^2 + alpha * ||W||_1
    """
    n_samples, n_features = X.shape
    _, n_targets = Y.shape

    # weights
    W = np.zeros((n_features, n_targets))

    # Learning rate
    L = np.linalg.norm(X, ord=2) ** 2 / n_samples
    learning_rate = 1 / L

    for i in range(max_iter):
        W_old = W.copy()

        # Gradient
        residuals = X @ W - Y
        gradient = (X.T @ residuals) / n_samples
        W_gradient_step = W - learning_rate * gradient

        # use of threshold
        W = soft_thresholding(W_gradient_step, alpha * learning_rate)

        # Convergence
        if np.linalg.norm(W - W_old) < tol:
            break

    return W

def lasso_regression(y, T_eff, P_lags, alpha=0.01):
    """
    Lasso estimator
    """
    N = y.shape[0]

    # Data
    y_list = []
    x_list = []
    for t in range(P_lags, T_eff + P_lags):
        y_list.append(y[:, t])
        x_list.append(y[:, t-P_lags:t][:, ::-1].flatten(order='F'))

    Y_target = np.vstack(y_list)   # Dimension: T x N
    X_features = np.vstack(x_list) # Dimension: T x NP

    W_est = lasso_solver(X_features, Y_target, alpha=alpha)

    # Result
    A_lasso_mat = W_est.T

    # Reshape
    A_lasso_tensor = A_lasso_mat.reshape(N, N, P_lags, order='F')

    return A_lasso_tensor



### Nuclear Norm Estimator ###


def nuclear_norm_solver_proximal(Y, X, lambda_val, max_iter=1000, tol=1e-4):
    """
    Résout: min (1/T) * || Y - A X ||_F^2 + lambda * || A ||_*
    with a gradient descent
    """
    N, T_eff = Y.shape
    NP = X.shape[0]  # (NP*T)

    # Learning step
    s_max_X = np.linalg.norm(X, ord=2)
    L = (2.0 / T_eff) * (s_max_X ** 2)
    step_size = 1.0 / L

    A = np.zeros((N, NP))

    for i in range(max_iter):
        A_old = A.copy()

        # gradient descent
        residuals = A @ X - Y
        gradient = (2.0 / T_eff) * (residuals @ X.T)

        A_temp = A - step_size * gradient
        U, S, Vt = np.linalg.svd(A_temp, full_matrices=False)

        # thrshold
        threshold = lambda_val * step_size
        S_new = np.maximum(S - threshold, 0)

        # Matrix with SVD
        A = U @ np.diag(S_new) @ Vt

        # Convergence
        if np.linalg.norm(A - A_old, 'fro') < tol:
            break

    return A

def nuclear_norm_estimator(y, T_eff, P_lags, lambda_val):
    """
    Compute the nuclear norm estimator
    """
    N = y.shape[0]

    # Data
    y_list = []
    x_list = []

    for t in range(P_lags, T_eff + P_lags):
        y_list.append(y[:, t])
        lagged_window = y[:, t-P_lags:t][:, ::-1]
        x_list.append(lagged_window.flatten(order='F'))

    Y_mat = np.vstack(y_list).T # N x T_eff
    X_mat = np.vstack(x_list).T # NP x T_eff

    A_hat_mat = nuclear_norm_solver_proximal(Y_mat, X_mat, lambda_val)

    # estimated tensor
    A_nn_tensor = np.zeros((N, N, P_lags))
    for k in range(P_lags):
        A_nn_tensor[:, :, k] = A_hat_mat[:, k*N : (k+1)*N]

    return A_nn_tensor

# used to find lambda for nuclear norm estimator

def grid_search_lambda_nn(y, P_lags, lambda_grid, split_ratio=0.8):
    """
    Grid Search  to find lambda for nuclear norm estimator
    """
    N, T_total = y.shape
    split_idx = int(T_total * split_ratio)

    # 1. Time Series Split
    # Train until se split
    y_train = y[:, :split_idx]
    T_eff_train = y_train.shape[1] - P_lags

    # Validation : the rest
    # We need the P last point of train for the validation
    y_val = y[:, split_idx:]
    T_val = y_val.shape[1]

    best_lambda = None
    best_mse = float('inf')
    results = {}

    for lam in lambda_grid:
        # train on test set
        try:
            A_est = nuclear_norm_estimator(y_train, T_eff_train, P_lags, lam)
        except Exception:
            # if divergence
            continue

        # Rolling Forecast 
        val_errors = []

        # We initialize with th end of the train set
        current_history = y[:, split_idx-P_lags : split_idx] # (N, P)

        for t in range(T_val):
            # True
            y_true = y_val[:, t]

            # Prediction
            y_hat = predict_var_step(A_est, current_history)

            # quadratic error
            mse_t = np.mean((y_true - y_hat)**2)
            val_errors.append(mse_t)

            # Update of history
            current_history = np.hstack([current_history[:, 1:], y_true.reshape(-1, 1)])

        mean_mse = np.mean(val_errors)
        results[lam] = mean_mse


        if mean_mse < best_mse:
            best_mse = mean_mse
            best_lambda = lam


    return best_lambda, results