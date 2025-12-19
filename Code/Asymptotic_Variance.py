from utils import *
from numpy.linalg import pinv

def calculate_asymptotic_variance_MLR(y, A_est, G, U1, U2, U3, T_eff, P):
    """
    Calculates the Asymptotic Variance of the MLR estimator based on Theorem 1.
    """
    N = y.shape[0]

    # Data
    y_list = []
    x_list = []
    for t in range(P, T_eff + P):
        y_t = y[:, t]
        X_t = y[:, t-P:t][:, ::-1]
        x_t = vec(X_t)
        y_list.append(y_t)
        x_list.append(x_t)

    Y_stack = np.vstack(y_list).T # N x T
    X_stack = np.vstack(x_list).T # NP x T

    # Covariances
    A_mat = tensor_mode(A_est, 1)
    E = Y_stack - A_mat @ X_stack

    # Sigma_epsilon
    Sigma_eps = (E @ E.T) / T_eff

    # Gamma_star (Predictor Covariance)
    Gamma_star = (X_stack @ X_stack.T) / T_eff

    # Information matrix
    try:
        Sigma_inv = np.linalg.inv(Sigma_eps)
    except:
        Sigma_inv = pinv(Sigma_eps)

    J = np.kron(Sigma_inv, Gamma_star)

    # Jacobian
    H_G = np.kron(np.kron(U3, U2), U1)

    term_U1 = np.kron(U3, U2) @ tensor_mode(G, 1).T
    H_U1 = np.kron(term_U1, np.eye(N))

    term_U2 = np.kron(U1, U3) @ tensor_mode(G, 2).T
    H_U2_raw = np.kron(term_U2, np.eye(N)) # This gives derivative of vec(A_(2))

    H_U2 = np.zeros_like(H_U2_raw)
    for col in range(H_U2_raw.shape[1]):
        vec_A2_prime = H_U2_raw[:, col]
        # Reshape to A_(2) shape (J, K*I) -> (N, P*N)
        A2_prime = vec_inv(vec_A2_prime, (N, P*N))
        A_prime_tensor = A2_prime.reshape(N, P, N, order='F').transpose(2, 0, 1)

        A1_prime = tensor_mode(A_prime_tensor, 1)
        H_U2[:, col] = vec(A1_prime)

    term_U3 = np.kron(U1, U2) @ tensor_mode(G, 3).T
    H_U3_raw = np.kron(term_U3, np.eye(P))

    H_U3 = np.zeros_like(H_U3_raw)
    for col in range(H_U3_raw.shape[1]):
        vec_A3_prime = H_U3_raw[:, col]
        # Reshape to A_(3) shape (K, I*J) -> (P, N*N)
        A3_prime = vec_inv(vec_A3_prime, (P, N*N))
        A_prime_tensor = A3_prime.reshape(P, N, N, order='F').transpose(2, 1, 0)

        A1_prime = tensor_mode(A_prime_tensor, 1)
        H_U3[:, col] = vec(A1_prime)

    # Concatenate H blocks
    H = np.hstack([H_G, H_U1, H_U2, H_U3])

    # Sigma MLR
    Inner = H.T @ J @ H
    Inner_inv = pinv(Inner)

    Sigma_MLR = H @ Inner_inv @ H.T

    # variance
    avar = np.trace(Sigma_MLR) / len(Sigma_MLR)

    return Sigma_MLR, avar

def calculate_covariances_OLS_RRR(y, A_est, T_eff, P_lags, rank_r1):
    """
    Calculates Asymptotic Covariance Matrices and Average Variances (AVar)
    for OLS and RRR estimators based on Corollary 2.
    """
    N = y.shape[0]

    # Information matric
    y_list = []
    x_list = []
    for t in range(P_lags, T_eff + P_lags):
        y_t = y[:, t]
        X_t = y[:, t-P_lags:t][:, ::-1]
        x_t = vec(X_t)
        y_list.append(y_t)
        x_list.append(x_t)

    Y_stack = np.vstack(y_list).T # N x T
    X_stack = np.vstack(x_list).T # NP x T

    # Residuals & Covariances using the MLR estimate as the center point
    A_mat = tensor_mode(A_est, 1)
    E = Y_stack - A_mat @ X_stack

    Sigma_eps = (E @ E.T) / T_eff
    Gamma_star = (X_stack @ X_stack.T) / T_eff

    try:
        Sigma_inv = np.linalg.inv(Sigma_eps)
    except:
        Sigma_inv = pinv(Sigma_eps)

    J = np.kron(Sigma_inv, Gamma_star)

    # OLS Covariance
    Sigma_OLS = pinv(J)
    avar_ols = np.trace(Sigma_OLS) / len(Sigma_OLS)

    # RRR covariance
    U_full, S_full, Vt_full = np.linalg.svd(A_mat, full_matrices=False)

    U = U_full[:, :rank_r1]
    D_diag = S_full[:rank_r1]
    D = np.diag(D_diag)
    V = Vt_full[:rank_r1, :].T # V is (NP x r1)

    # Jacobian
    R_U = np.kron(V @ D, np.eye(N))

    P_diag = np.zeros((rank_r1 * rank_r1, rank_r1))
    for i in range(rank_r1):
        P_diag[i*rank_r1 + i, i] = 1
    R_D = np.kron(V, U) @ P_diag

    m, n = (N*P_lags), rank_r1
    K_mn = np.zeros((m*n, m*n))
    for i in range(m):
        for j in range(n):
            row = i * n + j
            col = j * m + i
            K_mn[row, col] = 1

    R_V = np.kron(np.eye(N*P_lags), U @ D) @ K_mn

    # Concatenate R blocks
    R = np.hstack([R_U, R_D, R_V])

    # Sigma_RRR = R (R' J R)^+ R'
    Inner_RRR = R.T @ J @ R
    Inner_inv_RRR = pinv(Inner_RRR)
    Sigma_RRR = R @ Inner_inv_RRR @ R.T
    avar_rrr = np.trace(Sigma_RRR) / len(Sigma_RRR)

    return Sigma_OLS, avar_ols, Sigma_RRR, avar_rrr
