from utils import *


### Functions to generate stationarry with reduce rank datas ###

def get_spectral_radius(A_tensor):
    """Estimate the spectral radius of our matrix A by using companion matrix

    input : A_tensor """
    N, _, P = A_tensor.shape
    A_mat = tensor_mode(A_tensor,1) # [A1, ..., AP]

    # Use of a companion matrix to ensure stationarry data as explained in the paper.
    companion = np.zeros((N*P, N*P))
    companion[:N, :] = A_mat
    if P > 1:
        companion[N:, :-N] = np.eye(N * (P - 1))

    eigenvalues = np.linalg.eigvals(companion)
    return np.max(np.abs(eigenvalues))

def generate_stationary_mlr_data(N, P, T, ranks, target_rho=0.95, noise_std=0.1,show=False,eigenvalues=None):
    r1, r2, r3 = ranks

    # Orthonormalization
    U1 = np.linalg.qr(np.random.randn(N, r1))[0]
    U2 = np.linalg.qr(np.random.randn(N, r2))[0]
    U3 = np.linalg.qr(np.random.randn(P, r3))[0]

    # core tensor
    if eigenvalues is not None:
        G = np.zeros((r1, r2, r3))
        
        limit = min(r1, r2, r3, len(eigenvalues))
        for i in range(limit):
            G[i, i, i] = eigenvalues[i]
             
    else:
        G = np.random.randn(r1, r2, r3)

    # stationarity
    A_tensor = reconstruct_tensor(G, U1, U2, U3)
    rho = get_spectral_radius(A_tensor)

    if show:
        print(f"initial spectral radius: {rho:.4f}")

    decay_factor = 0.95 # On réduit de 5% à chaque échec
    iter_count = 0

    while rho >= target_rho:
        G = G * decay_factor
        A_tensor = reconstruct_tensor(G, U1, U2, U3)
        rho = get_spectral_radius(A_tensor)
        iter_count += 1
        if iter_count > 2000: raise ValueError("Don't find a solution")

    if iter_count > 0:
        if show:
            print(f"Succes : final spectral radius {rho:.4f}")

    # burn in to stabilize model
    burn_in = 100
    y = np.zeros((N, T + P + burn_in))
    epsilon = np.random.randn(N, T + P + burn_in) * noise_std
    A_mat = tensor_mode(A_tensor,1)

    for t in range(P, T + P + burn_in):
        # Lags [y_{t-P}, ..., y_{t-1}]
        lagged_values = y[:, t-P:t][:, ::-1]
        x_t = vec(lagged_values)
        y[:, t] = A_mat @ x_t + epsilon[:, t]

    # final value
    return y[:, burn_in:], A_tensor


### Sparse generation ###


def generate_block_sparse_ortho(rows, cols, s, max_iter=2000, tol=1e-6):
    """
    Generate a sparse orthogonal matrix

    Use the Alternating Projections method
    """
    # 1. Random initialization
    U = np.random.randn(rows, cols)

    for i in range(max_iter):
        # Equivalent to procruste problema (projection with SVD)
        P, S, Qt = np.linalg.svd(U, full_matrices=False)
        U_ortho = P @ Qt

        # Ensure sparcity (remove 0column)
        U_sparse = np.zeros_like(U_ortho)
        for j in range(cols):
            col = U_ortho[:, j]
            # keep only the greatest values s
            idx = np.argsort(np.abs(col))[::-1][:s]
            U_sparse[idx, j] = col[idx]

        # Verification of convergence (orthogonality and no too much displacement)
        diff = np.linalg.norm(U_sparse - U, 'fro')
        ortho_error = np.linalg.norm(U_sparse.T @ U_sparse - np.eye(cols), 'fro')

        U = U_sparse

        if diff < tol and ortho_error < 1e-2:
            # return if sparse and orthogonal
            return U_sparse

    # N too small, not orthogonal (retry)
    return generate_block_sparse_ortho(rows, cols, s)

# Adapt data to shorr
def generate_sparse_data_simulation(N, P, T, ranks, sparsity, target_rho=0.95,noise_std=0.1):
    r1, r2, r3 = ranks
    s1, s2, s3 = sparsity

    # Generate Ui
    U1 = generate_block_sparse_ortho(N, r1, s1)
    U2 = generate_block_sparse_ortho(N, r2, s2)
    U3 = generate_block_sparse_ortho(P, r3, s3)

    # Core
    G = np.random.randn(r1, r2, r3)

    #final tensor
    A_tensor = reconstruct_tensor(G, U1, U2, U3)

    # stationary data
    rho = get_spectral_radius(A_tensor)

    decay = 0.95
    while rho >= target_rho: # Target rho
        G = G * decay
        A_tensor = reconstruct_tensor(G, U1, U2, U3)
        rho = get_spectral_radius(A_tensor)


    # create Y
    burn_in = 100
    y = np.zeros((N, T + P + burn_in))
    epsilon = np.random.randn(N, T + P + burn_in) * noise_std

    A_mat = tensor_mode(A_tensor, 1)

    for t in range(P, T + P + burn_in):
        lagged_values = y[:, t-P:t][:, ::-1]
        x_t = vec(lagged_values)

        y[:, t] = A_mat @ x_t + epsilon[:, t]

    return y[:, burn_in:], A_tensor, G, [U1, U2, U3]
