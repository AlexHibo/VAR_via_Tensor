from utils import *
from tqdm import tqdm
import matplotlib.pyplot as plt
from Data_generation import generate_stationary_mlr_data
from Base_estimators import ordinary_least_square, reduced_rank_regression
from MLR import alternating_least_square

N = 8     # Variables
P = 4     # Lags
T = 300   # Temps de simulation
ranks = [3, 3, 2] # Rangs multilinaire

_ , A_true = generate_stationary_mlr_data(N, P, 1, ranks, target_rho=0.90, noise_std=0.1)
noise_std = 0.1
n_estimates = 20  # Number of simulation
list_T = [100, 200, 300, 600, 1000, 1500]# samples size

#bias
BiasSq_MLR, BiasSq_OLS, BiasSq_RRR = [], [], []

#Variance
Var_MLR, Var_OLS, Var_RRR = [], [], []


for T in list_T:


    est_MLR = []
    est_OLS = []
    est_RRR = []

    for i in tqdm(range(n_estimates)):
        # data
        y_sim = np.zeros((N, T + P + 100))
        epsilon = np.random.randn(N, T + P + 100) * noise_std
        A_mat = tensor_mode(A_true, 1) # [A1...AP]

        for t in range(P, T + P + 100):
            lagged_values = y_sim[:, t-P:t][:, ::-1]
            x_t = vec(lagged_values)
            y_sim[:, t] = A_mat @ x_t + epsilon[:, t]

        y_sim = y_sim[:, 100:] # Burn-in
        T_eff = y_sim.shape[1] - P

        # MLR
        A_init = A_true + np.random.randn(*A_true.shape) * 0.2
        A_mlr, _, _, _, _ = alternating_least_square(A_init, ranks, y_sim, T_eff, P)
        est_MLR.append(A_mlr)

        # OLS
        A_ols = ordinary_least_square(y_sim, T_eff, P)
        est_OLS.append(A_ols)

        # RRR (Rang r1)
        A_rrr = reduced_rank_regression(y_sim, T_eff, P, ranks[0])
        est_RRR.append(A_rrr)

    # Squared bias
    mean_MLR = np.mean(est_MLR, axis=0)
    mean_OLS = np.mean(est_OLS, axis=0)
    mean_RRR = np.mean(est_RRR, axis=0)

    BiasSq_MLR.append(np.mean((mean_MLR - A_true)**2))
    BiasSq_OLS.append(np.mean((mean_OLS - A_true)**2))
    BiasSq_RRR.append(np.mean((mean_RRR - A_true)**2))

    # variance
    Var_MLR.append(np.mean(np.var(est_MLR, axis=0)))
    Var_OLS.append(np.mean(np.var(est_OLS, axis=0)))
    Var_RRR.append(np.mean(np.var(est_RRR, axis=0)))

#Vizualisation
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Squared Bias
ax1.plot(list_T, BiasSq_OLS, marker='o', label='OLS', color='salmon', linestyle='-')
ax1.plot(list_T, BiasSq_RRR, marker='^', label='RRR', color='limegreen', linestyle='-')
ax1.plot(list_T, BiasSq_MLR, marker='s', label='MLR', color='cornflowerblue', linestyle='-')

ax1.set_xlabel('T (Sample Size)')
ax1.set_ylabel('Squared Bias')
ax1.set_title('Squared Bias vs Sample Size')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Variance
ax2.plot(list_T, Var_OLS, marker='o', label='OLS', color='salmon', linestyle='-')
ax2.plot(list_T, Var_RRR, marker='^', label='RRR', color='limegreen', linestyle='-')
ax2.plot(list_T, Var_MLR, marker='s', label='MLR', color='cornflowerblue', linestyle='-')

ax2.set_xlabel('T (Sample Size)')
ax2.set_ylabel('Variance')
ax2.set_title('Variance vs Sample Size')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()