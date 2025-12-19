from utils import *

def select_ranks(A_init, T_sample, N, P, constant_c=None):
    """
    estimate the rank of our data
    """
    # constant C
    if constant_c is None:
        # c = sqrt(NP * log(T) / 10T) as the paper advice
        constant_c = np.sqrt((N * P * np.log(T_sample)) / (10 * T_sample))

    estimated_ranks = []

    for i in range(1, 4): # Modes 1, 2, 3
        mat_i = tensor_mode(A_init, i)

        # SVD
        singular_values = np.linalg.svd(mat_i, compute_uv=False)

        # Ridge Ratio
        ratios = []
        limit = len(singular_values) - 1

        for j in range(limit):
            sigma_j = singular_values[j]        # sigma_r
            sigma_j_plus_1 = singular_values[j+1] # sigma_{r+1}

            # paper formula
            ratio = (sigma_j_plus_1 + constant_c) / (sigma_j + constant_c)
            ratios.append(ratio)

        # minimise
        best_rank = np.argmin(ratios) + 1
        estimated_ranks.append(best_rank)

    return estimated_ranks



from Data_generation import generate_stationary_mlr_data
import matplotlib.pyplot as plt
from Base_estimators import nuclear_norm_estimator,grid_search_lambda_nn, ordinary_least_square
from tqdm import tqdm

# Verify the consistency of the rank selection method through simulations
# We use the nuclear norm estimator to estimate A_init as described in the paper

def verify_rank_consistency_simulation():
  
    N = 10          # Variables
    P = 5           # Lags
    true_ranks = [3,3,3] # traget ranks

    # T values to test (T)
    T_values = [ 100, 400, 800, 1500, 2500, 3000, 4000]
    n_simulations = 50 # Number of repetition for each T (1000 in the paper, 50 is faster)

    success_rates = []

    print(f"Verification of consistency for the ranks : {true_ranks}...")

    for T in T_values:
        success_count = 0

        for sim in tqdm(range(n_simulations)):
            if len(set(true_ranks)) == 1:
                y_sim, A_true = generate_stationary_mlr_data(N, P, T, true_ranks, target_rho=0.9, eigenvalues=[2, 2, 2, 2])
            else:
                y_sim, A_true = generate_stationary_mlr_data(N, P, T, true_ranks, target_rho=0.9, eigenvalues=None)
            T_eff = y_sim.shape[1] - P

            #lambda_theory = 0.2 * np.sqrt((N * P) / T_eff)
            #Nuclear norm
            A_nn = ordinary_least_square(y_sim, T_eff, P)

            # rank
            estimated_ranks = select_ranks(A_nn, T_eff, N, P)


            if estimated_ranks == true_ranks:
                success_count += 1

        rate = success_count / n_simulations
        success_rates.append(rate)
        print(f"T={T}: Success rate = {rate:.2f}")

    # --- Vizualisation ---
    plt.figure(figsize=(8, 5))
    plt.plot(T_values, success_rates, 'o-', color='purple', label='Ridge-Type Ratio Estimator')
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel('Taille de l\'échantillon (T)')
    plt.ylabel('Proportion de Rangs Correctement Sélectionnés')
    plt.title('Vérification de la Consistance de la Sélection de Rang')
    plt.ylim(-0.05, 1.05)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    verify_rank_consistency_simulation()