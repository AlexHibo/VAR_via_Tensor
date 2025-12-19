# High-Dimensional Vector Autoregressive Modeling via Tensor Decomposition

**Authors:** Alexandre Mallez & Antoine Le Maguet


## 📖 Overview & Full Report

This repository contains the implementation of high-dimensional Vector Autoregressive (VAR) models using tensor decomposition, based on the paper **"High-dimensional vector autoregressive time series modeling via tensor decomposition"** (Wang et al., 2020).

We address the "curse of dimensionality" in standard VAR models (where parameters grow quadratically with dimension $N$) by restructuring the transition matrices into a 3rd-order tensor and applying **Tucker decomposition**.

> **📄 For full details:** > For a comprehensive explanation of the mathematical framework, theoretical proofs (consistency, asymptotic variance), and detailed interpretation of the results, please refer to the **[Report.pdf](Report.pdf)** included in this repository.

## 📂 Repository Structure

The project is organized as follows:

* **`MAIN.ipynb`**: The main Jupyter Notebook. Open this to run the simulations, test on real data, and visualize the results.
* **`Report.pdf`**: The detailed project report.
* **`Code/`**: Folder containing all the source code and helper scripts.

### Detailed File Descriptions (`Code/` folder)

#### 1. Core Estimators
* **`Code/MLR.py` (Multilinear Low-Rank Estimator)**
    * **Algorithm:** Alternating Least Squares (ALS).
    * **Function:** Estimates the Tucker factors ($U_1, U_2, U_3$) and the core tensor $\mathcal{G}$ to minimize squared loss under a fixed multilinear rank constraint.
    * **Initialization:** Uses Higher-Order Singular Value Decomposition (HOSVD).

* **`Code/SHORR.py` (Sparse Higher-Order Reduced-Rank Estimator)**
    * **Algorithm:** ADMM (Alternating Direction Method of Multipliers).
    * **Function:** Handles cases where the true factors are both low-rank and sparse. Includes projection steps onto the Stiefel manifold (orthogonality constraints).

#### 2. Baselines & Benchmarking
* **`Code/Base_estimators.py`**
    * Includes standard implementations for comparison:
        * **OLS:** Ordinary Least Squares.
        * **RRR:** Reduced-Rank Regression (matrix-based rank reduction).
        * **Nuclear Norm:** Convex relaxation for low-rank estimation.
        * **Lasso:** $L_1$ regularized linear regression.

#### 3. Data Generation & Analysis
* **`Code/Data_generation.py`**
    * `generate_stationary_mlr_data`: Generates synthetic VAR processes with a true tensor low-rank structure while ensuring stationarity (spectral radius < 1).
    * `generate_sparse_data_simulation`: Generates data with block-sparse orthogonal factors for SHORR validation.

* **`Code/Rank_selection.py`**
    * Implements the data-driven rank selection method using the ratio of singular values of the unfolded tensor modes.

* **`Code/Asymptotic_Variance.py`**
    * Computes the theoretical asymptotic variance of the MLR estimator to validate consistency (Theorem 1 of the paper).

* **`Code/Bias_Variance_MLR.py`**
    * A script to visualize the Bias-Variance tradeoff of MLR vs. OLS/RRR as a function of sample size $T$.

* **`Code/utils.py`**
    * Contains essential tensor operations: Mode-$n$ products, unfolding (matricization), vectorization, and Kronecker products.

## ⚙️ Installation

To run the code, ensure you have Python 3 installed along with the required scientific libraries:

```bash
pip install numpy matplotlib tensorly tqdm
