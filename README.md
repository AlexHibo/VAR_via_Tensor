# High-Dimensional Vector Autoregressive Modeling via Tensor Decomposition

**Authors:** Alexandre Mallez, Antoine Le Maguet

## 📖 Overview

This repository contains the implementation and analysis of high-dimensional Vector Autoregressive (VAR) models using tensor decomposition techniques. The project is based on the paper **"High-dimensional vector autoregressive time series modeling via tensor decomposition"** (Wang et al., 2020).

Standard VAR models suffer from the "curse of dimensionality," as the number of parameters ($N^2P$) grows quadratically with the number of time series. This project implements two estimators that restructure the transition matrices into a 3rd-order tensor and apply **Tucker decomposition** to enforce low-rank structures, significantly reducing the parameter space.

## 📂 Repository Structure

The codebase is organized into model implementations, data generation, baselines, and experimental scripts.

### 🚀 Main Execution
* **`MAIN.ipynb`** The central notebook orchestrating the project. It demonstrates:
    * Generation of synthetic data (Stationary Low-Rank and Sparse structures).
    * Training of **MLR** and **SHORR** models.
    * Comparison with baselines (OLS, RRR, Lasso, Nuclear Norm).
    * Visualization of signal reconstruction and convergence metrics.

### 🧠 Core Estimators
* **`MLR.py` (Multilinear Low-Rank Estimator)** * **Method:** Implements the **Alternating Least Squares (ALS)** algorithm to estimate the Tucker factors ($U_1, U_2, U_3$) and the core tensor $\mathcal{G}$.
    * **Initialization:** Uses Higher-Order Singular Value Decomposition (HOSVD).
    * **Goal:** Minimizes the squared loss under a fixed multilinear rank constraint.

* **`SHORR.py` (Sparse Higher-Order Reduced-Rank Estimator)** * **Method:** Implements the **ADMM (Alternating Direction Method of Multipliers)** algorithm.
    * **Features:** Enforces both low-rank structure (via orthogonality constraints on factors) and element-wise sparsity (via $L_1$ regularization).
    * **Subroutines:** Includes projection steps onto the Stiefel manifold.

### 📉 Baseline Models
* **`Base_estimators.py`** Contains standard and regularized approaches for performance benchmarking:
    * **OLS:** Ordinary Least Squares (Standard VAR, prone to overfitting in high dimensions).
    * **RRR:** Reduced-Rank Regression (Low-rank constraint on the unfolded matrix).
    * **Nuclear Norm:** Convex relaxation for low-rank estimation.
    * **Lasso:** $L_1$ penalized regression for sparsity.

### 📊 Data & Experiments
* **`Data_generation.py`** * **`generate_stationary_mlr_data`**: Simulates VAR processes with a true tensor low-rank structure. Ensures stationarity by controlling the spectral radius of the companion matrix.
    * **`generate_sparse_data_simulation`**: Generates data with block-sparse orthogonal factors for testing SHORR.

* **`Rank_selection.py`** * Implements the data-driven rank selection method proposed by Wang et al.
    * Uses the ratio of singular values of the unfolded tensor modes to automatically determine the optimal multilinear ranks $(r_1, r_2, r_3)$.

* **`Asymptotic_Variance.py`** * Computes the theoretical asymptotic variance of the estimators (Theorem 1 of the paper).
    * Used to validate the **consistency** of the MLR estimator by comparing theoretical bounds with empirical simulation results.

* **`Bias_Variance_MLR.py`** * A standalone script to visualize the **Bias-Variance tradeoff**.
    * Compares MLR, OLS, and RRR errors as a function of sample size $T$,
