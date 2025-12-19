from utils import *
from tqdm import tqdm

def admm_sub(B,y,X,lambda_,kappa=2,max_iter=1000, eps=10e-5):
  """
  Subroutine of the admm to estimate U_i at each iteration
  kappa is a regularization parameter
  Use of SOC
  """

  dim_x = X.shape[1]


  N=B.shape[0]
  W=B.copy()
  M=np.zeros_like(B)
  n = len(y)
  y = y.reshape(-1, 1)

  B_prev = B.copy()

  # For faster computation in B update:
  Matrix_Inv = np.linalg.pinv(X.T @ X/n + kappa * np.eye(X.shape[1]))

  Xyproduct = X.T @ y

  for k in range(max_iter):

    # B update
    # SOC (Splitting Orthogonality Constraints) problem
    C_vec = vec(W - M).reshape(-1, 1)
    G_vec = Matrix_Inv @ (Xyproduct/n + kappa * C_vec) # close form solution of the problem without constraint
    G = vec_inv(G_vec, B.shape)
    U_svd, s_vector, Vt_svd = np.linalg.svd(G, full_matrices=False)

    B = U_svd @ Vt_svd  # projection to find back B

    # W update

    #Seen as a Lasso problem : min{ 1/2|W - Z|_F^2  + tau * |W|_1 }
    Z = B + M
    tau = lambda_/(kappa)

    # close form solution with soft-treshold
    W = np.sign(Z) * np.maximum(0, np.abs(Z) - tau)

    # M update

    M = M + B - W

    # Convergence condition

    change = normF(B - B_prev)/(normF(B_prev) + 10e-8)
    if change < eps:
          break
    B_prev = B.copy()

  return B

def alternating_direction_method_of_multipliers(A_0,ranks,y,T_all,P,regularisators,A_true=None,show = None, eps=10e-5,max_iter=1000):
  """
  Implementation of the alternating_direction_method_of_multipliers to calculate the SHORR estimator

  We suppose y of shape(N*(T_all+P)) and A_0 of shape(N,N,P)
  Regularisators should be an array of 4 elements ϱ
  """
  N = A_0.shape[0]

  #Regularisators
  lambda_ = regularisators[0]

  rho = [regularisators[1], regularisators[2], regularisators[3]]

  # Initialization of dual variables :
  C_1 = np.zeros(shape=ranks)
  C_2 = np.zeros(shape=ranks)
  C_3 = np.zeros(shape=ranks)

  C = [C_1, C_2, C_3]

  # Initialization of the decomposition :

  G, factors = hosvd_decomposition(A_0,ranks)
  U1 = factors[0]
  U2 = factors[1]
  U3 = factors[2]

  # Initialization of the mode decomposition :

  G_1 = tensor_mode(G,1)
  G_2 = tensor_mode(G,2)
  G_3 = tensor_mode(G,3)

  _, s_1, V1t = np.linalg.svd(G_1, full_matrices=False)
  _, s_2, V2t = np.linalg.svd(G_2, full_matrices=False)
  _, s_3, V3t = np.linalg.svd(G_3, full_matrices=False)
  V_1 = V1t.T
  V_2 = V2t.T
  V_3 = V3t.T
  D_1 = np.diag(s_1)
  D_2 = np.diag(s_2)
  D_3 = np.diag(s_3)

  V = [V_1, V_2, V_3]
  D = [D_1, D_2, D_3]

  A_previous = A_0


  #Will be useful in the minimization, so we don't have to recompute it each time
  y_list = []
  X_list = []
  x_list = []
  for t in range(P,T_all+P):
      y_t = y[:, t]
      X_t = y[:, t-P:t][:,::-1]
      x_t = vec(X_t)
      y_list.append(y_t)
      X_list.append(X_t)
      x_list.append(x_t)

  y_stack = np.concatenate(y_list).reshape(-1, 1) #to compute only once

  # MAIN LOOP

  for k in tqdm(range(max_iter)):

        G1 = tensor_mode(G,1)


        #minimization of U_1 with regularization and orthogonal condition:
        A_1_list=[]
        for t in range(T_all):
           x_t = x_list[t]
           term = x_t.T @ np.kron(U3, U2) @ G1.T
           A_1_t = np.kron(term, np.eye(N))
           A_1_list.append(A_1_t)

        A_1 = np.vstack(A_1_list)

        # use of the sub with the good parametres
        lambda_1 = lambda_ * norm1(U2) * norm1(U3)
        U1 = admm_sub(U1, y_stack, A_1, lambda_1)


        #minimization of U_2 with regularization and orthogonal condition:
        A_2_list=[]
        for t in range(T_all):
           X_t = X_list[t]
           term_K = np.kron((X_t @ U3).T, np.eye(ranks[1]))
           A_2_t = U1 @ G1 @ term_K
           A_2_list.append(A_2_t)

        A_2 = np.vstack(A_2_list)

        # use of the sub with the good parametres
        lambda_2 = lambda_ * norm1(U1) * norm1(U3)
        U2 = admm_sub(U2.T, y_stack, A_2, lambda_2).T # don't forget transposition


        #minimization of U_3 with regularization and orthogonal condition:
        A_3_list=[]
        for t in range(T_all):
           X_t = X_list[t]
           term_K = np.kron(np.eye(ranks[2]), U2.T @ X_t)
           A_3_t = U1 @ G1 @ term_K
           A_3_list.append(A_3_t)

        A_3 = np.vstack(A_3_list)

        # use of the sub with the good parametres
        lambda_3 = lambda_ * norm1(U1) * norm1(U2)
        U3 = admm_sub(U3,y_stack,A_3,lambda_3)


        #minimization of the core G
        # we minimize on vec(G_1),because vec(G_2) and vec(G_3) are permutation of vec(G_1)
        # We then can retrieve a classical least square problem

        A_4_list=[]
        for t in range(T_all):
           x_t = x_list[t]
           term_inner = np.kron(U3, U2).T @ x_t
           A_4_t = np.kron(term_inner.T, U1)
           A_4_list.append(A_4_t)

        A_4=np.vstack(A_4_list)

        dim_vec_G_1 = ranks[0] * ranks[1] * ranks[2]

        # We add the other terms (ridge):

        A_4_tot = np.vstack([A_4, np.sqrt(rho[0])* np.eye(dim_vec_G_1), np.sqrt(rho[1])* np.eye(dim_vec_G_1), np.sqrt(rho[2])* np.eye(dim_vec_G_1)])

        Target_1 = D[0] @ V[0].T - tensor_mode(C[0], 1)
        t_1 = Target_1.flatten(order='F').reshape(-1, 1)

        Target_2 = D[1] @ V[1].T - tensor_mode(C[1], 2)
        Target_2_tens = Target_2.reshape((ranks[1], ranks[0], ranks[2]), order='F')

        t_2 = Target_2_tens.transpose(1, 0, 2).flatten(order='F').reshape(-1, 1)


        Target_3 = D[2] @ V[2].T - tensor_mode(C[2], 3)
        Target_3_tens = Target_3.reshape((ranks[2], ranks[0], ranks[1]), order='F')
        t_3 = Target_3_tens.transpose(1, 2, 0).flatten(order='F').reshape(-1, 1)

        y_stack_tot = np.vstack([y_stack,np.sqrt(rho[0]) * t_1,np.sqrt(rho[1]) * t_2,np.sqrt(rho[2]) * t_3])

        vec_G_1 = np.linalg.lstsq(A_4_tot,y_stack_tot,rcond=None)[0] #the vec of the mode-1 unfolding of G

        G_1 = vec_inv(vec_G_1.flatten(), (ranks[0], ranks[1]*ranks[2]))
        G = G_1.reshape(ranks, order='F')


        # Now update for the new variable :

        for i in range(3):

            # unfolding the tensor under mode i+1 (cause i start from 0
            G_unfolded = tensor_mode(G, i + 1)
            C_unfolded = tensor_mode(C[i], i + 1)

            Target_A = G_unfolded + C_unfolded

            # D_i update
            # we solve directly on the d_i indepedently

            numerator = np.sum(Target_A * V[i].T, axis=1)
            denominator = np.sum(V[i].T**2, axis=1)
            d_vec = numerator / (denominator + 1e-10) #to avoid division by 0

            D[i] = np.diag(d_vec)

            # V_i udpate
            # Orthogonal Procrustes problem

            Target_A = G_unfolded + C_unfolded # same target

            D_i = D[i]

            #argmin_{V} -2Tr(D V^T A^T) = argmax_{V} Tr(V^T A^T D)
            Z = Target_A.T @ D_i

            U, S, Qt = np.linalg.svd(Z, full_matrices=False)
            V[i] = U @ Qt


            # C_i update
            C_unfolded = C_unfolded + G_unfolded - D[i] @ V[i].T # add the constraint diff

            # Just reshape in a good way
            if i == 0: #mode1
              C_tensor = C_unfolded.reshape(ranks, order='F')

            elif i == 1: #mode2
              C_tensor = C_unfolded.reshape((ranks[1], ranks[0], ranks[2]), order='F').transpose(1, 0, 2)

            elif i == 2: #mode3
              C_tensor = C_unfolded.reshape((ranks[2], ranks[0], ranks[1]), order='F').transpose(1, 2, 0)

            C[i] = C_tensor


        # Condition for convergence :
        A_current = product_3(product_2(product_1(G,U1),U2),U3)

        change = normF(A_current - A_previous)/(normF(A_previous) + 10e-8)

        if change < eps:
           if show:
            print('Convergence at iteration :' + str(k+1))
           break

        if show and k%10==0:
           error = normF(A_current - A_true)
           print('Iteration ' + str(k+1) + ' - Change: ' + str(change) + ' - Error: ' + str(error))

        A_previous = A_current

  if k==max_iter-1 and show:
    print('Convergence not reached')

  return A_current,G,U1,U2,U3

from scipy.linalg import eigh


def empirical_covariance(X): 
 
    X_centre = X - np.mean(X, axis=0)
   
    T = X.shape[0]
    
    cov = (1 / (T - 1)) * np.dot(X_centre.T, X_centre)
    
    return cov
def lambda_optimal(N,P,T,y):
    """
    Compute the optimal lambda as described in the paper
    """
    cov = empirical_covariance(y.T)
    M = eigh(a=cov, subset_by_index=(N-1,N-1), eigvals_only=True)[0]
    factor = np.sqrt(np.log(N**2 * P) / T)
    return M * factor * 4 