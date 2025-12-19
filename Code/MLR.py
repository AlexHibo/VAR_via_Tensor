from utils import *

# Alternating Least Square Algorithm for MLR estimator

def alternating_least_square(A_0,ranks,y,T_all,P,A_true=None,show=False,eps=10e-5,max_iter=1000):
  """
  Implementation of the alternating least square algorithm

  We suppose y of shape(N*(T_all+P)) and A_0 of shape(N,N,P)
  """
  N = y.shape[0]
  G, factors = hosvd_decomposition(A_0,ranks)
  U1 = factors[0]
  U2 = factors[1]
  U3 = factors[2]

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

  y_stack = np.concatenate(y_list).reshape(-1, 1) #to computate only once

  # MAIN LOOP

  for k in range(max_iter):

        G1 = tensor_mode(G,1)

        #minimization of U_1:
        A_1_list=[]
        for t in range(T_all):
           x_t = x_list[t]
           term = x_t.T @ np.kron(U3, U2) @ G1.T
           A_1_t = np.kron(term, np.eye(N))
           A_1_list.append(A_1_t)

        A_1 = np.vstack(A_1_list)
        vec_U1 = np.linalg.lstsq(A_1,y_stack,rcond=None)[0]
        U1 = vec_inv(vec_U1.flatten(),(N, ranks[0]))

        #minimization of U_2:
        A_2_list=[]
        for t in range(T_all):
           X_t = X_list[t]
           term_K = np.kron((X_t @ U3).T, np.eye(ranks[1]))
           A_2_t = U1 @ G1 @ term_K
           A_2_list.append(A_2_t)

        A_2 = np.vstack(A_2_list)
        vec_U2_T = np.linalg.lstsq(A_2,y_stack,rcond=None)[0]
        U2 = vec_inv(vec_U2_T.flatten(),(ranks[1], N)).T

        #minimization of U_3:
        A_3_list=[]
        for t in range(T_all):
           X_t = X_list[t]
           term_K = np.kron(np.eye(ranks[2]), U2.T @ X_t)
           A_3_t = U1 @ G1 @ term_K
           A_3_list.append(A_3_t)

        A_3 = np.vstack(A_3_list)
        vec_U3 = np.linalg.lstsq(A_3,y_stack,rcond=None)[0]
        U3 = vec_inv(vec_U3.flatten(),(P, ranks[2]))

        #minimization of the core
        G_list=[]
        for t in range(T_all):
           x_t = x_list[t]
           term_inner = np.kron(U3, U2).T @ x_t
           G_t = np.kron(term_inner.T, U1)
           G_list.append(G_t)

        G_stack=np.vstack(G_list)
        vec_G_1 = np.linalg.lstsq(G_stack,y_stack,rcond=None)[0] #the vec of the mode-1 unfolding of G

        G_1 = vec_inv(vec_G_1.flatten(), (ranks[0], ranks[1]*ranks[2]))
        G = G_1.reshape((ranks[0], ranks[1], ranks[2]), order='F')

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

  #last step is equivalent to do the hosvd
  G, factors = hosvd_decomposition(A_previous)
  U1 = factors[0]
  U2 = factors[1]
  U3 = factors[2]
  G = product_3(product_2(product_1(A_previous,U1.T),U2.T),U3.T)
  A_final = A_previous

  return A_final,G,U1,U2,U3