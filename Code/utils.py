import numpy as np
import matplotlib.pyplot as plt
import tensorly as tl
from tensorly.decomposition import tucker

def product_1(A,B):
  """
  Return the 1 product between a tensor and a matrix
  """
  return np.einsum('ijk,si->sjk', A, B)

def product_2(A,B):
  """
  Return the 2 product between a tensor and a matrix
  """
  return np.einsum('ijk,sj->isk', A, B)

def product_3(A,B):
  """
  Return the 3 product between a tensor and a matrix
  """
  return np.einsum('ijk,sk->ijs', A, B)

def norm0(A):
  """
  Return the 0 "norm" of a matrix or a tensor
  """
  return np.count_nonzero(A)

def normF(A):
  """
  Return the Frobenius norm of a matrix or a tensor
  """
  return np.linalg.norm(A)

def norm1(A):
  """
  Return the 1 norm of a matrix or a tensor
  """
  return np.linalg.norm(A,ord=1)

def tensor_mode(X, mode_index):
    """
    tensor unfolding for 3d tensor

    Args:
        X (np.ndarray): 3d tensor(Shape: I, J, K).
        mode_index (int): the mod of unfolding.

    Returns:
        np.ndarray: unfolded matrix
    """

    # Dimensions I, J, K
    I, J, K = X.shape

    if mode_index == 1:
        # X_(1): Permutation (i, j, k) -> (0, 1, 2). columns = j, k
        permutation = (0, 2, 1)
        target_shape = (I, J * K)

    elif mode_index == 2:
        # X_(2): Permutation (j, k, i) -> (1, 2, 0). columns= k, i
        permutation = (1, 2, 0)
        target_shape = (J, K * I)

    elif mode_index == 3:
        # X_(3): Permutation (k, i, j) -> (2, 0, 1). columns = i, j
        permutation = (2, 1,0)
        target_shape = (K, I * J)

    X_permuted = X.transpose(permutation)
    X_unfolding = X_permuted.reshape(target_shape)

    return X_unfolding

def rank_1(X):
  return np.linalg.matrix_rank(tensor_mode(X,1))

def rank_2(X):
  return np.linalg.matrix_rank(tensor_mode(X,2))

def rank_3(X):
  return np.linalg.matrix_rank(tensor_mode(X,3))

def reconstruct_tensor(G, U1, U2, U3):
    """ 
    Reconstruct a tensor from its Tucker decomposition.
    Args:
        G (np.ndarray): Core tensor.
        U1 (np.ndarray): Factor matrix for mode 1.
        U2 (np.ndarray): Factor matrix for mode 2.
        U3 (np.ndarray): Factor matrix for mode 3.
    Returns:
        np.ndarray: Reconstructed tensor.
    """
    A = product_3(product_2(product_1(G,U1),U2),U3)
    return A

def hosvd_decomposition(X, ranks=None):
    """
    hosvd tucker decomposition of a tensor

    Args:
        X (np.ndarray): 3d tensor(shape: I, J, K).

    Returns:
        core (np.ndarray): core matrix (shape: r1,r2,r3)
        factors (3np.array): factors matrix (shape: r1*I,r2*J,r3*K)
    """
    if ranks==None:
       ranks = [rank_1(X), rank_2(X), rank_3(X)]
    tl.set_backend('numpy')
    core, factors = tucker(X, rank=ranks, init='svd', n_iter_max=0) #hosvd decomposition
    return core, factors
def vec(A):
  """
  Implement the vectorization of a matrix by stacking its columns.
  """
  return A.flatten(order='F')
def vec_inv(u,shape):
  """
  Implement the inverse vectorization of a vector
  """
  return u.reshape(shape, order='F')

def predict_var_step(A_tensor, history_window):

    N, _, P = A_tensor.shape
    y_pred = np.zeros(N)


    for k in range(P):
        lag_val = history_window[:, -(k+1)]
        y_pred += A_tensor[:, :, k] @ lag_val

    return y_pred

# Test code
if __name__ == "__main__":
    X = np.arange(1, 9).reshape(2,2,2)
    X=np.random.randn(4, 4, 4)

    core, factors= hosvd_decomposition(X)
    print('should be Id :')
    print()
    print(factors[0].T@factors[0]) #should be Id
    print()
    print('should be tensor 0 :')
    print()
    print(product_3(product_2(product_1(core,factors[0]),factors[1]),factors[2])-X) #should be X, so we compare with it.
    print()
    print('should be tensor 0 :')
    print()
    print(product_3(product_2(product_1(X,factors[0].T),factors[1].T),factors[2].T)-core)
    print('-----------------')
    A=[[1,2],[3,4],[5,6],[7,8]]
    A=np.array(A)
    print(A)
    print(vec(A))
    print(A.flatten(order='F'))
    print(vec(A).reshape(A.shape, order='F'))
    print(vec_inv(vec(A),A.shape))

