import numpy as np


def matrix_transpose(A: np.ndarray):
  """
  Return the transpose of matrix A. Do not use .T
  """
  assert len(A.shape) == 2

  rows, cols = A.shape
  A_t = np.zeros((cols, rows))
  for i in range(rows):
    A_t[:, i] = A[i]

  return A_t


matrix_transpose(np.random.rand(3, 4))
matrix_transpose(np.random.rand(1, 4))
matrix_transpose(np.random.rand(2, 2))
