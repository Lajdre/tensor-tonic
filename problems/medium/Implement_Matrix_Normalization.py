import numpy as np


def matrix_normalization(matrix, axis=None, norm_type="l2"):
  """Normalize a 2D matrix along specified axis using specified norm."""
  m = np.asarray(matrix, dtype=float)
  if norm_type not in ("l1", "l2", "max"):
    return None
  if m.ndim != 2:
    return None
  if axis is not None:
    if axis >= m.ndim or axis < -m.ndim:
      return None
  print(m)
  print(m.shape)
  m_normalied = m
  if norm_type == "l2":
    l2_norm = np.linalg.norm(m, axis=axis, keepdims=bool(axis))
    safe_norms = np.where(l2_norm == 0, 1, l2_norm)
    print(l2_norm)
    print(l2_norm.shape)
    m_normalied = m / safe_norms
  elif norm_type == "l1":
    l1_norm = np.linalg.norm(m, axis=axis, ord=1, keepdims=bool(axis))
    safe_norms = np.where(l1_norm == 0, 1, l1_norm)
    print(l1_norm)
    print(l1_norm.shape)
    m_normalied = m / safe_norms
  elif norm_type == "max":
    max_norm = np.linalg.norm(m, axis=axis, ord=np.inf, keepdims=bool(axis))
    safe_norms = np.where(max_norm == 0, 1, max_norm)
    print(max_norm)
    print(max_norm.shape)
    m_normalied = m / safe_norms

  print(m_normalied)
  print(m_normalied.shape)
  return m_normalied


matrix_normalization([[3, 4], [1, 0]], axis=1, norm_type="l1")
# [ [ 0.6, 0.8 ], [ 1, 0 ] ]
matrix_normalization([[1, 2], [3, 4]], axis=0, norm_type="l1")
# [ [ 0.25, 0.3333333333333333 ], [ 0.75, 0.6666666666666666 ] ]
matrix_normalization([[2, 8, 4], [1, 3, 9]], axis=1, norm_type="max")
# [[0.25, 1, 0.5], [0.1111111111111111, 0.3333333333333333, 1]]
