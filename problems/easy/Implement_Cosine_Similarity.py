import numpy as np


def cosine_similarity(a, b):
  """
  Compute cosine similarity between two 1D NumPy arrays.
  Returns: float in [-1, 1]
  """
  a = np.asarray(a, dtype=float)
  b = np.asarray(b, dtype=float)

  if a.shape != b.shape:
    raise ValueError("Vectors must have the same shape")

  # ec_norm_a = np.sqrt(np.sum(np.power(a, 2)))
  # ec_norm_b = np.sqrt(np.sum(np.power(b, 2)))
  ec_norm_a = np.linalg.norm(a)
  ec_norm_b = np.linalg.norm(b)
  if ec_norm_a == 0 or ec_norm_b == 0:
    return 0.0
  return np.dot(a, b) / (ec_norm_a * ec_norm_b)
