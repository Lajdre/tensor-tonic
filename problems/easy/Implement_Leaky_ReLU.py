import numpy as np


def leaky_relu(x, alpha=0.01):
  """Vectorized Leaky ReLU implementation."""
  x_arr = np.asarray(x, dtype=float)
  return np.where(x_arr >= 0, x_arr, alpha * x_arr)
