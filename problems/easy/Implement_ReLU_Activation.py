import numpy as np


def relu(x):
  """Implement ReLU activation function."""
  x_arr = np.asarray(x, dtype=np.float64)
  # return np.maximum(0.0, x_arr)
  return np.where(x_arr > 0, x_arr, 0)
