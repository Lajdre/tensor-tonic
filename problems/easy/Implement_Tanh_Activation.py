import numpy as np


def tanh(x):
  """
  Implement Tanh activation function.
  """
  x = np.asarray(x, dtype=float)
  # return np.tanh(x)
  x_exp = np.exp(x)
  negative_x_exp = np.exp(-x)
  return (x_exp - negative_x_exp) / (x_exp + negative_x_exp)
