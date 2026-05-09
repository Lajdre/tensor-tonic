import numpy as np
import numpy.typing as npt


def covariance_matrix(X, verbose=False) -> npt.NDArray[np.float64] | None:
  """Compute covariance matrix from dataset X."""
  x = np.asarray(X, dtype=float)

  x_shape = x.shape
  n = x_shape[0]
  if len(x_shape) < 2 or n < 2:
    return None

  # Center the data
  mu = np.mean(x, axis=0)
  x_centered = x - mu

  if verbose:
    print(x_shape)
    print(x)
    print(mu.shape)
    print(mu)
    print(x_centered)

  # Comupte Covariance Matrix
  n = x.shape[0]
  return (x_centered.T @ x_centered) / (n - 1)


expected = np.array([[1.0, 1.0], [1.0, 1.0]])
result = covariance_matrix([[1, 2], [2, 3], [3, 4]])
assert result is not None
assert np.allclose(result, expected)

expected = np.array([[0.5, -0.5], [-0.5, 0.5]])
result = covariance_matrix([[1, 0], [0, 1]])
assert result is not None
assert np.allclose(result, expected)

result = covariance_matrix(X=[[1, 2, 3]])
assert result is None
