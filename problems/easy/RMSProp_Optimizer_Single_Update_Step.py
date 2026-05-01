import numpy as np
import numpy.typing as npt


def rmsprop_step(
  w: npt.ArrayLike, g: npt.ArrayLike, s: npt.ArrayLike, lr=0.001, beta=0.9, eps=1e-8
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
  """
  Perform one RMSProp update step.
  w = parameters, g = gradients, s = squared gradient accumulator
  Return tuple (new_w, new_s) with same shapes as inputs
  """
  w_arr = np.asarray(w, dtype=np.float64)
  s_arr = np.asarray(s, dtype=np.float64)
  g_arr = np.asarray(g, dtype=np.float64)
  # Update running average
  s_t = beta * s_arr + (1 - beta) * g_arr**2
  # Parameter update
  w_t = w_arr - lr / np.sqrt(s_t + eps) * g_arr

  return (w_t, s_t)
