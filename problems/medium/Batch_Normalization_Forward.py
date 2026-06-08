import numpy as np


def batch_norm_forward(x, gamma, beta, eps=1e-5):
  """
  Forward-only BatchNorm for (N,D) or (N,C,H,W).

  x: (N, D) or (N, C, H, W)
  gamma: (D,) or (C,)
  beta: (D,) or (C,)
  """
  x_arr = np.asarray(x, dtype=float)
  gamma_arr = np.asarray(gamma, dtype=float)
  beta_arr = np.asarray(beta, dtype=float)

  print(f"{x=}")
  if x_arr.ndim == 2:
    mu = np.mean(x_arr, axis=0, keepdims=True)
    print(mu)
    var = np.var(x_arr, axis=0, mean=mu, keepdims=True)
    print(var)
    x_hat = (x_arr - mu) / np.sqrt(var + eps)
    print(x_hat)
    print(x_hat.shape)
    return gamma_arr * x_hat + beta_arr
  elif x.ndim == 4:
    # ((N, C, H, W))
    # (Batch, Channels, Height, Width)
    # Batch Normalization is designed to normalize statistics per channel.
    mu = np.mean(x_arr, axis=(0, 2, 3), keepdims=True)
    print("mu:")
    print(mu)
    print(mu.shape)
    var = np.var(x_arr, axis=(0, 2, 3), mean=mu, keepdims=True)
    print("var:")
    print(var)
    print(var.shape)
    x_hat = (x_arr - mu) / np.sqrt(var + eps)
    print("x_hat:")
    print(x_hat)
    print(x_hat.shape)
    print("gamma:")
    gamma_arr = gamma_arr[np.newaxis, :, np.newaxis, np.newaxis]
    # gamma_arr = gamma_arr[None, :, None, None]
    # gamma_arr = gamma_arr.reshape(1, 2, 1, 1)
    # gamma_arr = gamma_arr.reshape(1, -1, 1, 1)
    # gamma_arr = np.expand_dims(gamma_arr, axis=(0, 2, 3))
    # gamma_arr = gamma_arr.view(1, -1, 1, 1) # torch
    # gamma_arr = gamma_arr.unsqueeze(0).unsqueeze(2).unsqueeze(3) # torch
    # gamma_arr = gamma_arr.unsqueeze(-1).unsqueeze(-1).unsqueeze(0) # torch
    print(gamma_arr)
    print(gamma_arr.shape)
    beta_arr = beta_arr[None, :, None, None]
    print(beta_arr)
    print(beta_arr.shape)
    ret = gamma_arr * x_hat + beta_arr
    print("ret:")
    print(ret)
    print(ret.shape)
    return ret
  else:
    raise ValueError("Only 2 and 4 dims implemented")


batch_norm_forward([[1, 2], [3, 6], [5, 10]], [1, 0.5], [0, 1])
batch_norm_forward([[[[1]], [[2]]], [[[3]], [[4]]]], [1, 0.5], [0, -1])
