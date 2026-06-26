import math

import numpy as np
import torch


def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
  """
  Generate sinusoidal positional encodings.
  """
  pos = np.arange(seq_length).reshape(-1, 1)  # (seq_length, 1)
  even_dims = np.arange(0, d_model, 2)
  odd_dims = np.arange(1, d_model, 2)
  # if d_model is even (generally always) this can be removed. Multi-head
  # attention requires d_model mod num_heads = 0 too

  # 1 / 10000^(2i/d_model) = exp(2i * −log(10000)/d_model)
  division_term_even = np.exp(even_dims * -np.log(10000) / d_model)
  division_term_odd = np.exp(odd_dims * -np.log(10000) / d_model)

  template = np.ones((seq_length, d_model))

  template[:, even_dims] *= np.sin(pos * division_term_even)
  template[:, odd_dims] *= np.cos(pos * division_term_odd)

  return template


# Assumes even d_model
def positional_encoding_pytorch(
  seq_length: int,
  d_model: int,
  device: torch.device | None = None,
  dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
  pos = torch.arange(seq_length, device=device, dtype=dtype).unsqueeze(1)

  even_dims = torch.arange(0, d_model, 2, device=device, dtype=dtype)

  div_term = torch.exp(-math.log(10000.0) * even_dims / d_model)

  pe = torch.zeros((seq_length, d_model), device=device, dtype=dtype)

  pe[:, 0::2] = torch.sin(pos * div_term)
  pe[:, 1::2] = torch.cos(pos * div_term)

  return pe
