import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(
  Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
  """
  Compute scaled dot-product attention.
  """
  d_k = Q.size(-1)

  matches = Q @ K.permute((0, 2, 1)) / math.sqrt(d_k)  # K.transpose(-2, -1))

  if mask is not None:
    matches = matches.masked_fill(mask == 0, float("-inf"))

  softmax_matches = F.softmax(matches, dim=-1)

  return softmax_matches @ V


def test_basic_shape():
  Q = torch.randn(1, 4, 64)
  K = torch.randn(1, 4, 64)
  V = torch.randn(1, 4, 64)

  output = scaled_dot_product_attention(Q, K, V)

  assert output.shape == (1, 4, 64)


def test_batch_shape():
  Q = torch.randn(2, 8, 32)
  K = torch.randn(2, 8, 32)
  V = torch.randn(2, 8, 32)

  output = scaled_dot_product_attention(Q, K, V)

  assert output.shape == (2, 8, 32)


def test_different_query_key_lengths():
  # makes sense for non-standard self-attention (e.g., cross-attention)
  Q = torch.randn(1, 3, 64)
  K = torch.randn(1, 5, 64)
  V = torch.randn(1, 5, 64)

  output = scaled_dot_product_attention(Q, K, V)

  assert output.shape == (1, 3, 64)


def test_masking():
  Q = torch.randn(1, 2, 4)
  K = torch.randn(1, 3, 4)
  V = torch.randn(1, 3, 6)

  # Mask out last key
  mask = torch.tensor([[[1, 1, 0], [1, 1, 0]]])  # shape (1,2,3)

  output = scaled_dot_product_attention(Q, K, V, mask=mask)

  assert output.shape == (1, 2, 6)


def test_known_small_example():
  """
  Deterministic test with small tensors where
  result can be computed manually.
  """
  Q = torch.tensor([[[1.0, 0.0]]])  # (1,1,2)
  K = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1,2,2)
  V = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])  # (1,2,2)

  output = scaled_dot_product_attention(Q, K, V)

  # Manually compute expected
  d_k = 2
  scores = torch.tensor([[1.0, 0.0]]) / math.sqrt(d_k)
  weights = torch.softmax(scores, dim=-1)
  expected = weights @ V[0]

  assert torch.allclose(output[0], expected, atol=1e-5)


def test_numerical_stability_large_values():
  Q = torch.randn(1, 4, 64) * 1000
  K = torch.randn(1, 4, 64) * 1000
  V = torch.randn(1, 4, 64)

  output = scaled_dot_product_attention(Q, K, V)

  assert not torch.isnan(output).any()
  assert not torch.isinf(output).any()


def test_gradient_flow():
  Q = torch.randn(1, 3, 8, requires_grad=True)
  K = torch.randn(1, 3, 8, requires_grad=True)
  V = torch.randn(1, 3, 8, requires_grad=True)

  output = scaled_dot_product_attention(Q, K, V)

  loss = output.sum()
  loss.backward()

  assert Q.grad is not None
  assert K.grad is not None
  assert V.grad is not None


# test_basic_shape()
# test_batch_shape()
# test_different_query_key_lengths()
test_masking()
# test_known_small_example()
# test_numerical_stability_large_values()
# test_gradient_flow()
