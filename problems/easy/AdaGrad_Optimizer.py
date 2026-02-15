import numpy as np


def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
  """
  Perform one AdaGrad update step.
  Where: w = parameters, g = gradients, G = accumulated squared gradients, η = learning rate, ε = stability constant
  Return tuple (new_w, new_G) with same shapes as inputs
  """
  new_G = G + np.pow(g, 2)
  update = lr / np.sqrt(new_G + eps) * g
  return w - update, new_G
