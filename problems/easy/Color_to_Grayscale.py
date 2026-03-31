def color_to_grayscale(image):
  """
  Convert an RGB image to grayscale using luminance weights.
  """
  r_weight = 0.299
  g_weight = 0.587
  b_weight = 0.114
  for i, row in enumerate(image):
    for j, cell in enumerate(row):
      cell_val = r_weight * cell[0] + g_weight * cell[1] + b_weight * cell[2]
      image[i][j] = cell_val

  return image


color_to_grayscale([[[255, 0, 0]]])
