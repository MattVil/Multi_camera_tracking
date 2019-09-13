import cv2
import numpy as np

def custom_perspectiveTransform(src, dst):
  """Custom implementation of cv2.getPespectiveTransform

  args:
    src: np.array of shape (n, 2)
    dst: np.array of shape (n, 2)
  """
  pass


def transpose_on_playground(players, perspective_Ms):
  """Transpose player position in camera space to playground space.

  args:
    players: array of array of player position. players[0] : array of players
             positions in camera 0.
    perspective_Ms : list of homography matrix, one or each camera.
  """
  projection = []
  for i, M in enumerate(perspective_Ms):
    for player in players[i]:
      cam_pos = player.copy()
      cam_pos.append(1)
      cam_pos = np.float32(np.asarray(cam_pos))
      dot_product = M.dot(cam_pos.T)
      dot_product = dot_product.T
      x = int(dot_product[0]/dot_product[2])
      y = int(dot_product[1]/dot_product[2])
      projection.append((x, y))
  return projection
