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
      # print(cam_pos)
      cam_pos = np.float32(np.asarray(cam_pos))
      dot_product = M.dot(cam_pos.T)
      dot_product = dot_product.T
      x = int(dot_product[0]/dot_product[2])
      y = int(dot_product[1]/dot_product[2])
      projection.append((x, y))
  return projection

def distance(p1, p2):
  """"""
  return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def fuse_points(points, d):
  """"""
  ret = []
  d2 = d * d
  n = len(points)
  taken = [False] * n
  for i in range(n):
    if not taken[i]:
      count = 1
      point = [points[i][0], points[i][1]]
      taken[i] = True
      for j in range(i+1, n):
        if distance(points[i], points[j]) < d2:
          print(distance(points[i], points[j]))
          point[0] += points[j][0]
          point[1] += points[j][1]
          count+=1
          taken[j] = True
      point[0] /= count
      point[1] /= count
      ret.append((int(point[0]), int(point[1])))
  return ret
