import cv2
import math
import numpy as np

def split_image(image, nb_split=4, overlap=0.0):
  """Split a image into nb_split sub-images
  /!\ overlap not working properly
  args:
    image: source image to be splited
    nb_split: number of sub-images created. Must be a square number
    overlap: overlap size between sub-images
  """
  height, width = image.shape[:2]
  overlap_h, overlap_w = overlap*height, overlap*width
  new_h = int(height/math.sqrt(nb_split) + overlap_h)
  new_w = int(width/math.sqrt(nb_split) + overlap_w)

  subImages = []
  for i in range(int(math.sqrt(nb_split))):
    for j in range(int(math.sqrt(nb_split))):
      x = int(i * (new_w - 1/2*overlap_w))
      y = int(j * (new_h - 1/2*overlap_h))
      subImages.append(image[y:y+new_h, x:x+new_w])
  return subImages


def split_image_divisor(image, scale):
  """"""
  height, width = image.shape[:2]
  max_size = max(height, width)
  divisor = []
  for i in range(2, max_size):
    if(height%i == 0 and width%i == 0):
      divisor.append(i)

  divisor = divisor[::-1]
  idx = int(scale*len(divisor))
  box_size = divisor[idx]
  nb_x = int(width/box_size)
  nb_y = int(height/box_size)
  subImages = []
  for i in range(nb_x):
    for j in range(nb_y):
      x = int(i*box_size)
      y = int(j*box_size)
      subImages.append({'image':image[y:y+box_size, x:x+box_size],
                        'position':(x, y)})
  return subImages

def calibration_to_array(calibration_path):
  """Load the calibration file and return calibration between camera space
  and playground"""
  calibration = np.load(calibration_path)
  cam, playground = [], []
  for points in calibration:
    cam.append(np.asarray(points[0]))
    playground.append(np.asarray(points[1]))
  return np.float32(np.asarray(cam)), np.float32(np.asarray(playground))

def merge_points(points, merge_dst=5):
  """"""
  to_be_merge = []
  for i, p1 in enumerate(points):
    for j, p2 in enumerate(points):
      if i < j:
        break
      if(math.sqrt((p1[0]+p2[0])**2 + (p1[1]+p2[1])**2) < merge_dst):
        to_be_merge.append((i, j))
