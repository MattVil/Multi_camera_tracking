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

def calibration_to_array(calibration_path):
  """Load the calibration file and return calibration between camera space
  and playground"""
  calibration = np.load(calibration_path)
  cam, playground = [], []
  for points in calibration:
    cam.append(points[0])
    playground.append(points[1])
  return cam, playground
