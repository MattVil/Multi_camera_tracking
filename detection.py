import cv2
import random
import numpy as np

def simulate_detection(image, max_player=6):
  """Simultate the detection of players in an image."""
  height, width = image.shape[:2]
  nb_player = random.randint(1, max_player)
  positions = []
  for player in range(nb_player):
    x = random.randint(0, width)
    y = random.randint(0, height)
    positions.append([x, y])
  return positions
