import os
import cv2
import math
import numpy as np

from data_utils import split_image

DATASET_PATH = '/workspace/Dataset/soccer'
VIDEO_NAME = "0056_2013-11-03 18:01:14.248366000.h264"

def display(playground, frame0, frame1, frame2, ret, split=True):
  """"""
  if not ret:
    return None
  height, width = frame0.shape[:2]
  scale = 1/2.2

  cv2.imshow("Playground", playground)
  cv2.imshow("Cam0", cv2.resize(frame0, (int(scale*width), int(scale*height))))
  cv2.imshow("Cam1", cv2.resize(frame1, (int(scale*width), int(scale*height))))
  cv2.imshow("Cam2", cv2.resize(frame2, (int(scale*width), int(scale*height))))

  cv2.moveWindow("Playground", 700, 500)
  cv2.moveWindow('Cam0', 0, 0)
  cv2.moveWindow('Cam1', 700, 0)
  cv2.moveWindow('Cam2', 1300, 0)

  if split:
    for i, frame in enumerate([frame0, frame1, frame2]):
      nb_split = 4
      roi = split_image(frame, nb_split=nb_split)
      for j, sub_img in enumerate(roi):
        show_s = (int(1/math.sqrt(nb_split)*width*scale),
                  int(1/math.sqrt(nb_split)*height*scale))
        cv2.imshow("Frame{}.{}".format(i, j), cv2.resize(sub_img, show_s))
        cv2.moveWindow("Frame{}.{}".format(i, j), i*700+j*15, 500+j*15)

def main():
    playground = cv2.imread(os.path.join(DATASET_PATH, 'playground.jpg'))
    capCam0 = cv2.VideoCapture(os.path.join(DATASET_PATH, '0/', VIDEO_NAME))
    capCam1 = cv2.VideoCapture(os.path.join(DATASET_PATH, '1/', VIDEO_NAME))
    capCam2 = cv2.VideoCapture(os.path.join(DATASET_PATH, '2/', VIDEO_NAME))
    ret = True

    while(capCam0.isOpened() and capCam1.isOpened() and capCam2.isOpened() and ret):
      ret0, frame0 = capCam0.read()
      ret1, frame1 = capCam1.read()
      ret2, frame2 = capCam2.read()
      ret = ret0 and ret1 and ret2

      display(playground, frame0, frame1, frame2, ret, split=False)

      k = cv2.waitKey(0)
      if(k == 27):
        break

    capCam0.release()
    capCam1.release()
    capCam2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
