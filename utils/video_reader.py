import os
import cv2
import numpy as np

DATASET_PATH = '/workspace/Dataset/soccer'
VIDEO_NAME = "0056_2013-11-03 18:01:14.248366000.h264"

def display(frame0, frame1, frame2, ret):
  """"""
  if not ret:
    return None
  height, width = frame0.shape[:2]
  scale = 1/2.2

  cv2.imshow("Cam0", cv2.resize(frame0, (int(scale*width), int(scale*height))))
  cv2.imshow("Cam1", cv2.resize(frame1, (int(scale*width), int(scale*height))))
  cv2.imshow("Cam2", cv2.resize(frame2, (int(scale*width), int(scale*height))))

  cv2.moveWindow('Cam0', 0, 0)
  cv2.moveWindow('Cam1', 700, 0)
  cv2.moveWindow('Cam2', 1300, 0)

def main():
    capCam0 = cv2.VideoCapture(os.path.join(DATASET_PATH, '0/', VIDEO_NAME))
    capCam1 = cv2.VideoCapture(os.path.join(DATASET_PATH, '1/', VIDEO_NAME))
    capCam2 = cv2.VideoCapture(os.path.join(DATASET_PATH, '2/', VIDEO_NAME))
    ret = True

    while(capCam0.isOpened() and capCam1.isOpened() and capCam2.isOpened() and ret):
      ret0, frame0 = capCam0.read()
      ret1, frame1 = capCam1.read()
      ret2, frame2 = capCam2.read()
      ret = ret0 and ret1 and ret2

      display(frame0, frame1, frame2, ret)

      k = cv2.waitKey(0)
      if(k == 27):
        break

    capCam0.release()
    capCam1.release()
    capCam2.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
