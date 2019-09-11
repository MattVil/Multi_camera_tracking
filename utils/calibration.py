import os
import cv2
from video_reader import DATASET_PATH, VIDEO_NAME

PLAYGROUND_PTS = [(480, 86), (480, 401), (411, 318), (411, 169), (252, 290),
                  (252, 198), (252, 401), (252, 86), (93, 318), (93, 169),
                  (25, 401), (25, 86)]
CURRENT_PT = (0.0)
CALIBRATION_PTS = []

def draw_circle(event, x, y, flags, param):
  """"""
  global mouseX, mouseY
  if event == cv2.EVENT_LBUTTONDOWN:
    CALIBRATION_PTS.append(((x, y), CURRENT_PT))
    mouseX, mouseY = x, y

def main():
  """"""
  playground_org = cv2.imread(os.path.join(DATASET_PATH, 'playground.jpg'))
  for i in range(3):
    cap = cv2.VideoCapture(os.path.join(DATASET_PATH, str(i)+'/', VIDEO_NAME))
    if not cap.isOpened():
      print("Error reading video stream.")
      exit()


    ret, frame = cap.read()

    for point in PLAYGROUND_PTS:
      global CURRENT_PT
      CURRENT_PT = point
      playground = playground_org.copy()
      cv2.circle(playground, point, 3, (255, 0, 0), -1)
      cv2.imshow("Playground", playground)
      cv2.moveWindow("Playground", 1400, 0)

      cv2.namedWindow('Camera')
      cv2.setMouseCallback('Camera', draw_circle)
      cv2.imshow('Camera', frame)

      k = cv2.waitKey(0)
      if(k == 27):
        break

    print("#"*80)
    print("Calibration for Cam{}".format(i))
    points = CALIBRATION_PTS
    print(points)
    CALIBRATION_PTS = []

  cap.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
