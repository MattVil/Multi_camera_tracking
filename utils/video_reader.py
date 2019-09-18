import sys
sys.path.append('../')
import os
import cv2
import math
import numpy as np
import tensorflow as tf

from data_utils import split_image, split_image_divisor, calibration_to_array
from detection import simulate_detection, load_model, PATH_TO_FROZEN_GRAPH
from detection import run_inference
from projection import transpose_on_playground, fuse_points

DATASET_PATH = '/workspace/Dataset/soccer'
VIDEO_NAME = "0056_2013-11-03 18:01:14.248366000.h264"

def display(playground, frame0, frame1, frame2, ret=True, cam_pts=[],
            playground_pts=[], split=True):
  """"""

  if not ret:
    return None
  height, width = frame0.shape[:2]
  scale = 1/2.2

  frame0_d = frame0.copy()
  frame1_d = frame1.copy()
  frame2_d = frame2.copy()
  playground_d = playground.copy()

  if cam_pts:
    h = 0
    for i, frame_d in enumerate([frame0_d, frame1_d, frame2_d]):
      for j, player in enumerate(cam_pts[i]):
        cv2.circle(frame_d, tuple(player), 10, (0, 0, 200), -1)
        cv2.putText(frame_d,'{}'.format(h),
            (player[0]+10, player[1]+10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,255,255),
            2)
        h = h+1

  if playground_pts:
    for i, player_proj in enumerate(playground_pts):
      cv2.circle(playground_d, player_proj, 5, (0, 0, 200), -1)
      cv2.putText(playground_d,'{}'.format(i),
          (player_proj[0]+10, player_proj[1]+10),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.4,
          (0,0,0),
          1)

  frame0_d = cv2.resize(frame0_d, (int(scale*width), int(scale*height)))
  frame1_d = cv2.resize(frame1_d, (int(scale*width), int(scale*height)))
  frame2_d = cv2.resize(frame2_d, (int(scale*width), int(scale*height)))

  cv2.imshow("Playground", playground_d)
  cv2.imshow("Cam0", frame0_d)
  cv2.imshow("Cam1", frame1_d)
  cv2.imshow("Cam2", frame2_d)

  cv2.moveWindow("Playground", 700, 500)
  cv2.moveWindow('Cam0', 0, 0)
  cv2.moveWindow('Cam1', 700, 0)
  cv2.moveWindow('Cam2', 1300, 0)

  if split:
    for i, frame in enumerate([frame0, frame1, frame2]):
      nb_split = 4
      roi = split_image_divisor(frame, 0.01)
      for j, sub_img in enumerate(roi):
        show_s = (int(1/math.sqrt(nb_spldst0it)*width*scale),
                  int(1/math.sqrt(nb_split)*height*scale))
        cv2.imshow("Frame{}.{}".format(i, j), cv2.resize(sub_img, show_s))
        cv2.moveWindow("Frame{}.{}".format(i, j), i*700+j*15, 500+j*15)

def main():
  playground = cv2.imread(os.path.join(DATASET_PATH, 'playground.jpg'))
  capCam0 = cv2.VideoCapture(os.path.join(DATASET_PATH, '0/', VIDEO_NAME))
  capCam1 = cv2.VideoCapture(os.path.join(DATASET_PATH, '1/', VIDEO_NAME))
  capCam2 = cv2.VideoCapture(os.path.join(DATASET_PATH, '2/', VIDEO_NAME))
  ret = True
  cam0, cam1, cam2 = [], [], []

  # Load the video in memory
  while(capCam0.isOpened() and capCam1.isOpened() and capCam2.isOpened() and ret):
    ret0, frame0 = capCam0.read()
    ret1, frame1 = capCam1.read()
    ret2, frame2 = capCam2.read()
    ret = ret0 and ret1 and ret2
    if ret:
      cam0.append(frame0)
      cam1.append(frame1)
      cam2.append(frame2)

  # Setup projection calibration
  src0, dst0 = calibration_to_array("calibration_cam0.npy")
  src1, dst1 = calibration_to_array("calibration_cam1.npy")
  src2, dst2 = calibration_to_array("calibration_cam2.npy")
  M0, status = cv2.findHomography(src0, dst0, method=cv2.LMEDS)
  M1, status = cv2.findHomography(src1, dst1, method=cv2.LMEDS)
  M2, status = cv2.findHomography(src2, dst2, method=cv2.LMEDS)

  i=80
  nb_images = min(len(cam0), len(cam1), len(cam2))-1

  graph = load_model("../"+PATH_TO_FROZEN_GRAPH)
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)

      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')


      while(True):
        frame0 = cam0[i]
        frame1 = cam1[i]
        frame2 = cam2[i]

        # Simulate player position for projection
        cam_points = []
        for i, frame in enumerate([frame0, frame1, frame2]):
          cam_points.append([])
          for sub_frame in split_image_divisor(frame, 0.01):
            points = run_inference(sub_frame['image'], graph, sess, tensor_dict, image_tensor)
            (dx, dy) = sub_frame['position']
            cam_points[i] = cam_points[i] + [[int(x+dx), int(y+dy)] for (x, y) in points]

        # compute players projection on the playground
        projection = transpose_on_playground(cam_points, [M0, M1, M2])
        projection = fuse_points(projection, 20)

        display(playground, frame0, frame1, frame2, cam_pts=cam_points,
                playground_pts=projection, split=False)

        k = cv2.waitKey(0)
        if(k == 100):
          i = i+5 if i < nb_images else i
        elif(k == 113):
          i = i-5 if i > 0 else i
        elif(k == 27):
          break

  capCam0.release()
  capCam1.release()
  capCam2.release()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()
