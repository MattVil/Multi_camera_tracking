import os
import cv2
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from utils import constant
from data_loader import StreamLoader, VideoLoader 
from detection import load_model, run_inference
from utils.data_utils import calibration_to_array, split_image_divisor
from projection import transpose_on_playground, fuse_points

class Tracker(ABC):
  """"""

  def __init__(self, mode, data_sources_n):
    assert(mode in ['stream',  'video'])
    if mode == 'stream':
      self.data_loader = StreamLoader(data_sources_n)
    elif mode == 'video':
      self.data_loader = VideoLoader(data_sources_n)

  @abstractmethod
  def run(self):
    pass

  def __compute_homography(self, calibration_files):
    """"""
    M = []
    for file in calibration_files:
      src, dst = calibration_to_array(os.path.join("./utils/", file))
      M0, status = cv2.findHomography(src, dst, method=cv2.LMEDS)
      M.append(M0)
    return M


class SimpleTracker(Tracker):
  """"""

  def __init__(self, mode, data_sources):
    super().__init__(mode, data_sources)

  def run(self):
    """"""
    M = self.__compute_homography(constant.CALIBRATION_FILES)
    __is_running = True

    mog2s = [cv2.createBackgroundSubtractorMOG2() for i in range(3)]
    while(__is_running):

      frames, __is_running = self.data_loader.get_next_frames()

      for i, frame in enumerate(frames):
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # frame = cv2.GaussianBlur(frame, (5, 5), 0)
        # frame = cv2.GaussianBlur(frame, (5, 5), 0)


        mog2_mask = mog2s[i].apply(frame)
        ret, mask = cv2.threshold(mog2_mask, 245, 255, cv2.THRESH_BINARY)
        mask = cv2.erode(mask, np.ones((2,2), np.uint8))
        mask = cv2.erode(mask, np.ones((5,5), np.uint8))
        mask = cv2.erode(mask, np.ones((3,3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8)) 
        mask = cv2.erode(mask, np.ones((5,5), np.uint8)) 
        mask = cv2.erode(mask, np.ones((3,3), np.uint8)) 
        mask = cv2.dilate(mask, np.ones((5,5), np.uint8)) 
        # mask = cv2.dilate(mask, np.ones((10,10), np.uint8)) 
        with_mask = cv2.addWeighted(frame, 1, 
                                    cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB), 0.7, 0)
        cv2.imshow('mog2_{}'.format(i), cv2.resize(mask, (480, 360)))
        cv2.imshow('Origin_{}'.format(i),cv2.resize(frame, (480, 360)))
        cv2.imshow('masked_{}'.format(i),cv2.resize(with_mask, (480, 360)))
        cv2.moveWindow('Origin_{}'.format(i), i*480+100, 0)
        cv2.moveWindow('mog2_{}'.format(i), i*480+100, 470)
        cv2.moveWindow('masked_{}'.format(i), i*480+100, 0)
      k = cv2.waitKey(0)
      if k == 27:
          break

  def __compute_homography(self, calibration_files):
    """"""
    M = []
    for file in calibration_files:
      src, dst = calibration_to_array(os.path.join("./utils/", file))
      M0, status = cv2.findHomography(src, dst, method=cv2.LMEDS)
      M.append(M0)
    return M





class TFObjectDetectionAPITracker(Tracker):
  """"""

  def __init__(self, mode, data_sources):
    Tracker.__init__(self, mode, data_sources)

  def run(self):
    """"""
    M = self.__compute_homography(constant.CALIBRATION_FILES)
    graph, tensor_dict, image_tensor=self.__load_model(constant.PATH_TO_FROZEN_GRAPH)
    with graph.as_default():
      with tf.Session() as sess:
        nb_f = 0
        while(True):
          print(nb_f)
          nb_f = nb_f + 1
          frames, ret = self.data_loader.get_next_frames()

          cam_points = [[] for _ in frames]
          for i, frame in enumerate(frames):
            for sub_frame in split_image_divisor(frame, 0.01):
              # object detection
              points = run_inference(sub_frame['image'], graph, sess, tensor_dict, image_tensor)
              (dx, dy) = sub_frame['position']
              cam_points[i]=cam_points[i]+[[int(x+dx), int(y+dy)] for (x, y) in points]

          # compute players projection on the playground
          projection = transpose_on_playground(cam_points, M)
          projection = fuse_points(projection, 20)

  def __load_model(self, path_to_frozen_graph):
    """Load weigths from file and return computation graph, tensor_dict."""
    detection_graph = tf.Graph()
    with detection_graph.as_default():
      od_graph_def = tf.GraphDef()
      with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
      # Get handles to input and output tensors
      with tf.Session() as sess:
         ops = tf.get_default_graph().get_operations()
         all_tensor_names = {output.name for op in ops for output in op.outputs}
         tensor_dict = {}
         for key in ['num_detections', 'detection_boxes', 'detection_scores',
                     'detection_classes', 'detection_masks']:
           tensor_name = key + ':0'
           if tensor_name in all_tensor_names:
             tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                 tensor_name)
         image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

    return detection_graph, tensor_dict, image_tensor


