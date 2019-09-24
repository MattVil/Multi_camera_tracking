import os
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod

from utils.constant import
from detection import load_model
from utils.data_utils import calibration_to_array


class SimpleTracker(Tracker):
  """"""

  def __init__(self):
    super(AbstractOperation, self).__init__()

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

  def __get_frames(self, streams):
    """"""
    frames = []
    ret = True
    for stream in streams:
      ret0, frame = stream.read()
      frames.append(frame)
      ret = ret and ret0
    return frames, ret

  def __compute_homography(self, cam_names):
    """"""
    M = []
    for name in cam_names:
      src, dst = calibration_to_array(name)
      M0, status = cv2.findHomography(src, dst, method=cv2.LMEDS)
      M.append(M0)
    return M

  def run(self, streams):
    """"""
    graph, tensor_dict, image_tensor = load_model("../"+PATH_TO_FROZEN_GRAPH)
    with graph.as_default():
      with tf.Session() as sess:
        while(True):
          frames, ret = self.__get_frames(streams)

          cam_points = [[] for _ in frames]
          for i, frame in enumerate(frames):
            for sub_frame in split_image_divisor(frame, 0.01):
              # object detection
              points = run_inference(sub_frame['image'], graph, sess, tensor_dict, image_tensor)
              (dx, dy) = sub_frame['position']
              cam_points[i] = cam_points[i] + [[int(x+dx), int(y+dy)] for (x, y) in points]

          # compute players projection on the playground
          projection = transpose_on_playground(cam_points, [M0, M1, M2])
          projection = fuse_points(projection, 20)


class Tracker(ABC):
  """"""

  def __init__(self):
    pass

  @abstractmethod
  def run(self):
    pass
