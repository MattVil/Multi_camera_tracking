import os
import cv2
import random
import numpy as np
import tensorflow as tf

from utils.data_utils import split_image, split_image_divisor

MODEL_NAME = "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
PATH_TO_FROZEN_GRAPH = "model/" + MODEL_NAME + "/frozen_inference_graph.pb"
PATH_TO_LABELS = "model/" + MODEL_NAME + "/mscoco_label_map.pbtxt"

DATASET_PATH = '/workspace/Dataset/soccer'
VIDEO_NAME = "0056_2013-11-03 18:01:14.248366000.h264"

def simulate_detection(image, max_player=6):
  """Simultate the detection of players frozen_inference_graph.pbtxtin an image."""
  height, width = image.shape[:2]
  nb_player = random.randint(1, max_player)
  positions = []
  for player in range(nb_player):
    x = random.randint(0, width)
    y = random.randint(0, height)
    positions.append([x, y])
  return positions

def load_model(path_to_frozen_graph):
  """Load weigths from file and return computation graph."""
  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(path_to_frozen_graph, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return detection_graph

def load_model_map(path):
  """"""
  return label_map_util.create_category_index_from_labelmap(
    PATH_TO_LABELS,
    use_display_name=True)

def load_image_into_numpy_array(image):
  """"""
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

def bbox_to_point(bbox, img_size, proj_type='footprint'):
  """"""
  x, y = 0, 0
  x = int((bbox[0] * img_size + bbox[2] * img_size)/2)
  if(proj_type == 'footprint'):
    y = int(bbox[3] * img_size)
  else: # 'centroid'
    y = int((bbox[1] * img_size + bbox[3] * img_size)/2)
  return [x, y]

def iou(bbox1, bbox2):
  """return the IoU (Intersection over Union) of 2 bboxes"""
  # determine the coordinates of the intersection rectangle
  x_left = max(bbox1[0], bbox2[0])
  y_top = max(bbox1[1], bbox2[1])
  x_right = min(bbox1[2], bbox2[2])
  y_bottom = min(bbox1[3], bbox2[3])

  if x_right < x_left or y_bottom < y_top:
    return 0.0

  intersection_area = (x_right - x_left) * (y_bottom - y_top)

  bb1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
  bb2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

  iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
  return iou

def merge_bboxes(bboxes, iou_thresh=0.05):
  """"""
  no_more_merge = False
  while(not no_more_merge):
    new_bboxes, rm_bboxes = [], []
    no_more_merge = True
    for i, bbox1 in enumerate(bboxes):
      for j, bbox2 in enumerate(bboxes):
        if(i < j):
          # if bbox2 in bbox1 then keep bbox1
          if(bbox1[0]<=bbox2[0] and bbox1[1]<=bbox2[1] and 
             bbox1[2]>=bbox2[2] and bbox1[3]>=bbox2[3]):
            rm_bboxes.append(bbox2)
            no_more_merge = False
          # if bbox1 in bbox2 then keep bbox2
          elif(bbox2[0]<=bbox1[0] and bbox2[1]<=bbox1[1] and 
               bbox2[2]>=bbox1[2] and bbox2[3]>=bbox1[3]):
            rm_bboxes.append(bbox1)
            no_more_merge = False
          # if overlapping bboxes then merges
          elif(iou(bbox1, bbox2) > iou_thresh):
            new_bboxes.append((min(bbox1[0], bbox2[0]),
                               min(bbox1[1], bbox2[1]),
                               max(bbox1[2], bbox2[2]),
                               max(bbox1[3], bbox2[3])))
            rm_bboxes.append(bbox1)
            rm_bboxes.append(bbox2)
            no_more_merge = False
    rm_bboxes = [t for t in (set(tuple(i) for i in rm_bboxes))]
    for rm_bbox in rm_bboxes:
      bboxes.remove(rm_bbox)
    bboxes = bboxes + new_bboxes
    bboxes = [t for t in (set(tuple(i) for i in bboxes))]
    print("\t"+str(bboxes))
  return bboxes

def run_inference(image, graph, session, tensor_dict, image_tensor, confidence=0.3):
  """"""
  img_org_size = image.shape[0]
  resize_ratio = img_org_size/640
  img_rsd = cv2.resize(image, (640, 640))
  img_rsd = np.expand_dims(img_rsd, axis=0)

  # with graph.as_default():
  #   with session as sess:
  output_dict = session.run(tensor_dict, feed_dict={image_tensor: img_rsd})
  # all outputs are float32 numpy arrays, so convert types as appropriate
  output_dict['num_detections'] = int(output_dict['num_detections'][0])
  output_dict['detection_classes'] = output_dict[
    'detection_classes'][0].astype(np.int64)
  output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
  output_dict['detection_scores'] = output_dict['detection_scores'][0]
  points, bboxes = [], []
  for k in range(output_dict['num_detections']):
    classId = int(output_dict['detection_classes'][k])
    score = float(output_dict['detection_scores'][k])
    bbox = [float(v) for v in output_dict['detection_boxes'][k]]
    if score > confidence and classId == 1:
      x = bbox[1] * 640
      y = bbox[0] * 640
      right = bbox[3] * 640
      bottom = bbox[2] * 640
      bboxes.append((int(x), int(y), int(right), int(bottom)))
  print("#"*30)
  print(bboxes)
  bboxes = merge_bboxes(bboxes)
  print(bboxes)
  for bbox in bboxes:
    points.append([i*resize_ratio for i in bbox_to_point(bbox, 1)])
  return points

def run_inference_full():
  """"""
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
  nb_images = min(len(cam0), len(cam1), len(cam2))-1

  graph = load_model(PATH_TO_FROZEN_GRAPH)
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

      # Run inference
      i = 0
      while(True):
        image = cam2[i]
        splited_imgs = split_image_divisor(image, 0.01)
        org_img = []
        for j, img in enumerate(splited_imgs):
          im = cv2.resize(img['image'], (640, 640))
          org_img.append(im)
          splited_imgs[j] = np.expand_dims(im, axis=0)

        for j, img in enumerate(splited_imgs):
          output_dict = sess.run(tensor_dict, feed_dict={image_tensor: img})
          # all outputs are float32 numpy arrays, so convert types as appropriate
          output_dict['num_detections'] = int(output_dict['num_detections'][0])
          output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.int64)
          output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
          output_dict['detection_scores'] = output_dict['detection_scores'][0]
          print("#"*80)
          print(i)
          print(j)
          print(output_dict)
          num_detections = output_dict['num_detections']
          bboxes = []
          for k in range(num_detections):
            classId = int(output_dict['detection_classes'][k])
            score = float(output_dict['detection_scores'][k])
            bbox = [float(v) for v in output_dict['detection_boxes'][k]]
            if score > 0.3 and classId == 1:
              x = bbox[1] * 640
              y = bbox[0] * 640
              right = bbox[3] * 640
              bottom = bbox[2] * 640
              bboxes.append((int(x), int(y), int(right), int(bottom)))
          print("#"*30)
          print(bboxes)
          bboxes = merge_bboxes(bboxes)
          print(bboxes)
          for (x, y, right, bottom) in bboxes:
            cv2.rectangle(org_img[j], (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            cv2.circle(org_img[j], tuple(bbox_to_point((x, y, right, bottom), 1)), 10, (0, 0, 200), -1)
          cv2.imshow("sub{}".format(j), org_img[j])
        cv2.imshow("src", cam0[i])

        k = cv2.waitKey(0)
        if(k == 100):
          i = i+1 if i < nb_images else i
        elif(k == 113):
          i = i-1 if i > 0 else i
        elif(k == 27):
          break

def main():
  run_inference_full()

if __name__ == '__main__':
  main()
