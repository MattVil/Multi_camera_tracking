DATASET_PATH = '/workspace/Dataset/soccer'
VIDEO_NAME = "0056_2013-11-03 18:01:14.248366000.h264"

CALIBRATION_FILES = ["calibration_cam0.npy", "calibration_cam1.npy",                                       "calibration_cam2.npy"]

MODEL_NAME = "ssd_resnet50_v1_fpn_shared_box_predictor_640x640_coco14_sync_2018_07_03"
PATH_TO_FROZEN_GRAPH = "model/" + MODEL_NAME + "/frozen_inference_graph.pb"
PATH_TO_LABELS = "model/" + MODEL_NAME + "/mscoco_label_map.pbtxt"

