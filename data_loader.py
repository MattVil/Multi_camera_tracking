import os
import cv2
from abc import ABC, abstractmethod

class DataLoader(ABC):
  """"""

  def __init__(self):
    pass

  @abstractmethod
  def get_next_frames(self):
    pass

class StreamLoader(DataLoader):
  """"""

  def __init__(self, streams_n):
    DataLoader.__init__(self)
    self.streams_n = streams_n
    self.streams = self.__setup_streams()

  def __setup_streams(self):
    """"""
    streams = []
    for stream_n in self.streams_n:
      streams.append(cv2.VideoCapture(stream_n))
    return streams
    

  def get_next_frames(self):
    """"""
    frames = []
    ret = True
    for stream in self.streams:
      if(not stream.isOpened()):
        return [], False
      frame, r = stream.read()
      frames.append(frame)
      ret = ret and r
    return frames, ret

class VideoLoader(DataLoader):
  """"""

  def __init__(self, videos_n):
    DataLoader.__init__(self)
    self.videos_n = videos_n
    self.videos = self.__setup_videos()
    self.current_frame_idx = 0

  def __setup_videos(self):
    """"""
    videos =[]
    for i, video_n in enumerate(self.videos_n):
      assert(os.path.exists(video_n))
      videos.append([])
      cap = cv2.VideoCapture(video_n)
      ret = True
      p = 0
      while(cap.isOpened()and ret):
        p = p+1
        ret, frame = cap.read()
        videos[i].append(frame)
    # cut videos to the same size 
    videos = [v[:min([len(x) for x in videos])] for v in videos]
    self.nb_frame = len(videos[0]) 
    return videos

  def get_next_frames(self):
    """"""
    if(self.current_frame_idx < self.nb_frame-1):
      self.current_frame_idx = self.current_frame_idx + 1
    frames = [vid[self.current_frame_idx] for vid in self.videos]
    return frames, True  
    
  def get_previous_frames(self):
    """"""
    if(self.current_frame_idx > 0):
      self.current_frame_idx = self.current_frame_idx - 1
    frames = [vid[self.current_frame_idx] for vid in self.videos]
    return frames, True