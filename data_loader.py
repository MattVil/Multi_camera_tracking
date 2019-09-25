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
    super(DataLoader, self).__init__()
    self.streams_n = streams_n
    self.streams = self.__setup_streams()

  def __setup_streams(self):
    """"""
    for stream_n in self.streams_n:
      self.streams = cv2.VideoCapture(stream_n)

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
    super(DataLoader, self).__init__()
    self.videos_n = videos_n
    self.videos = self.__setup_videos()
    self.current_frame_idx = 0

  def __setup_videos(self):
    """"""
    for i, video_n in enumerate(self.videos_n):
      self.videos.append([])
      cap = cv2.VideoCapture(video_n)
      while(cap.isOpened()):
        frame, r = cap.read()
        self.videos[i].append(frame)
      # cut videos to the same size 
      self.videos = [v[:min([len(x) for x in self.videos])] for v in self.videos]
      self.nb_frame = len(self.videos[0]) 

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