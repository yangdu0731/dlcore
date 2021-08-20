import os
import cv2
import datasets

import pandas as pd
import numpy as np
import torch.nn.functional as F

from torch.utils.data import Dataset


class Manager(Dataset):
  def __init__(self, cfg):
    super(Manager, self).__init__()

    self.cfg = cfg

    self.cfg["info"] = pd.read_csv(self.cfg["info_dir"], " ", header=None)


    #self.preproceess = preprocess
    #self.transform = datasets.transform.Manager(self, preprocess["transform"])

  def __len__(self):
    return len(self.info)

  #def __getitem__(self, idx):
  #  if self.source_type == "video_frames":
  #    iterm_path, iterm_num_frames, iterm_label = self.info.iloc[idx]

  #    frames = np.zeros(
  #               [
  #                 self.preprocess["num_clips"],
  #                 1,
  #                 self.preprocess["num_frames"],
  #                 self.preprocess["data_size"][0],
  #                 self.preprocess["data_size"][1],
  #                 3
  #               ]
  #             ) # [n, 1, t, h, w, 3]

  #    frames_range = (self.preprocess["num_frames"]-1) * self.preprocess["stride"] + 1         

  #    if iterm_num_frames >= self.preprocess["num_clips"]*frames_range:
  #      div_frames_range = (iterm_num_frames-self.preprocess["num_clips"]*frames_range)) // \
  #                         (self.preprocess["num_clips"] + 1)
  #    else:
  #      div_frames_range = (self.preprocess["num_clips"]*frames_range - iterm_num_frames) // \
  #                         np.max([self.preprocess["num_clips"], 1])

  #    for n in range(self.preprocess["num_clips"]):
  #      if self.preprocess["num_clips"] <= 1:
  #        start_idx = np.random.randint(
  #                      0,
  #                      np.max([iterm_num_frames-frames_range, 0])+1
  #                    )
  #      elif iterm_num_frames >= self.preprocess["num_clips"]*frames_range:
  #        start_idx = div_frames_range+n*(div_frames_range+frames_range)
  #      elif iterm_num_frames < self.preprocess["num_clips"]*frames_ranage:
  #        start_idx = n*(frames_range-div_frames_range)
  #      
  #      for t in range(self.preprocess["num_frames"]):
  #        frame_idx = start_idx+t*self.preprocess["stride"]+1
  #        frame_idx = np.clip(frame_idx, 1, iterm_num_frames)
  #        frame_path = os.path.join(
  #                       self.dataset_dir,
  #                       iterm_path.split('.')[0],
  #                       self.data_format%(frame_idx)
  #                     )

  #        with open(frame_path, "rb") as f:
  #          frame_str = f.read()

  #        frame = cv2.imdecode(np.frombuffer(frame_str, np.uint8), cv2.IMREAD_COLOR) # bgr
  #        frame = cv2.resize(frame, (self.preprocess["data_size"][1], self.preprocess["data_size"][0])
  #        frames[n, 0, t, :, :, :] = frame

  #    data = frames
  #    #
  #    data = np.transpose(data, [0, 1, 2, 5, 3, 4]).astype(np.float32)
  #    target = iterm_label
  #  elif self.source_type == "video":
  #    pass
  #  elif self.source_type == "image":
  #    pass
  #  

  #  return data, target
