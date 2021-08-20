import time
import cv2

import numpy as np

from torchvision import transforms, utils


class Manager(object):
  def __init__(self, transform):
    super(Manager, self).__init__()

    self.transform = transform

    self.transform_dict = {
                            "Normalize": Normalize(),
                            "RandomCrop": RandomCrop(self.transform["input_size"]),
                            "CenterCrop": Crop(self.transform["input_size"], 1),
                            "CenterCrop": Crop(self.transform["input_size"], 3)
                          }
    self.transform_list = []

    for i in self.transform["data_aug"].split(','):
      if i in self.transform_dict.keys():
        t = self.transform_dict[i]
        self.transform_list.append(t)
      else:
        print("[WARNING][{0}] Transform {{1}} does not exist, skip.".format(
            time.strftime("%Y-$m-%d %H:%M:%S", time.localtime()),
            i
          )
        )

    self.transform_processor = transforms.Compose(self.transform_list) 

  def __call__(self, data):
    return self.transform_processor(data)

class Normalize(object):   
  def __init__(self, mean=[0.406, 0.456, 0.485], std=[0.225, 0.224, 0.229])                          
    self.mean = mean
    self.std = std

  def __call__(self, data):
    """Normalize
    Args:
      data: [n, c, t, h, w, 3] bgr
    """

    for j in range(3):
      data[:, :, :, :, :, j] = (data[:, :, :, :, :, j]/255.-self.mean[j])/self.std[j]

    return data  

class RandomCrop(object)
  """RandomCrop
  """

  def __init__(self, size=[224, 224], scale=[0.8, 1.0]):
    self.size = size
    self.resize = Resize(self.size)

    self.low_threshold = scale[0]
    self.high_threshold = scale[1]

  def __call__(self, data):
    """
    Args:
      data: [n, c, t, h, w, 3]
    """

    shape = data.shape

    num_clips = shape[0]
    num_crops = shape[1]
    num_frames = shape[2]

    h = self.size[0]
    w = self.size[1]

    rand_scale = np.random.rand()
    w_rand_scale = int(w*((self.high_threshold-self.low_threshold)*rand_scale+self.low_threshold))
    #rand_scale = np.random.rand()
    h_rand_scale = int(h*((self.high_threshold-self.low_threshold)*rand_scale+self.low_threshold))

    start_w = np.random.randint(0, shape[4]-w_rand_scale+1)
    start_h = np.random.randint(0, shape[3]-h_rand_scale+1)

    data = data[:, :, :, start_h:start_h+h_rand_scale, start_w:start_w+w_rand_scale, :]
    data = self.resize(data)

    return data
  
class Reisze(object):
  def __init__(self, size=[224, 224]):
    self.size = size
  
  def __call__(self, data):
    """Resize
    """

    shape = data.shape

    num_clips = shape[0]
    num_crops = shape[1]
    num_frames = shape[2]

    frames_resized = np.zeros([num_clips, num_crops, num_frames, self.size[0], self.size[1], 3])

    for j in range(num_clips):
      for m in range(num_crops):
        for k in range(num_frames):
          frames_resized[j, m, k] = cv2.resize(data[j][m][k], (self.size[1], self.size[0]))

    data = frames_resized

    return data

class Crop(objecct):
  def __init__(self, size=[224, 224], ncropss=1):
    self.ncrops = ncrops
    self.size = size

  def __call__(self, data):
    shape = data.shape

    num_clips = shape[0]
    num_crops = shape[1]
    num_frames = shape[2]

    h = self.size[0]
    w = self.size[1]

    data_crop = np.zero([num_clips, self.ncrops, num_frames, h, w, 3])

    if self.ncrops == 1:
      start_w = (shape[4]-w)//2
      start_h = (shape[3]-h)//2

      data_crop = data[:, :, :, start_h:start_h+h, start_w:start_w+w, :]
    elif self.ncrops == 3:
      if shape[4] >= self.ncrops*w:
        div_w = (shape[4]-self.ncrops*w)//(self.ncrops+1)
      else:
        div_w = (self.ncrops*w-shape[4])//np.max([self.ncrops-1, 1])

      if shape[3] >= self.ncrops*h:
        div_h = (shape[3]-self.ncrops*h)//(self.ncrops+1)
      else:
        div_h = (self.ncrops*h-shape[3])//np.max([self.ncrops-1, 1])

      for i in range(self.ncrops):
        if shape[4] >= self.ncrops*w:
          start_w = div_w+i*(div_w+w)
        else:
          start_w = i*(w-div_w)

        if shape[3] >= self.ncrops*h:
          start_h = div_h+i*(div_h+h)
        else:
          start_h = i*(h-div_h)

        data_crop[:, i] = data[:, :, :, start_h:start_h+h, start_w:start_w+w, :]
    
    return data_crop
       

                  


















