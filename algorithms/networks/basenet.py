import torch
import torchvision

import torch.nn as nn

class Manager(torch.nn.Module):
  def __init__(self, cfg):
    super(Manager, self).__init__()

    self.cfg = cfg

    self.backbone = getattr(torchvision.modelsm self.cfg["task"]["model_info"]["backbone"]["net"])(pretrained=self.cfg["task"]["model_info"]["backbone"]["pretrain"])

    self.fc = nn.Sequential(
                nn.Dropout(self.cfg["task"]["model_info"]["backbone"]["dropout"]),
                nn.Linear(in_features=2048, out_features=self.cfg["task"]["data_info"]["source"]["num_classes"], bias=True)
              )

    setattr(self.backbone, "fc", self.fc) 

  def forward(self, x):
    batch_size, num_clips, num_crops, num_frames, num_channels, h, w = x.shape
    x = x.view(-1, num_channels, h, w)
    x = self.backbone(x)
    x = x.view(batch_size, num_clips, num_crops, num_frames, -1)
    x = torch.mean(x, dim=[1, 2, 3])

    return x
