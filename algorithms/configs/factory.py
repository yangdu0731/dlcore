import torch
import yaml
import random
import argparse
import os
import datasets
import networks
import time

import numpy as np


class Manager(object):
  def __init__(self):
    super(Manager, self).__init__()
    print("[START][{0}] Algorithm".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

    self.cfg = {}

    # Devicv/Datasets/Networks Settings
    self.cfg["args"] = self.parse_args()
    self.cfg["env"] = self.read_yaml("./configs/settings/env.yaml")
    self.cfg["task"] = self.read_yaml("./configs/settings/" + self.cfg["args"].task + ".yaml")
    self.set_env()
    self.log()

  def distributed_run(self, gpu):
    self.cfg["env"]["distribution"]["gpu"] = gpu
    self.cfg["env"]["distribution"]["rank"] = self.cfg["env"]["distribution"]["rank"]*self.cfg["env"]["distribution"]["ngpus_per_node"]+self.cfg["env"]["distribution"]["gpu"] 

    torch.distributed.init_process_group(
      init_method = self.cfg["env"]["distribution"]["dist_url"],
      backend = self.cfg["env"]["distribution"]["dist_backend"],
      world_size = self.cfg["env"]["distribution"]["world_size"],
      rank = self.cfg["env"]["distribution"]["rank"]
    )

    self.dataset = datasets.factory.Manager(self.cfg)
    self.network = networks.factory.Manager(self.cfg, self.dataset)
    self.network.run()


  def run(self):
    torch.multiprocessing.spawn(
      self.distributed_run,
      nprocs = self.cfg["env"]["distribution"]["ngpus_per_node"]
    )

  def set_env(self):
    # Distribution
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg["env"]["distribution"]["gpus"]

    self.cfg["env"]["distribution"]["dist_url"] = self.cfg["args"].dist_url
    self.cfg["env"]["distribution"]["ngpus_per_node"] = torch.cuda.device_count()
    self.cfg["env"]["distribution"]["rank"] = self.cfg["args"].rank
    self.cfg["env"]["distribution"]["world_size"] *= self.cfg["env"]["distribution"]["ngpus_per_node"]

    # Reproducity
    os.environ["PYTHONHASHSEED"] = str(self.cfg["env"]["reproducity"]["seed"])
    torch.manual_seed(self.cfg["env"]["reproducity"]["seed"])
    torch.cuda.manual_seed_all(self.cfg["env"]["reproducity"]["seed"])
    random.seed(self.cfg["env"]["reproducity"]["seed"])
    np.random.seed(self.cfg["env"]["reproducity"]["seed"])
    torch.backends.cudnn.benchmark = self.cfg["env"]["reproducity"]["benchmark"]
    torch.backends.cudnn.deterministic = self.cfg["env"]["reproducity"]["deterministic"]

  def read_yaml(self, file_path):
    class Loader(yaml.SafeLoader):
      def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

      def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
          return yaml.load(f, Loader)

    Loader.add_constructor("!include", Loader.include)      

    with open(file_path, 'r') as f:
      res = yaml.load(f, Loader)

    return res
    
  def parse_args(self):
    parser = argparse.ArgumentParser(description="Arg. Parser")

    parser.add_argument("--task", dest="task", default="video_classification", type=str)
    parser.add_argument("--mode", dest="mode", default="train", type=str)
    parser.add_argument("--debug", dest="debug", action="store_true")
    parser.add_argument("--tag", dest="tag", default="", type=str)
    parser.add_argument("--dist_url", dest="dist_url", default="tcp://127.0.0.1:8080", type=str)
    parser.add_argument("--rank", dest="rank", default=0, type=int)

    args, unknown = parser.parse_known_args()

    return args

  def log(self):
    # Summary
    print("[TASK][{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.cfg["args"]))

    # Environment
    print("[ENV][{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.cfg["env"]))
    
    # Dataset
    print("[DATASET][{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.cfg["task"]["dataset"]))

    # Networks
    print("[NETWORKS][{0}] {1}".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()), self.cfg["task"]["network"]))
