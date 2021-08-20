import datasets
import torch


class Manager(object):
  def __init__(self, cfg):
    super(Manager, self).__init__()

    self.cfg = cfg

    print(self.cfg)

    self.train_dataset = datasets.loader.Manager(
                           self.cfg["task"]["dataset"]["train"]
                         )

    self.val_dataset = datasets.loader.Manager(
                           self.cfg["task"]["dataset"]["val"]
                         )                     


    self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset)  
    # self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False)


    self.train_loader = torch.utils.data.DataLoader(
                          self.train_dataset,
                          batch_size=self.cfg["task"]["train"]["batch_size"]//self.cfg["env"]["distribution"]["ngpus_per_node"],
                          shuffle=False,
                          num_workers = int((self.cfg["env"]["distribution"]["num_workers"]+self.cfg["env"]["distribution"]["ngpus_per_node"]-1)/self.cfg["env"]["distribution"]["ngpus_per_node"]),
                          pin_memory=True,
                          sampler=self.train_sampler
                        )

    self.val_loader = torch.utils.data.DataLoader(
                          self.val_dataset,
                          batch_size=self.cfg["task"]["val"]["batch_size"]//self.cfg["env"]["distribution"]["ngpus_per_node"],
                          shuffle=False,
                          num_workers = int((self.cfg["env"]["distribution"]["num_workers"]+self.cfg["env"]["distribution"]["ngpus_per_node"]-1)/self.cfg["env"]["distribution"]["ngpus_per_node"]),
                          pin_memory=True,
                          #sampler=self.val_sampler
                        )
