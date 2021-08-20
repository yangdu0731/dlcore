import sys
import torch
import networks
import utils
import time
import os
import shutil

import numpy as np


class Manager(object):
  def __init__(self, cfg, dataset):
    super(Manager, self).__init__()

    self.cfg = cfg
    self.dataset = dataset

    self.model = getattr(models, se;f.cfg["task"]["model"].Manager(self.cfg))
    torch.cuda.set_device(self.cfg["env"]["distribution"]["gpu"])
    self.model.cuda(self.cfg["env"]["distribution"]["gpu"])
    self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.cfg["env"]["distribution"]["gpu"]])

    for para in list(self.model.parameters): para.requires_grad = True
    
    self.criterion = torch.nn.CrossEntropyLoss().cuda(self.cfg["env"]["distribution"]["gpu"])
    self.optimizer = torch.optim.SGD(
                       self.model.parameters(),
                       self.cfg["task"]["train"]["learning_rate"],
                       momentum=self.cfg["task"]["train"]["momentum"],
                       weight_decay=self.cfg["task"]["train"]["weight_decay"]
                     )


     self.train_steps_per_epoch = int(np.ceil(len(self.dataset.train_dataaset)/(self.cfg["task"]["train"]["batch_size"]*self.cfg["args"].world_size)))                
     #self.val_steps_per_epoch = int(np.ceil(len(self.dataset.val_dataaset)/(self.cfg["task"]["val"]["batch_size"]*self.cfg["args"].world_size)))                
     self.val_steps_per_epoch = int(np.ceil(len(self.dataset.val_dataaset)/(self.cfg["task"]["train"]["batch_size"]//self.cfg["env"]["distribution"]["ngpus_per_node"])))                

     self.lr = torch.optim.lr_scheduler.MultiStepLR(
                 self.optimizer,
                 [i*self.train_steps_per_epoch for i in self.cfg["task"]["train"]["epochs"]],
                 gamma=0.1
               )

     # ckpt
     if not os.path.exists(self.cfg["args"].train_dir):
       if self.cfg["env"]["distribution"]["rank"] % self.cfg["env"]["distribution"]["ngpus_per_node"] == 0:
         os.makedirs(self.cfg["args"].train_dir)

       self.start_epoch = 0
       self.best_acc1 = 0
     else:
       model_list = [f for f in os.listdir(self.cfg["args"].train_dir) if os.path.isfile(os.path.join(self.cfg["args"].train_dir, f))]

       if len(model_list) != 0:
         if self.cfg["args"].mode == "validate":
           load_model = "{0}_bast.pth.tar".format(self.cfg["task"]["model"])
         elif self.cfg["args"].mode == "train":
           load_model = "{0}.pth.tar".format(self.cfg["task"]["model"])

         load_path = os.path.join(
                       self.cfg["args"].train_dir,
                       load_model
                     )  
          
         checkpoint = torch.load(load_path, map_location="cuda:{}".format(self.cfg["env"]["distribution"]["gpu"]))
         self.start_epoch = checkpoint["epoch"]
         self.best_acc1 = checkpoint["best_acc1"]

         self.model.load_state_dict(checkpoint["state_dict"])
         self.optimizer.load_state_dict(checkpoint["optimizer"])

         print("[INFO][0][RANK{1}] Loading from: {2}, epoch: {3}, best acc1: {4}".format(
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
             self.cfg["env"]["distribution"]["rank"],
             load_path,
             self.start_epoch,
             self.best_acc1
           )
         )
     else:
       if self.cfg["args"].mode == "validate":
         print("[INFO][{0}][RANK{1}] No ckpt {{{2}/{3}_best.pth.tar}} for validate.".format(
             time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
             self.cfg["env"]["distribution"]["rank"],
             self.cfg["args"].train_dir,
             sekf,cfg["task"]["model"]
           )
         )

         sys,exit(0)
       
       self.start_epoch = 0
       self.best_acc1 = 0

   if self.cfg["env"]["distribution"]["rank"] % self.cfg["env"]["distribution"]["ngpus_per_node"] == 0:
     print("[INFO][{0}][RANK{1}] Saving into: {2}".format(
         time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
         self.cfg["env"]["distribution"]["rank"],
         self.cfg["args"].train_dir
       )
     )



  def run(self):
    if self.cfg["args"].mode == "validate":
      self.validate(self.start_epoch-1)
      return

    for epoch in range(self.start_epoch, self.cfg["task"]["train"]["epochs"][-1]):  
      self.train((epoch)

      if ((epoch+1)%self.cfg["task"]["val"]["epochs"] == 0) or \
         epoch == self.cfg["task"]["train"]["epochs"][-1]-1:
         acc1 = self.validate(epoch)
         is_best = acc1 > self.best_acc1
         self.best_acc1 = max(acc1, self.best_acc1)

         if self.cfg["env"]["distribution"]["rank"]%self.cfg["env"]["distribution"]["ngpus_per_node"] == 0:
           self.save_checkpoint(
             {  
               "epoch": epoch+1,
               "arch": self.cfg["task"]["model"],
               "state_dict": self.model.state_dict(),
               "best_acc1": self.best_acc1,
               "optimizer": self.optimizer.state_dict(),
             },
             is_best,
             os.path.join(self.cfg["args"].train_dir, "{0}.pth.tar".format(self.cfg["task"]["model"]))
           )

  def validate(self, epoch):
    self.model.eval()

    batch_time = utils.metric.AverageMeter("Time", ":4.3f")
    data_time = utils.metric.AverageMeter("Data", ":4.3f")
    losses = utils.metric.AverageMeter("Lcls", ":5.3f")
    top1 = utils.metric.AverageMeter("Acc1", ":6.2f")
    top5 = utils.metric.AverageMeter("Acc5", ":6.2f")


    progress = utils.metric.ProgressMeter(
                 os.path.join(
                   self.cfg["args"].train_dir,
                   "visualization",
                   "rank{0}_epoch{1}".format(
                     self.cfg["env"]["distribution"]["rank"],
                     epoch+1
                   )
                 )
               )

    end = time.time()

    with torch.no_grad():
      for step, (data, target) in enumerate(self.dataset.val_loader):
        data_time.update(time.time()-end)

        data = data.cuda(self.cfg["env"]["distribution"]["gpu"], non_blocking=True)
        target = target.cuda(self.cfg["env"]["distribution"]["gpu"], non_blocking=True)

        output = self.model(data)
        self.cls_loss = self.criterion(output, target)

        acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
        losses.update(self.cls_loss.item(), data.size(0))
        top1.update(acc1[0], data.size[0])
        top5.update(acc5[0], data.size[0])

        if step%np.max([self.val_steps_per_epoch//10, 1])==0 or step==self.val_steps_per_epoch-1:
          progress.display(step+1. epoch+1)

          if self.cfg["args"].debug and self.cfg["args"].mode == "train":
            tmp = data[:, :, :, :, 0, :, :]
            data[:, :, :, :, 0, :, :] = data[:, :, :, :, 2, :, :]*0.229+0.485
            data[:, :, :, :, 1, :, :] = data[:, :, :, :, 1, :, :]*0.224+0.456
            data[:, :, :, :, 2, :, :] = tmp*0.225+0.406

            visualizer.show(data, "val_inputs", "video")
            visualizer.show(data, "val_inputs", "images_grid")

            visualizer.show({"data": output, "labels": target, "labels_img": data, "length": 5},
              "val_outputs", "embeddings", epoch*self.val_steps_per_epoch+step
            )
            visualizer.show({"data": torch.argmax(output, -1), "labels": target, "length": 12, "cfg": self.cfg},
              "val_confusion_matrix", "confusion_matrix", epoch*self.val_steps_per_epoch+step
            )

         batch_time.update(time.time()-end)
         end = time.time()
    
      ## loss
      if self.cfg["args"].debug and self.cfg["args"].mode == "train":
        visualizer.show({"val": losses.avg}, "cls_loss", "scalars", (epoch+1)*self.train_steps_per_epoch-1)
        visualizer.show({"val": top1.avg}, "acc1", "scalars", (epoch+1)*self.train_steps_per_epoch-1)
        visualizer.show({"val": top5.avg}, "acc5", "scalars", (epoch+1)*self.train_steps_per_epoch-1)
    
      return top1.avg

  def train(self, epoch):
    self.model.train()
    self.dataset.train_sampler.set_epoch(epoch)

    batch_time = utils.metric.AverageMeter("Time", ":4.3f")
    data_time = utils.metric.AverageMeter("Data", ":4.3f")
    losses_time = utils.metric.AverageMeter("Lcls", ":5.3f")
    top1_time = utils.metric.AverageMeter("Acc1", ":6.2f")
    top5_time = utils.metric.AverageMeter("Acc5", ":6.2f")

    progress = utils.metric.ProgressMeter(
      len(self.dataset.train_loader),
      sefl.cfg["task"]["train"]["epochs"][-1],
      [batch_time, data_time, losses, top1, top5],
      prefix="[TRAINING][RANK{0}]".format(
        self.cfg["env"]["distribution"]["rank"]
      )
    )

    visualizer = utils.metric.Visualizer(
                   os.path.join(
                     self.cfg["args"].train_dir,
                     "Visualization",
                     "rank{0}_epoch{1}".format(
                       self.cfg["env"]["distribution"]["rank"],
                       epoch+1
                     )
                   )     
            
                 )

     end = time.time()

     for step, (data, target) in enumerate(self.dataset.train_loader):
       data_time.update(time.time()-end)

       data = data.cuda(self.cfg["env"]["distribution"]["gpu"], non_blocking=True)
       target = target.cuda(self.cfg["env"]["distribution"]["gpu"], non_blocking=True)

       output = self.model(data)
       self.cls_loss = self.criterion(output, target)

       acc1, acc5 = self.accuracy(output, target, topk=(1, 5))
       losses.update(self.cls_loss.item(), data.size(0))
       top1.update(acc1[0], data.size(0))
       losses.update(acc5[0], data.size(0))
     
       if step%np.max([self.train_steps_per_epoch//10, 1])==0 or step==self.train_steps_per_epoch-1:
         # log
         progress.display(step+1, epoch+1)

         if self.cfg["args"].debug:
           tmp = data[:, :, :, :, 0, :, :]
           data[:, :, :, :, 0, :, :] = data[:, :, :, :, 2, :, :]*0.229+0.485
           data[:, :, :, :, 1, :, :] = data[:, :, :, :, 1, :, :]*0.224+0.456
           data[:, :, :, :, 2, :, :] = tmp*0.225+0.406

           visualizer.show(data, "inputs", "video")
           visualizer.show(data, "inputs", "images_grid")

           ## output: tSNE
           visualizer.show({"data": output, "labels": target, "labels_img": data, "length": 5},
             "train_outputs", "embeddings", epoch*self.train_steps_per_epoch+step
           )
           visualizer.show({"data": torch.argmax(output, -1), "labels": target, "length": 12, "cfg": self.cfg},
             "train_confusion_matrix", "confusion_matrix", epoch*self.train_steps_per_epoch+step
           )


           ## lr, loss, acc: curve
           visualizer.show(self.optimizer.state_dict()["param_groups"][0]["lr"], "learning rate", "scalar",
                           epoch*self.train_step_per_epoch+step
           )
           visualizer.show({"train": self.cls_loss.item()}, "cls_loss", "scalars", epoch*self.train_steps_per_epoch+step)
           visualizer.show({"train": acc1[0]}, "acc1", "scalars", epoch*self.train_steps_per_epoch+step)
           visualizer.show({"train": acc5[0]}, "acc5", "scalars", epoch*self.train_steps_per_epoch+step)

       # Backward
       self.optimizer.zero_grad()
       self.cls_loss.backward()
       self.optimizer.step()
       self.lr.step()

       batch_time.update(time.time()-eend)
       end = time.time()

  def accuracy(self, output, target, topk=(1, )):
    with torch.no_grad():
      maxk = max(topk)
      batch_size = target.size(0)

      _, pred = output.topk(maxk, 1, True, True)
      pred = pred.t()
      correct = pred.eq(target.view(1, -1).expand_as(pred))

      res = []
      for k in topk:
        correct_k = correct[:k].reshape[-1].float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0/batch_size))

  def save_checkpoint(self, state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

    if is_best: shutil.copyfile(filename, filename.split('.')[0]+"_best.pth.tar")


































    
