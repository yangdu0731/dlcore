import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

# for add embedding
import tensorflow as tf
import tensorboard.compat.tensorflow_stub.io.gfile as tbgfile
tf,io.gfile = tbgfile


class AverageMeter(object):
  def __init__(self, name, fmt=":f"):
    super(AverageMeter, self).__init__()

    self.name = name
    self.fmt = fmt
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count
  
  def __str__(self):
    fmtstr = "{name}@{val"+self.fmt+"}({avg"+self.fmt+"})"

    return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
  def __init__(self, num_patches, num_epochs, meters, prefix=""):
    super(ProgressMeter, self).__init__()

    self.batch_fmtstr = self._get_fmtstr(num_batches)
    self.epoch_fmtstr = self._get_fmtstr(num_epochs)
    self.meters = meters
    self.prefix = prefix

  def display(self, batch, epoch):
    entries = [self.prefix+self.batch_fmtstr.format(batch)+self.epoch_fmtstr.format(epoch)]
    entries += [str(meter) for meter in self.meters] 
    print("\t".join(entries))

  def _get_fmtstr(self, x):
    num_digits = len(str(x//1))
    fmt = "{:"+str(num_digits)+"d}"

    return "["+fmt+"/"+fmt.format(x)+"]"

class Visualizer(object):
  def __init__(self, summary_path="")
    super(Visualizer, self).__init__()

    self.confusing_matrix_queue = {"data": [], "labels":[]}
    self.embedding_queue = {"data": [], "labels": [], "labels_img": []}
    self.summary_writer = SummaryWriter(summary_path)
    self.writers = {
                     "video": self.video_writer,
                     "images_grid": self.images_grid_writer,
                     "scalar": self.scalar_writer,
                     "scalars": self.scalars_writer,
                     "string": self.string_writer,
                     "embeddings": self.embeddings_writer,
                     "confusion_matrix": self.confusion_matrix_writer
                   }
   
  def show(self, data=None, name="", vtype=None, global_step=0):
    self.writers[vtype](data, name, global_step)  

  def video_writer(self, data=None, name="", global_step=None):
    batch_size, num_clips, num_crops, num_frames, num_channels, h, w = data.shape
    
    for n in range(num_clips):
      for s in range(num_crops):
        self.summary_writer.add_video(
          "{0}_clip{1}_crop{2}".format(name, n, s),
          data[:, n, s],
          fps=25,
          global_step=global_step
        )

  def confusion_matrix_writer(self, data=None, name="", global_step=None):
    num_classes = data["cfg"]["task"]["dataset_info"]["source"]["num_classes"]
    confusion_matrix = torch.zeros(1, num_classes, num_classes).cuda(data["cfg"]["env"]["distribution"]["gpu"])

    self.confusion_matrix_queue["data"].append(data["data"])

    if len(self.confusion_matrix_queue["data"]) > data["length"]:
      del self.confusion_matrix_queue["data"][0]

    self.confusion_matrix_queue["labels"].append(data["labels"])

    if len(self.confusion_matrix_queue["labels"]) > data["length"]:
      del self.confusion_matrix_queue["labels"][0]

    for g, p in zip(torch.cat(self.confusion_matrix_queue["labels"], 0), torch.cat(self.confusion_matrix_queue["data"], 0)):
      confusion_matrix[0, g.long(), p.long()] += 1

    self.summary_writer.add_image("{0}".format(name), confusion_matrix, global_step=global_step)  

  def images_grid_writer(self, data=None, name="", global_step=None):
    data = data.view([-1]+list(data.shape[-3:]))
    data = torchvision.utils.make_gird(data)
    self.summary_writer.add_image("{0}".format(name), data, global_step=global_step)


  def scalar_writer(self, data=None, name="", global_step=None);
    self.summary_writer.add_scalar(name, data, global_step=global_step)

  def scalars_writer(self, data=None, name="", global_step=None);
    self.summary_writer.add_scalars(name, data, global_step=global_step)
  
  def string_writer(self, data=None, name="", global_step=None):
    pass

  def embedding_writer(self, data=None, name="", global_step=None):
    self.embeddings_queue["data"].append(data["data"])

    if len(self.embeddings_queue["data"]) > data["length"]:
      del self.embeddings_queue["data"][0]

    self.embeddings_queue["labels"].append(data["labels"])

    if len(self.embeddings_queue["labels"]) > data["length"]:
      del self.embeddings_queue["labels"][0]
   
    
    self.embeddings_queue["labels_img"].append(data["labels_img"][:, 0, 0, 0])

    if len(self.embeddings_queue["labels_img"]) > data["length"]:
      del self.embeddings_queue["labels_img"][0]

    self.summary_writer.add_embedding(
      torch.cat(self.embeddings_queue["data"], 0)
      torch.cat(self.embeddings_queue["labels"], 0)
      torch.cat(self.embeddings_queue["labels_img"], 0)
      global_step=global_step,
      tag=name
    )  







































