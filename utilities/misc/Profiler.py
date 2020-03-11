from train_epoch import *
import torch.cuda.profiler
from torch.autograd.profiler import profile

self = RetinaNet()
self.initialize_training()

img_data = torch.randn((8, 3, 768, 1376), requires_grad=True)
img_anno = torch.randn((8, 49, 5), requires_grad=True)
x = [img_data.cuda().half(), img_anno.cuda()]
with torch.cuda.profiler.profile():
    self.retinanet(x) # Warmup CUDA memory allocator and profiler
    with torch.autograd.profiler.emit_nvtx():
        self.retinanet(x)
