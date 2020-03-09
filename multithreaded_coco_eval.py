import logging
from pathlib import Path
from time import time
from functools import partial

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from train_epoch import RetinaNet
import os
import numpy as np
from fixed_pycocotools import FixedCOCOeval as COCOeval
import torch
from tqdm import tqdm
import threading
from time import sleep


# self = RetinaNet()
# dataset = self.get_validation_dataloader().dataset

class MultithreadedCOCOEval:
    def __init__(self, dataset, model, device=None, threshold=0.05):
        self.image_ids, self.results = list(), list()
        self.model = model
        self.dtype = self.get_dtype(model)
        self.dataset = dataset
        self.threshold = threshold
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.training = False
        image_count = len(self.dataset)
        self.images_indices = list(range(image_count))
        self.indices_grouped = np.array_split(self.images_indices, os.cpu_count())
        self.pbar = tqdm(total=image_count, position=0, leave=True, desc='Multithreaded Validation')
        self.processing_complete = False

    def evaluation(self):
        self.__multiprocessor__()
        self.__evaluation__()
        # self.__receptionist__()
        # self.__evaluation__()
        # self.__waiter__()
        # self.__evaluation__()

    def __evaluation__(self):
        if not len(self.results):
            return
        coco_true = self.dataset.coco
        coco_pred = coco_true.loadRes(self.results)
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = self.image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        self.model.train()
        self.pbar.close()
        return

    def __waiter__(self):
        while self.images_indices:
            sleep(5)

    def __receptionist__(self):
        jobs = []
        for cpu_idx in range(0, os.cpu_count()):
            thread = threading.Thread(target=self.__worker__, args=(self.indices_grouped[cpu_idx],))
            jobs.append(thread)
        for j in jobs:
            j.start()
        for j in jobs:
            j.join(5)

    def __worker__(self, index_list):
        with torch.no_grad():
            for index in index_list:
                # self.images_indices.pop(index)  # shrink this list with every workers call
                data = self.dataset[index]
                scale = data['scale']
                img_data = data['img'].permute(2, 0, 1).to(self.dtype).to(self.device)
                self.model.eval()
                self.model.training = False
                scores, labels, boxes = self.model(img_data.unsqueeze(dim=0))
                scores = scores.cpu()
                labels = labels.cpu()
                boxes = boxes.cpu()
                # correct boxes for image scale
                boxes /= scale
                if boxes.shape[0] > 0:
                    boxes[:, 2] -= boxes[:, 0]  # change to (x, y, w, h) (MS COCO standard)
                    boxes[:, 3] -= boxes[:, 1]
                    # compute predicted labels and scores
                    for box_id in range(boxes.shape[0]):
                        score = float(scores[box_id])
                        label = int(labels[box_id])
                        box = boxes[box_id, :]
                        # scores are sorted, so we can break
                        if score < self.threshold:
                            break
                        # append detection for each positively labeled class
                        image_result = {'image_id'   : self.dataset.image_ids[index],
                                        'category_id': self.dataset.label_to_coco_label(label),
                                        'score'      : float(score),
                                        'bbox'       : box.tolist()}
                        # append detection to results
                        self.results.append(image_result)
                # append image to list of processed images
                self.image_ids.append(self.dataset.image_ids[index])
                self.pbar.update(n=1)

    def __partial__(self, index):
        # self.images_indices.pop(index)  # shrink this list with every workers call
        data = self.dataset[index]
        scale = data['scale']
        img_data = data['img'].permute(2, 0, 1).to(self.dtype).to(self.device)
        self.model.eval()
        self.model.training = False
        scores, labels, boxes = self.model(img_data.unsqueeze(dim=0))
        scores = scores.cpu()
        labels = labels.cpu()
        boxes = boxes.cpu()
        # correct boxes for image scale
        boxes /= scale
        if boxes.shape[0] > 0:
            boxes[:, 2] -= boxes[:, 0]  # change to (x, y, w, h) (MS COCO standard)
            boxes[:, 3] -= boxes[:, 1]
            # compute predicted labels and scores
            for box_id in range(boxes.shape[0]):
                score = float(scores[box_id])
                label = int(labels[box_id])
                box = boxes[box_id, :]
                # scores are sorted, so we can break
                if score < self.threshold:
                    break
                # append detection for each positively labeled class
                image_result = {'image_id'   : self.dataset.image_ids[index],
                                'category_id': self.dataset.label_to_coco_label(label),
                                'score'      : float(score),
                                'bbox'       : box.tolist()}
                # append detection to results
                self.results.append(image_result)
        # append image to list of processed images
        self.image_ids.append(self.dataset.image_ids[index])
        self.pbar.update(n=1)

    def __multiprocessor__(self):
        cores = os.cpu_count()
        with ThreadPoolExecutor(max_workers=cores*2) as executor:
            executor.map(self.__partial__, self.images_indices, timeout=30)

    @staticmethod
    def get_dtype(_model):
        model_weights_dtype = [v.dtype for k, v in _model.state_dict().items() if 'weight' in k]
        model_weights_dtype = set(model_weights_dtype)
        if len(model_weights_dtype) != 1:
            return Exception('Too many dtypes returned from model weights.')
        return model_weights_dtype.pop()


if __name__ == '__main__':
    net = RetinaNet()
    net.initialize_training()
    net.initialize_dataloaders()
    self = MultithreadedCOCOEval(net.validation_dataset, net.retinanet)
    # self.__receptionist__()
    self.__multiprocessor__()
    self.__evaluation__()


"""
Found a checkpoint file at: savefiles/checkpoints/1583512758_retinanet_4.pt, the checkpoint made on Friday, 06. March 2020 08:39AM
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.008
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.019
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.006
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.003
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.032
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.006
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.026
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.047
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.128
 
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.008
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.019
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.006
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.004
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.032
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.006
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.025
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.047
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.039
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.127
Multithreaded Validation: 100%|████████████▉| 2845/2846 [21:18<00:00,  2.22it/s]
"""



#
# class MultiThreadedTraining(RetinaNet):
#     def __init__(self, **kwargs):
#         super(MultiThreadedTraining, self).__init__(**kwargs)
#         self.initialize_training()
#         self.initialize_dataloaders()
#         image_count = len(self.training_dataset)
#         self.pbar = tqdm(total=image_count, position=0, leave=True, desc='Multithreaded Validation')
#         self.__multiprocessor__()
#
#     def __partial__(self, data):
#         self.optimizer.zero_grad()
#         img_data = data['img'].to(self.device).to(torch.float32)
#         img_anno = data['annot'].to(self.device).to(torch.float32)
#         cls_loss, box_loss = self.retinanet([img_data, img_anno])
#         del img_data
#         del img_anno
#         with self.amp.scale_loss(cls_loss + box_loss, self.optimizer) as scaled_loss:
#             scaled_loss.backward()
#         torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer), 0.1)
#         self.optimizer.step()
#         del scaled_loss
#         del box_loss
#         del cls_loss
#         self.pbar.update(self.batch_size)
#
#     def __multiprocessor__(self):
#         self.retinanet.training = True
#         self.retinanet.train()
#         self.retinanet.freeze_bn()
#         cores = os.cpu_count()
#         with ThreadPoolExecutor(max_workers=cores * 2) as executor:
#             executor.map(self.__partial__, self.training_dataloader, timeout=30)
#
#     @staticmethod
#     def get_dtype(_model):
#         model_weights_dtype = [v.dtype for k, v in _model.state_dict().items() if 'weight' in k]
#         model_weights_dtype = set(model_weights_dtype)
#         if len(model_weights_dtype) != 1:
#             return Exception('Too many dtypes returned from model weights.')
#         return model_weights_dtype.pop()