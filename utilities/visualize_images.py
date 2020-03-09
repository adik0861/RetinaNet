from dataloader import *
from torch.utils.data import DataLoader
from torchvision import transforms
from numpy import random
import re
import skimage
from skimage import io
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np
from train_epoch import RetinaNet
from torchvision.transforms import ToPILImage, ToTensor

self = RetinaNet()
self.initialize_dataloaders()
root_dir = self.training_dataloader.root_dir


def unnormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def vis(trainset=True):
    if trainset is True:
        dataset = self.training_dataloader
    else:
        dataset = self.validation_dataset
    image_tensor = next(iter(dataset))
    img = image_tensor['img'][0, :, :]
    scale = image_tensor['scale'][0]
    _annot = image_tensor['annot'][0, :, :]
    annot = _annot.clone()
    file_name = os.path.join('/home/adityakunapuli/data/train/images', image_tensor['info'][0]['file_name'])
    # im = np.array(Image.open(file_name), dtype=np.uint8)
    im = (ToPILImage()(unnormalize(img)))
    fig = plt.figure(figsize=(16, 9))
    # fig, ax = plt.subplots(1)
    ax = fig.add_subplot(111)
    ax.imshow(im, origin='upper', interpolation='none')
    for i in range(annot.shape[0]):
        print(annot[i, :4])
        x1, y1 = annot[i, :2]  # numpy indexing is exclusive--i.e. :2 means up to 2 but not 2
        x2, y2 = annot[i, 2:4]
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(xy=(x1, y1), width=w, height=h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f'{int(x1), int(y1)}')
        # ax.annotate(f'({int(x1)}, {int(y1)})', xy=(x1, y1), xycoords='data',
        #             xytext=(0.01, 0.01), textcoords='axes fraction',
        #             arrowprops=dict(arrowstyle='wedge'))
    plt.show()


def get_sample(idx_or_tensor):
    if isinstance(idx_or_tensor, int):
        _sample = dataset.naive_getitem(idx_or_tensor)
    else:
        _sample = idx_or_tensor
    img = _sample['img']
    annot = _sample['annot']
    im = img
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.imshow(im, origin='upper', interpolation='none')
    for i in range(annot.shape[0]):
        x1, y1, x2, y2, lbl = annot[i, :]
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(xy=(x1, y1), width=w, height=h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f'{int(x1), int(y1)}')
    plt.show()
    return _sample


def visualize_base_image(tensor):
    # just to verify cropping
    if isinstance(tensor, dict):
        tensor = tensor['img']
    im = tensor
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.imshow(im, origin='upper', interpolation='none')
    plt.show()

np.set_printoptions(suppress=True)

def visual_entire_sample(image_tensor):
    img = image_tensor['img']
    _annot = image_tensor['annot']
    annot = _annot.clone()
    im = img
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.imshow(im, origin='upper', interpolation='none')
    for i in range(annot.shape[0]):
        x1, y1 = annot[i, :2]
        x2, y2 = annot[i, 2:4]
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(xy=(x1, y1), width=w, height=h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f'{int(x1), int(y1)}')
    plt.show()



# TODO: WORKS WITH VAL
"""
root_dir = self.validation_dataset.root_dir
def vis(dataset):
    image_tensor = next(iter(dataset))
    img = image_tensor['img']
    scale = image_tensor['scale']
    _annot = image_tensor['annot']
    annot = _annot.clone()
    file_name = os.path.join(root_dir, image_tensor['info']['file_name'])
    im = np.array(Image.open(file_name), dtype=np.uint8)
    fig, ax = plt.subplots(1)
    ax.imshow(im,origin='upper')
    for i in range(annot.shape[0]):
        print(annot[i, :4])
        x1, y1 = annot[i, :2]  # numpy indexing is exclusive--i.e. :2 means up to 2 but not 2
        x2, y2 = annot[i, 2:4]
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(xy=(x1, y1), width=w, height=h, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(x1, y1, f'{int(x1), int(y1)}')
        # ax.annotate(f'({int(x1)}, {int(y1)})', xy=(x1, y1), xycoords='data',
        #             xytext=(0.01, 0.01), textcoords='axes fraction',
        #             arrowprops=dict(arrowstyle='wedge'))
    plt.show()

file_name
a = list(range(255, 275, 1))
string = '/home/adityakunapuli/data/train/images/uav0000243_00001_v/0000{num}.jpg'
img_list = [string.format(num=x) for x in list(range(255, 275, 1))]



num_images = 10
assert num_images % 2 == 0 and num_images > 0
idx = random.randint(1, len(self.training_dataset))
idxs = list(range(idx - int(num_images / 2), idx + int(num_images / 2), 1))
# images = [self.training_dataset[self.training_dataset.image_ids[x]] for x in idxs]
image_id_idxs = [self.training_dataset.image_ids.index(x) for x in idxs]
images = [self.training_dataset[x] for x in image_id_idxs]

# string = re.sub(r'([0]+)([1-9]\d+)(\.jpg)', r'\1{num}\3', file_name)
# file_num = int(re.findall(r'[0]+([1-9]\d+)\.jpg', file_name)[0])
#
# img_list = [string.format(num=x) for x in list(range(file_num - split, file_num + split, 1))]

for img in images:
    file_name = os.path.join(root_dir, img['info']['file_name'])
    annot = img['annot']
    plt.cla()
    plt.axis("off")
    im = np.array(Image.open(file_name), dtype=np.uint8)
    plt.imshow(im)
    _annot = annot.clone()
    mask = _annot[:, 4] != 1
    a = _annot[mask, :]

    for i in range(a.shape[1]):
        # print(a[i, :4])
        x1, y1 = a[i, :2]  # numpy indexing is exclusive--i.e. :2 means up to 2 but not 2
        x2, y2 = a[i, 2:4]
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle(xy=(x1, y1), width=w, height=h, linewidth=1, edgecolor='r', facecolor='none')
        plt.gca().add_patch(rect)
        plt.gca().text(x1, y1, f'{int(x1), int(y1)}')
        plt.gca().text(0, 0, f'{file_name}')
    plt.show()

from numpy import genfromtxt

anno_file = '/home/adityakunapuli/data/val/annotations/uav0000086_00000_v.txt'
json_image_id = 1
annotations = genfromtxt(anno_file, delimiter=',')
img_one = annotations[annotations[:, 0] == json_image_id]
a = {k: v for k, v in self.validation_dataset.coco.imgs.items() if
     f'uav0000086_00000_v/000000{json_image_id}' in v['file_name']}

b = {k: v for k, v in self.validation_dataset.coco.imgs.items() if v['id'] == 0}
# {0: {'file_name': 'uav0000086_00000_v/0000001.jpg', 'height': 756, 'id': 0, 'width': 1344}}
# get all 36 annos for image_id = 0
a = [x for x in self.validation_dataset.coco.anns.values() if x['image_id'] == 0]

z = self.validation_dataset[0]
z = next(iter(self.validation_dataloader))  # uav0000086_00000_v/0000001.jpg
z['annot']

"""

from dataloader import RandomCropOrScale
self = RandomCropOrScale()

dataset = CocoDataset(root_dir='/home/adityakunapuli/data', set_name='val',
                      sub_dir='uav0000268_05773_v',
                      transform=transforms.Compose([Normalizer(), RandomCropOrScale()]), sort=False)


sample = get_sample(random.randint(len(dataset)))
visualize_base_image(sample)

transformed_sample = self.random_crop(sample)

# sampler_val = AspectRatioBasedSampler(dataset, batch_size=1, shuffle=False)
# dataloader = DataLoader(dataset, num_workers=0,
#                         collate_fn=collater,
#                         batch_sampler=sampler_val, pin_memory=True)

