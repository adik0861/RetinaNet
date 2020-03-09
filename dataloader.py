from __future__ import print_function, division
from PIL import Image
import torchvision.transforms.functional as F
import random
from pathlib import Path

import numpy as np
import skimage
import skimage.color
import skimage.io
import skimage.transform
import torch
from pycocotools.coco import COCO
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torchvision.transforms import ToPILImage, ToTensor
import skimage.util


class CocoDataset(Dataset):
    def __init__(self, root_dir, set_name='train', transform=None, sub_dir=None, categories=None, sort=False):
        self.root_dir = root_dir
        self.set_name = set_name
        self.transform = transform
        self.categories = categories
        self.visualization = sort
        file_name = '.'.join([self.set_name, 'json'])

        self.root_dir = Path(self.root_dir).joinpath(self.set_name, 'images')
        if sub_dir is not None:
            self.json_path = self.root_dir.joinpath(sub_dir, file_name)
        else:
            self.json_path = self.root_dir.joinpath(file_name)
        self.coco = COCO(self.json_path)
        self.image_ids = self.coco.getImgIds()
        if sort is True:
            self.image_ids.sort()  # careful sorting as it could possibly return a none type
        self.labels, self.classes, self.coco_labels, self.coco_labels_inverse = dict(), dict(), dict(), dict()
        self.load_classes()

    def load_classes(self):
        # load class names (name -> label)
        if self.categories is None:
            categories = self.coco.loadCats(self.coco.getCatIds())
        else:
            categories = self.categories
        categories.sort(key=lambda x: x['id'])
        for c in categories:
            self.coco_labels[len(self.classes)] = c['id']
            self.coco_labels_inverse[c['id']] = len(self.classes)
            self.classes[c['name']] = len(self.classes)
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        info = self.load_image_info(idx)
        sample = {'img': img, 'annot': annot, 'info': info}
        if self.visualization is True:
            print(info['file_name'])
        if self.transform:
            sample = self.transform(sample)
        return sample

    def naive_getitem(self, idx):
        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        info = self.load_image_info(idx)
        sample = {'img': img, 'annot': annot, 'info': info}
        if self.visualization is True:
            print(info['file_name'])
        return sample

    def load_image_info(self, idx):
        return self.coco.loadImgs(self.image_ids[idx])[0]

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = str(self.root_dir.joinpath(image_info['file_name']))
        img = skimage.io.imread(path)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        return img.astype(np.float32) / 255.0

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))
        # some images appear to miss annotations (like image with id 257034)
        if len(annotations_ids) == 0:
            print(f'Skipping, image_idx = {image_index}')
            return annotations
        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue
            if a['category_id'] == 0:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = self.coco_label_to_label(a['category_id'])
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]
        return annotations

    def coco_label_to_label(self, coco_label):
        return self.coco_labels_inverse[coco_label]

    def label_to_coco_label(self, label):
        return self.coco_labels[label]

    def image_aspect_ratio(self, image_index):
        image = self.coco.loadImgs(self.image_ids[image_index])[0]
        return float(image['width']) / float(image['height'])

    @staticmethod
    def num_classes():
        return 12


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
    info = [s['info'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)
    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)
    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales, 'info': info}


class RandomCropOrScale(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, min_w=1344, min_h=756, p=0.5):
        self.min_w = min_w
        self.min_h = min_h
        self.min_side = min(min_w, min_h)
        self.max_side = max(min_w, min_h)
        self.p = p

    def __call__(self, sample):
        self.random_crop(sample)

    def random_crop(self, sample):
        # img, annots, info = sample['img'], sample['annot'], sample['info']
        img, _annots = sample['img'], sample['annot']
        info = sample['info']
        rows, cols, cns = img.shape
        input_image_width = max(rows, cols)
        input_image_height = min(rows, cols)
        # print(sample.keys())
        # w, h = image.size
        if input_image_width == self.min_w and input_image_height == self.min_h:
            return {'img': torch.from_numpy(img), 'annot': torch.from_numpy(annots), 'scale': 1, 'info': info}
        top_edge = random.randint(0, input_image_height - self.min_h)
        left_edge = random.randint(0, input_image_width - self.min_w)
        right_edge = left_edge + self.min_w
        bottom_edge = top_edge + self.min_h

        bottom_crop = abs(bottom_edge - input_image_height)
        right_crop = abs(right_edge - input_image_width)

        _crop = ((top_edge, bottom_crop), (left_edge, right_crop), (0, 0))
        image = skimage.util.crop(ar=img, crop_width=_crop, copy=True)
        # visualize_tensor(image)

        # annotation begins before left edge
        # x1, y1, x2, y2, lbl = annots
        annots = _annots.copy()
        x1 = annots[:, 0]
        x1[x1 < left_edge] = 0
        annots[:, 0] = x1
        # annotation begins before top edge
        y1 = annots[:, 1]
        y1[y1 < top_edge] = 0
        annots[:, 1] = y1
        # annotation extends beyond right edge
        x2 = annots[:, 2]
        x2[x2 > right_edge] = self.min_w
        annots[:, 0] = x2
        # annotation extends beyond bottom edge
        y2 = annots[:, 3]
        y2[y2 > bottom_edge] = self.min_h
        annots[:, 3] = y2
        # return {'img': torch.from_numpy(image), 'annot': torch.from_numpy(annots), 'scale': 1, 'info': info}
        return {'img': torch.from_numpy(image.copy()), 'annot': torch.from_numpy(annots), 'scale': 1, 'info': info}

    def rescale(self, sample):
        image, annots = sample['img'], sample['annot']
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side
        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32
        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        annots[:, :4] *= scale
        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Resizer(object):
    # """Convert ndarrays in sample to Tensors."""
    #
    # def __init__(self, max_side=1024, min_side=608):
    #     self.min_side = min_side
    #     self.max_side = max_side

    def __call__(self, sample, max_side=1344, min_side=756):
        image, annots = sample['img'], sample['annot']
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)
        if largest_side * scale > max_side:
            scale = max_side / largest_side
        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32
        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        annots[:, :4] *= scale
        return {'img' : torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale,
                'info': sample['info']}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots, 'info': sample['info']}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots, 'info': sample['info']}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean is None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std is None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last=False, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.groups = self.group_images()

    def __iter__(self):
        if self.shuffle is True:
            random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        order = list(range(len(self.data_source)))
        if self.shuffle is True:
            random.shuffle(order)
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
                range(0, len(order), self.batch_size)]


# class AspectRatioBasedSampler(Sampler):
#     def __init__(self, data_source, batch_size, drop_last=False, shuffle=True):
#         super().__init__(data_source)
#         self.data_source = data_source
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.shuffle = shuffle
#         self.groups = self.group_images()
#
#     def __iter__(self):
#         if self.shuffle is True:
#             random.shuffle(self.groups)
#         for group in self.groups:
#             yield group
#
#     def __len__(self):
#         if self.drop_last:
#             return len(self.data_source) // self.batch_size
#         else:
#             return (len(self.data_source) + self.batch_size - 1) // self.batch_size
#
#     def group_images(self):
#         order = list(range(len(self.data_source)))
#         if self.shuffle is True:
#             random.shuffle(order)
#         return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
#                 range(0, len(order), self.batch_size)]

        # order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))
        # # divide into groups, one group = one batch
        # return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in
        #         range(0, len(order), self.batch_size)]

#
# if __name__ == '__main__':
#     from torchvision import transforms
#     from torch.utils.data import DataLoader
#
#     training_dataset = CocoDataset(root_dir='/home/adityakunapuli/data', set_name='train',
#                                    transform=transforms.Compose([Normalizer(), Augmenter(), RandomCropOrScale()]))
#
#     sampler_train = AspectRatioBasedSampler(training_dataset, batch_size=2, shuffle=False)
#     training_dataloader = DataLoader(dataset=training_dataset, num_workers=0, collate_fn=collater,
#                                      batch_sampler=sampler_train)
#
#     next(iter(training_dataloader))
#
# # transform=transforms.Compose([RandomCropOrScale(),
# #                               Normalizer(), Augmenter(),
# #                               ]))
