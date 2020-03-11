import matplotlib.patches as patches
import matplotlib.pyplot as plt
from numpy import random
from torchvision.transforms import transforms

from dataloader import *

np.set_printoptions(suppress=True)


def unnormalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                      std=[0.229, 0.224, 0.225])])


class GetSample:
    def __init__(self, _root_dir='/home/adityakunapuli/data'):
        self.dataset = CocoDataset(root_dir=_root_dir, set_name='val',
                                   sub_dir='uav0000268_05773_v',
                                   transform=transforms.Compose([Normalizer(), Resizer()]),
                                   sort=False)
        self.sample = self.get_sample()
    
    @staticmethod
    def visualize_base_image(tensor):
        im = tensor
        fig = plt.figure(figsize=(16, 9))
        ax = fig.add_subplot(111)
        ax.imshow(im, origin='upper', interpolation='none')
        plt.show()
    
    def visualize_sample(self, sample):
        if isinstance(sample, dict):
            img = sample['img']
            annot = sample['annot']
            fig = plt.figure(figsize=(16, 9))
            ax = fig.add_subplot(111)
            ax.imshow(img, origin='upper', interpolation='none')
            for i in range(annot.shape[0]):
                x1, y1 = annot[i, :2]
                x2, y2 = annot[i, 2:4]
                w = x2 - x1
                h = y2 - y1
                rect = patches.Rectangle(xy=(x1, y1), width=w, height=h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f'{int(x1), int(y1)}')
            plt.show()
        else:
            self.visualize_base_image(tensor=sample)
    
    def get_sample(self, idx_or_tensor=None):
        if idx_or_tensor is None:
            idx_or_tensor = np.random.randint(len(self.dataset))
        return self.dataset.naive_getitem(idx_or_tensor)
    
    def get_crop(self, sample, min_w=1344, min_h=756):
        input_image_width = max(sample['img'].shape[:0])
        input_image_height = min(sample['img'].shape[:1])
        if input_image_width == min_w and input_image_height == min_h:
            return self.sample
        # Generate random initial point for crop
        top_edge = random.randint(0, input_image_height - min_h)
        left_edge = random.randint(0, input_image_width - min_w)
        # TODO: regen new points if no annotations in crop
        right_edge = left_edge + min_w
        bottom_edge = top_edge + min_h
        
        bottom_crop = abs(bottom_edge - input_image_height)
        right_crop = abs(right_edge - input_image_width)
        
        _crop = ((top_edge, bottom_crop), (left_edge, right_crop), (0, 0))
        image = skimage.util.crop(ar=sample['img'], crop_width=_crop, copy=True)
        crop = {'left_edge': left_edge, 'top_edge': top_edge, 'right_edge': right_edge, 'bottom_edge': bottom_edge}
        return {'img': torch.from_numpy(image), 'crop': crop}
    
