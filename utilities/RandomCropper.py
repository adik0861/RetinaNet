from numpy import random

from dataloader import *
from utilities.GetSample import GetSample

np.set_printoptions(suppress=True)


class RandomCrop(GetSample):
    """
    This class will chose at random whether to (random) crop an image or rescale an image.
    The cropping method will loop through possible crops to ensure that the resulting cropped
    image contains annotations.  Note that by cropping, we may (usually) loose some annotations,
    so we're loosing information essentially.
    
    # TODO: Implement a RandomScaleAndCrop function
    """
    def __init__(self, sample=None, min_w=1344, min_h=756, p=0.5):
        super(RandomCrop, self).__init__()
        self.p = p
        self.min_w = min_w
        self.min_h = min_h
        self.min_side = min(min_w, min_h)
        self.max_side = max(min_w, min_h)
        
        if sample is not None:
            self.sample = sample
        self.img = self.sample['img']
        self.annot = self.sample['annot']
        self.cropped_sample = self.random_crop()
        
        self.visualize_sample(self.sample)
        self.visualize_sample(self.cropped_sample)
        
    def random_crop_or_rescale(self):
        if np.random.rand() > self.p:
            return self.random_crop()
        else:
            return self.rescale()
            
    def random_crop(self):
        input_img_w = max(self.img.shape[:2])
        input_img_h = min(self.img.shape[:2])
        if input_img_w == self.min_w and input_img_h == self.min_h:
            return self.sample
        while True:
            crops = self.get_cropped_indices(input_img_w, input_img_h)
            cropped_annos = self.crop_annotations(crops)
            if cropped_annos is not False:
                cropped_img = self.cropped_image(crops, input_img_w, input_img_h)
                return {'img': torch.from_numpy(cropped_img),
                        'annot': torch.from_numpy(cropped_annos),
                        'info' : self.annot['info'],
                        'scale': 1}
            else:
                continue
    
    def get_cropped_indices(self, input_img_w, input_img_h):
        # Generate random initial point for crop
        left_edge = random.randint(0, input_img_w - self.min_w)
        top_edge = random.randint(0, input_img_h - self.min_h)
        right_edge = left_edge + self.min_w
        bottom_edge = top_edge + self.min_h
        return {'left_edge': left_edge, 'top_edge': top_edge, 'right_edge': right_edge, 'bottom_edge': bottom_edge}
    
    def cropped_image(self, crops, input_img_w, input_img_h):
        left_edge, top_edge, right_edge, bottom_edge = list(crops.values())
        bottom_crop = abs(bottom_edge - input_img_h)
        right_crop = abs(right_edge - input_img_w)
        _crop = ((top_edge, bottom_crop), (left_edge, right_crop), (0, 0))
        return skimage.util.crop(ar=self.img, crop_width=_crop, copy=True)
    
    def crop_annotations(self, crops):
        left_edge, top_edge, right_edge, bottom_edge = list(crops.values())
        annots = self.annot.copy()
        x1, y1, x2, y2, lbl = np.hsplit(annots, 5)
        p1 = np.array(list(zip(x1[:, 0], y1[:, 0])))
        p2 = np.array(list(zip(x2[:, 0], y2[:, 0])))
        cropped_img_p1 = np.array([left_edge, top_edge])
        cropped_img_p2 = np.array([right_edge, bottom_edge])
        p1_any = np.all((p1 > cropped_img_p1) & (p1 < cropped_img_p2), axis=1)
        p2_any = np.all((p2 > cropped_img_p1) & (p2 < cropped_img_p2), axis=1)
        rows_to_crop = p1_any | p2_any
        crop = np.array([left_edge, top_edge, left_edge, top_edge, 0])
        # Shift annotations up and to the left, this takes care of any annotations
        # who's points exceed the left and top coordinaes of the cropped image
        annots = annots[rows_to_crop] - crop
        # And now set any points that lay outside the bottom and right
        # of the cropped coordinates to the edge of cropped coords.
        annots[annots[:, 2] > self.min_w, 2] = self.min_w
        annots[annots[:, 3] > self.min_h, 3] = self.min_h
        if annots.size == 0:
            print('Cropped image contains no annotations.  Skipping...')
            return False
        else:
            return annots

    def rescale(self):
        image = self.img.copy()
        annots = self.annot.copy()
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
        return {'img': torch.from_numpy(new_image),
                'annot': torch.from_numpy(annots),
                'info' : self.annot['info'],
                'scale': scale}


if __name__ == '__main__':
    self = RandomCrop()
