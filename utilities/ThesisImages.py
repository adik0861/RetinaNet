from pprint import pprint

import matplotlib.pyplot  as plt
import numpy as np

from train_epoch import RetinaNet

self = RetinaNet()
self.get_training_dataloader()
self.get_validation_dataloader()
coco = self.training_dataset.coco

_labels = {x['id']: x['name'] for x in coco.cats.values()}
labels = [f'{v} ({k})' for k, v in _labels.items()]
cats = [x['category_id'] for x in coco.anns.values()]
a1 = np.histogram(cats, bins=list(range(12)))
a2 = a1[0] / len(cats)
d = {labels[i]: a2[i] for i in range(12)}
a3 = np.arange(len(d))
plt.bar(a3, d.values(), align='center', width=0.45)
plt.xticks(a3, d.keys(), rotation='vertical')
ymax = max(d.values()) + 1
plt.ylim(0, 0.45)
plt.show()

# get per image counts of the different labels
labels2 = {k: f'{k} ({v})' for k, v in _labels.items()}
avg = []
for image in coco.imgToAnns.values():
    a = [x['category_id'] for x in image]
    _count = np.unique(a, return_counts=True)
    _counts = dict(zip(_count[0], _count[1]))
    _totals = sum(_count[1])
    _perImageCounts = {k: 0 for k in range(12)}
    _perImageHist = {k: 0 for k in range(12)}
    for k, v in _counts.items():
        _perImageCounts[k] = v
        _perImageHist[k] = v / _totals
    avg.append(list(_perImageCounts.values()))
print(np.array(avg).mean(axis=0))
avgs = np.array(avg).mean(axis=0).astype('int')
allcats = [labels2[x] for x in list(range(12))]
perImageCounts = dict(zip(allcats, avgs))
pprint(perImageCounts)

"""
{'0 (ignored)': 2,
 '1 (pedestrian)': 9,
 '10 (motor)': 4,
 '11 (others)': 0,
 '2 (people)': 3,
 '3 (bicycle)': 1,
 '4 (car)': 20,
 '5 (van)': 1,
 '6 (truck)': 1,
 '7 (tricycle)': 1,
 '8 (awning-tricycle)': 0,
 '9 (bus)': 0}
"""


# CUSTOM DATASET:
image_list = set()
z = None
for image in coco.imgToAnns.values():
    a = [x['category_id'] for x in image]
    _count = np.unique(a, return_counts=True)
    _counts = dict(zip(_count[0], _count[1]))
    if _counts.get(1, 0) > 10:
        image_list.add(image[0]['image_id'])
