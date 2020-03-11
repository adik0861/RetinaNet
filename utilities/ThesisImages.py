import matplotlib.pyplot  as plt
import numpy as np

from train_epoch import RetinaNet

self = RetinaNet()
self.get_training_dataloader()
coco = self.training_dataset.coco

_labels = {x['id']: x['name'] for x in coco.cats.values()}
labels = [f'{v} ({k})' for k, v in _labels.items()]
cats = [x['category_id'] for x in coco.anns.values()]
a = np.histogram(cats, bins=list(range(13)))
b = a[0] / len(cats)
d = {labels[i]: b[i] for i in range(12)}
X = np.arange(len(d))
plt.bar(X, d.values(), align='center', width=0.5)
plt.xticks(X, d.keys(), rotation='vertical')
ymax = max(d.values()) + 1
plt.ylim(0, ymax)
plt.show()

