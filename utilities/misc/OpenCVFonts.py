import cv2
import matplotlib.pyplot as plt
import numpy as np
from utilities.GetSample import *

self = GetSample()
sample = self.sample

caption = 'person'
caption = caption.upper()
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
    b = np.array((x1, y1, x2, y2)).astype(int)
    cv2.putText(img=img, text=caption, org=(20, 350), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
    cv2.putText(img=img, text=caption, org=(20, 400), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1,
                color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
    cv2.rectangle(img=img, pt1=(10, 10), pt2=(20, 450), color=(0, 0, 255), thickness=1)
plt.show()

# Define 500 (height) x 500 (width) 3 layers (BGR)
size = (1000, 600, 3)
# Empty array
# img = cv2.PutText(image, Text, Org, FontFace, FontScale, Color[, Thickness[, LineType[, BottomLeftOrigin]]] )
image = np.ones(size, dtype=np.uint8)
cv2.rectangle(image, (0, 0), (size[0], size[1]), (255, 255, 255), -1)

# Image, text, position (lower left), font, scale, color, line thickness, type
cv2.putText(image, 'FONT_HERSHEY_SIMPLEX', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 200), 2, cv2.LINE_AA)
cv2.putText(image, 'FONT_HERSHEY_PLAIN', (20, 100), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 200, 0), 2, cv2.LINE_AA)
cv2.putText(image, 'FONT_HERSHEY_DUPLEX', (20, 150), cv2.FONT_HERSHEY_DUPLEX, 1.2, (200, 0, 0), 2, cv2.LINE_AA)
cv2.putText(image, 'FONT_HERSHEY_COMPLEX', (20, 200), cv2.FONT_HERSHEY_COMPLEX, 1.2, (0, 100, 100), 2, cv2.LINE_AA)
cv2.putText(image, 'FONT_HERSHEY_TRIPLEX', (20, 250), cv2.FONT_HERSHEY_TRIPLEX, 1.2, (100, 100, 0), 2, cv2.LINE_AA)
cv2.putText(image, 'FONT_HERSHEY_COMPLEX_SMALL', (20, 300), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.2, (100, 0, 100), 2,
            cv2.LINE_AA)


# cv2.imwrite('Result.Png', img)

fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111)
image = np.ones(size, dtype=np.uint8)
ax.imshow(image, origin='upper', interpolation='none')
plt.show()
