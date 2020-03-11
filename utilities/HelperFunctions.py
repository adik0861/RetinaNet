import os
import re
import shutil


def banner(text):
    from colors import color
    cols = shutil.get_terminal_size()[0]
    _pad = int((cols - len(text)) / 2)
    pad = _pad * ' '
    line = cols * '_'
    text = ''.join([pad, text, pad])
    text = '\n'.join([line, text])
    message = color(f'{text}', bg='yellow', style='bold')
    print(message)
# banner('Training Epoch 4')


def get_dtype(_model):
    model_weights_dtype = [v.dtype for k, v in _model.state_dict().items() if 'weight' in k]
    model_weights_dtype = set(model_weights_dtype)
    if len(model_weights_dtype) != 1:
        return Exception('Too many dtypes returned from model weights.')
    return model_weights_dtype.pop()


def visualize_image(image):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)
    ax.imshow(image, origin='upper', interpolation='none')
    plt.show()


def get_colors(n):
    import random
    ret = dict()
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        ret[i] = (int(r) % 256, int(g) % 256, int(b) % 256)
    return ret




def img_to_tensor(image_path):
    from torchvision import transforms
    from PIL.Image import fromarray
    from skimage import io, color
    transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])
    img = io.imread(image_path)
    if len(img.shape) == 2:
        img = color.gray2rgb(img)
    np_img = fromarray(img)
    image_tensor = transforms(np_img)
    new_shape = [1] + list(image_tensor.shape)
    return image_tensor.view(new_shape)


def sorted_images(images_dir):
    images_dir = str(images_dir)
    images = [x for x in os.listdir(images_dir) if x.endswith('.jpg')]
    images.sort(key=lambda x: int(re.findall(r'(\d+).jpg', x)[0]))
    return [os.path.join(images_dir, x) for x in images]


def process_image(img):
    import numpy as np
    import cv2
    img[img < 0] = 0
    img[img > 255] = 255
    img = np.transpose(img, (1, 2, 0))
    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return img
