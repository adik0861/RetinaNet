import os
import re
import shutil
from pathlib import Path
from time import time

import colors
import torch
from colors import color
# from google.cloud import storage
from google.cloud import storage


def activate_cpu_cores():
    """
    https://github.com/pytorch/pytorch/issues/8126#issuecomment-533844168
    taskset -p XXX PID is a linux command, that can change the CPU affinity of certain process (assgined by the PID).
    XXX should be an Hexadecimal integer and it determines which CPUs to use for the process.
    For instance, 0xff is 1111 1111 in binary, which will activate the first 8 CPUs.
    If you want to use the CPU 2, 4, 6 and 8, you can set XXX to 0x55(01010101). 0xffffffff
    for all the fist 32 CPUs of course.
    """
    os.system("taskset -p 0xff %d" % os.getpid())


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


def cuda_check():
    if torch.cuda.is_available():
        print(colors.black('SUCCESS: CUDA DETECTED!', fg='green'))
        return torch.device('cuda')
    else:
        print(colors.black('WARNING: CUDA NOT DETECTED!', style='bold', fg='red'))
        return torch.device('cpus')


def flush_saved_files(save_dir='savefiles/checkpoints/'):
    if not os.path.exists(save_dir):
        return None
    file_cnt = len(os.listdir(save_dir))
    if file_cnt > 0:
        print(colors.color(f'{file_cnt} files detected', style='bold', fg='red'))
        for file in os.listdir(save_dir):
            print(colors.red(f'\tDeleted {file}'))
            os.remove(save_dir + file)
            return True


# ------------------------------------
# Load Checkpoint
# ------------------------------------
def _load_checkpoint(self, last_checkpoint_path):
    try:
        checkpoint = torch.load(last_checkpoint_path)
        self.amp.load_state_dict(checkpoint['amp'])
        self.retinanet.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epoch = int(checkpoint['epoch']) if checkpoint['epoch'] != 'End' else 0
    except Exception as error:
        raise error
    else:
        self.print_checkpoint_info(checkpoint_path=last_checkpoint_path, saved=checkpoint['saved'])


def load_checkpoint(self):
    if self.checkpoint_dir.joinpath('_tmp.pt').exists():
        self._load_checkpoint(last_checkpoint_path=self.checkpoint_dir.joinpath('_tmp.pt'))
    elif self.checkpoint_dir.joinpath('last_checkpoint').exists():
        _last_checkpoint_path = self.checkpoint_dir.joinpath('last_checkpoint')
        try:
            last_checkpoint_path = Path(_last_checkpoint_path).read_text()
            self._load_checkpoint(last_checkpoint_path=last_checkpoint_path)
        except Exception as error:
            raise error
        else:
            self.epoch = 1
            print(color(f'Starting off at epoch = {self.epoch}'))
    else:
        print(color('No checkpoints found, starting from scratch', fg='yellow'))


# ------------------------------------
# Save Checkpoint
# ------------------------------------
def get_save_name(self, epoch):
    return '_'.join([self.current_datetime, 'retinanet', str(epoch) + '.pt'])


def get_save_path(self, epoch, tmp=False):
    checkpoint_name = self.get_save_name(epoch)
    if tmp is True:
        checkpoint_name = '_tmp.pt'
    return str(self.checkpoint_dir.joinpath(checkpoint_name))


def save_checkpoint(self, epoch, tmp=False):
    completed = True if epoch == self.epochs - 1 else False
    checkpoint = {'model'    : self.retinanet.state_dict(),
                  'optimizer': self.optimizer.state_dict(),
                  'amp'      : self.amp.state_dict(),
                  'epoch'    : epoch,
                  'completed': str(completed),
                  'saved'    : time(), }
    checkpoint_path = self.get_save_path(epoch, tmp=tmp)
    torch.save(checkpoint, checkpoint_path)
    last_checkpoint = self.checkpoint_dir.joinpath('last_checkpoint')  # noting the final checkpoint
    last_checkpoint.write_text(checkpoint_path)
    if os.path.exists(checkpoint_path):
        self.save_checkpoint_storage(checkpoint_path)
        return color(f'Succesfully saved checkpoint to:\t{checkpoint_path}', fg='green')
    else:
        raise Exception(color(f'Unable to save checkpoint at:\n{checkpoint_path}', fg='red'))


def delete_temp_checkpoint(self):
    if self.checkpoint_dir.joinpath('_tmp.pt').exists():
        try:
            os.remove(self.checkpoint_dir.joinpath('_tmp.pt'))
        except Exception as e:
            print(e)


def save_checkpoint_storage(checkpoint_path, bucket='visdrone'):
    bucket = storage.Client().bucket(bucket)
    blob = bucket.blob(Path(checkpoint_path).name)
    blob.upload_from_filename(checkpoint_path)


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
