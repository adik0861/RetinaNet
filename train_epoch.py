import shutil
from time import time
import numpy as np
import os
import re
import sys
from concurrent.futures.thread import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from contextlib import redirect_stdout

import colors
import pytz
import torch
import torch.optim as optim
from apex import amp, optimizers
from colors import color
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import model
from dataloader import CocoDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer, RandomCropOrScale

from google.cloud import storage


def cuda_check():
    if torch.cuda.is_available():
        print(colors.black('SUCCESS: CUDA DETECTED!', style='bold', fg='green'))
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


class Initialization:
    def __init__(self, epochs=1, batch_size=1, stride=2, workers=0, root_dir='/home/adityakunapuli/data',
                 verbose=False):
        self.verbose = verbose
        self.batch_size = batch_size
        self.epochs = epochs
        self.stride = stride
        self.workers = workers
        self.root_dir = root_dir
        # Prefix current_datetime to savefiles
        self.current_datetime = str(datetime.now().timestamp()).split('.')[0]
        # Determine gpu/cpu
        self.device = cuda_check()
        # Define subdirs within savefiles
        self.checkpoint_dir, self.saved_images = self.initialize_subdirectories()
        
        self.retinanet, self.optimizer, self.scheduler, self.amp, self.epoch = [None, None, None, None, None]
        
        self.batches = 0
        self.image_count = 0
        self.epoch = 0
        
        self.cols = shutil.get_terminal_size().columns
        self.loss_log = self.checkpoint_dir.joinpath('loss.log')
        self.epoch_loss = []
        self.checkpoint_path = None
        self.training_dataloader, self.training_dataset = [None, None]
        self.validation_dataloader, self.validation_dataset = [None, None]
        # This only here for validation against datasets too small to contain all the necessary labels
        self.categories = [{'id': 0, 'name': 'ignored', 'supercategory': 'other'},
                           {'id': 1, 'name': 'pedestrian', 'supercategory': 'person'},
                           {'id': 2, 'name': 'people', 'supercategory': 'person'},
                           {'id': 3, 'name': 'bicycle', 'supercategory': 'vehicle'},
                           {'id': 4, 'name': 'car', 'supercategory': 'vehicle'},
                           {'id': 5, 'name': 'van', 'supercategory': 'vehicle'},
                           {'id': 6, 'name': 'truck', 'supercategory': 'vehicle'},
                           {'id': 7, 'name': 'tricycle', 'supercategory': 'vehicle'},
                           {'id': 8, 'name': 'awning-tricycle', 'supercategory': 'vehicle'},
                           {'id': 9, 'name': 'bus', 'supercategory': 'vehicle'},
                           {'id': 10, 'name': 'motor', 'supercategory': 'vehicle'},
                           {'id': 11, 'name': 'others', 'supercategory': 'other'}]
    
    @staticmethod
    def initialize_subdirectories():
        save_dir = Path('savefiles')
        checkpoint_dir = save_dir.joinpath('checkpoints')
        checkpoint_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        saved_images = save_dir.joinpath('images')
        saved_images.mkdir(parents=True, exist_ok=True, mode=0o777)
        return checkpoint_dir, saved_images


class RetinaNet(Initialization):
    def __init__(self, **kwargs):
        super(RetinaNet, self).__init__(**kwargs)
    
    # ------------------------------------
    # Initialization
    # ------------------------------------
    def initialize_training(self):
        self.retinanet = model.resnet50(num_classes=12, pretrained=True).to(self.device)
        self.optimizer = optimizers.FusedAdam(params=self.retinanet.parameters(), lr=1e-5)
        self.amp = amp
        self.retinanet, self.optimizer = self.amp.initialize(models=self.retinanet, optimizers=self.optimizer)
        self.load_checkpoint()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=3, verbose=self.verbose)
    
    def initialize_dataloaders(self, **kwargs):
        self.get_training_dataloader()
        self.get_validation_dataloader(**kwargs)
        self.print_batch_statistics()
    
    # ------------------------------------
    #  Dataloaders
    # ------------------------------------
    def get_dataset(self, set_name, sub_dir=None):
        with redirect_stdout(None):
            training_dataset = CocoDataset(root_dir=self.root_dir, set_name=set_name, transform=None, sub_dir=sub_dir)
        [min_w, min_h] = self.get_min_size(training_dataset)
        if set_name == 'val':
            _transforms = transforms.Compose([Normalizer(), Resizer()])
        else:
            _transforms = transforms.Compose([Normalizer(), Augmenter(), RandomCropOrScale(min_w=min_w, min_h=min_h)])
        training_dataset.transform = _transforms
        return training_dataset
    
    def get_training_dataloader(self, set_name='train'):  # this can be used for entire sets
        with redirect_stdout(None):
            self.training_dataset = CocoDataset(root_dir=self.root_dir, set_name=set_name, transform=None)
        [min_w, min_h] = self.get_min_size(self.training_dataset)
        self.training_dataset.transform = transforms.Compose([Normalizer(), Augmenter(), Resizer()])
        # RandomCropOrScale(min_w, min_h)])
        # training_dataset = self.get_dataset(set_name=set_name)
        sampler_train = AspectRatioBasedSampler(self.training_dataset, batch_size=self.batch_size, shuffle=True)
        self.training_dataloader = DataLoader(dataset=self.training_dataset, num_workers=self.workers,
                                              collate_fn=collater, batch_sampler=sampler_train, pin_memory=True)
        self.print_data_statistics(data_loader=self.training_dataloader, set_type='Training')
    
    def get_validation_dataloader(self, sub_dir=None, sort=False, set_name='val'):
        with redirect_stdout(None):
            self.validation_dataset = CocoDataset(root_dir=self.root_dir, set_name=set_name, sub_dir=sub_dir,
                                                  transform=transforms.Compose([Normalizer(), Resizer()]),
                                                  categories=self.categories, sort=sort)
            # validation_dataset = self.get_dataset(set_name=set_name, sub_dir=sub_dir)
        sampler_val = AspectRatioBasedSampler(self.validation_dataset, batch_size=1, shuffle=False)
        self.validation_dataloader = DataLoader(self.validation_dataset, num_workers=self.workers,
                                                collate_fn=collater, batch_sampler=sampler_val, pin_memory=True)
        self.print_data_statistics(data_loader=self.validation_dataloader, set_type='Validation')
    
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
                self.epoch += 1
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
    
    @staticmethod
    def save_checkpoint_storage(checkpoint_path, bucket='visdrone'):
        bucket = storage.Client().bucket(bucket)
        blob = bucket.blob(Path(checkpoint_path).name)
        blob.upload_from_filename(checkpoint_path)
    
    # ------------------------------------
    # Helper Functions
    # ------------------------------------
    @staticmethod
    def get_last_epoch(checkpoint_path):
        _find = re.findall(r'(\d+).pt', checkpoint_path)
        return _find[0] if len(_find) > 0 else None
    
    @staticmethod
    def convert_datetime(epoch_datetime):
        dt = datetime.fromtimestamp(epoch_datetime, pytz.timezone('America/Los_Angeles'))
        return dt.strftime('%A, %d. %B %Y %I:%M%p')
    
    def print_checkpoint_info(self, checkpoint_path, saved):
        checkpoint_path = str(checkpoint_path)
        date_string = f' that was created on {self.convert_datetime(saved)}'
        print(color(f'Found a checkpoint file at: {checkpoint_path}{date_string}', fg='green'))
    
    @staticmethod
    def print_data_statistics(data_loader, set_type):
        print(colors.yellow('{} images = {:,}'.format(set_type, len(data_loader.dataset.image_ids))))
        print(colors.yellow('{} annotations = {:,}'.format(set_type, len(data_loader.dataset.coco.anns))))
    
    def print_batch_statistics(self):
        self.image_count = len(self.training_dataset.image_ids)
        self.batches = len(self.training_dataloader.batch_sampler.groups)
        print(colors.yellow('Batch size = {0:,}'.format(self.batch_size)))
        print(colors.yellow('Number of batches = {0:,}\n'.format(self.batches)))
    
    def progress(self, i):
        _left = len(str(f'{self.image_count:,}')) * 2 + 1
        _percent = 100 * self.batch_size * i / self.image_count
        progress_string = f'{self.batch_size * i:,}/{self.image_count:,}'.rjust(_left) + f'{_percent:1.2f}%'.rjust(8)
        remaining = self.image_count - self.batch_size * i
        return remaining, progress_string
    
    def print_loss(self, epoch, i, cls, box, ttl, rate):
        remaining, progress_string = self.progress(i)
        _mins, _sec = divmod(remaining / rate, 60)
        _hour, _min = divmod(_mins, 60)
        eta = f'{int(_hour):2d}:{int(_min):02d}:{int(_sec):02d}'
        a = 'E: [' + f'{epoch}'.rjust(2) + ']   ' + \
            'I: [' + f'{i:,}/{self.batches:,}'.rjust(len(f'{self.batches:,}') * 2 + 1) + ']   ' + \
            'P: [' + f'{progress_string}'.rjust(len(str(self.image_count)) * 2 + 11) + ']   ' + \
            'R: [' + f'{rate:1.2f} im/s'.rjust(3) + f' {eta}]   ' + \
            'C: [' + f'{cls:1.3f}'.rjust(5) + ']   ' + \
            'B: [' + f'{box:1.3f}'.rjust(5) + ']   ' + \
            'T: [' + f'{ttl:1.3f}'.rjust(5) + ']'
        b = f'{epoch}, {i}, {cls}, {box}, {ttl}'
        return a, b
    
    @staticmethod
    def get_min_size(dataset=None):
        img_sizes = {v['height'] * v['width']: [v['width'], v['height']] for v in dataset.coco.imgs.values()}
        [x, y] = img_sizes[min(img_sizes.keys())]
        return [x, y]
    
    def initialize_weights(self):
        _model = self.retinanet.state_dict()
        for name, param in _model.items():
            if 'weight' in name:
                _model[name] = torch.full_like(param, 0.01)
        self.retinanet.load_state_dict(_model)
    
    # ------------------------------------
    # Train one epoch
    # ------------------------------------
    def train_batch(self, data):
        self.optimizer.zero_grad()
        img_data = data['img'].to(self.device).to(torch.float32)
        img_anno = data['annot'].to(self.device).to(torch.float32)
        classification_loss, regression_loss = self.retinanet([img_data, img_anno])
        del img_data
        del img_anno
        with self.amp.scale_loss(classification_loss + regression_loss, self.optimizer) as scaled_loss:
            scaled_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer), 0.1)
        self.optimizer.step()
        del scaled_loss
        return classification_loss, regression_loss
    
    def train_epoch(self, epoch_num, dataloader=None):
        print(f'Training Epoch {epoch_num}')
        if dataloader is None:
            dataloader = self.training_dataloader
        # run_losses = list()
        # pbar = tqdm(total=self.image_count, file=sys.stdout, ncols=80, unit=' images')

        with open(self.loss_log, 'a') as log_file:
            for i, data in enumerate(dataloader, 1):
                start_time = time()
                classification_loss, regression_loss = self.train_batch(data)
                del data
                _rate = self.batch_size / (time() - start_time)
                ls = self.print_loss(epoch=epoch_num, i=i, cls=float(classification_loss), box=float(regression_loss),
                                     ttl=float(classification_loss + regression_loss), rate=_rate)
                del regression_loss
                del classification_loss
                print(ls[1], file=log_file)
                if i < 25:
                    print(ls[0])
                if i % 100 == 0:
                    msg = self.save_checkpoint(epoch=epoch_num, tmp=True)
                    print(msg)
                    print(ls[0])
            self.epoch_loss.append(classification_loss + regression_loss)
    