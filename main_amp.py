import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from apex import amp, optimizers
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import coco_eval
import model
from dataloader import CocoDataset, \
    collater, Resizer, AspectRatioBasedSampler, Augmenter, Normalizer
import colors


def cuda_check():
    if torch.cuda.is_available():
        print(colors.black('SUCCESS: CUDA DETECTED!', style='bold+underline', fg='green'))
        return torch.device('cuda')
    else:
        print(colors.black('WARNING: CUDA NOT DETECTED!', style='bold+underline', fg='red'))
        return torch.device('cpus')


def print_loss(epoch, iteration, cls_loss, box_loss, run_loss):
    loss_str = ','.join([f'"E": {epoch}',
                         f'"I": {iteration}',
                         f'"C": {cls_loss:1.3f}',
                         f'"R": {box_loss:1.3f}',
                         f'"Avg": {run_loss:1.3f}'])
    a = loss_str.replace('"', '').replace(',', '\t\t')
    b = f'{epoch}, {iteration}, {cls_loss}, {box_loss}, {run_loss}'
    return a, b


def flush_saved_files(save_dir='savefiles/checkpoints/'):
    if not os.path.exists(save_dir):
        return None
    file_cnt = len(os.listdir(save_dir))
    if file_cnt > 0:
        print(colors.color(f'{file_cnt} files detected', style='bold+underline', fg='red'))
        for file in os.listdir(save_dir):
            print(colors.red(f'\tDeleted {file}'))
            os.remove(save_dir + file)


class Initialization:
    def __init__(self, training_images=None, validation_images=None,
                 training_annotations=None, validation_annotations=None,
                 epochs=1, batch_size=1, resize=None, stride=8, workers=10):
        self.batch_size = batch_size
        self.training_annotations = training_annotations
        self.validation_annotations = validation_annotations
        self.epochs = epochs
        self.stride = stride
        self.workers = workers
        self.training_images = training_images
        self.validation_images = validation_images
        self.resize = resize
        # Prefix current_datetime to savefiles
        self.current_datetime = str(datetime.now().timestamp()).split('.')[0]
        # Determine gpu/cpu
        self.device = cuda_check()
        # Define root dir for all saved files
        self.save_dir = Path('savefiles')  # for abs path: Path.cwd().joinpath('savefiles')

        # Define subdirs within savefiles
        self.tb_dir = self.save_dir.joinpath('tensorboard')
        self.tb_train = self.tb_dir.joinpath('train')
        self.tb_val = self.tb_dir.joinpath('val')
        self.checkpoint_dir = self.save_dir.joinpath('checkpoints')
        self.saved_images = self.save_dir.joinpath('images')
        self.loss_log = self.checkpoint_dir.joinpath('loss.log')

        # Make dirs if they don't exist
        self.tb_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        self.saved_images.mkdir(parents=True, exist_ok=True, mode=0o777)

        self.epoch_loss = []
        # Utilized for printing after certain number of iterations
        self.loss = {}
        # Load checkpoint information
        last_checkpoint_path = self.checkpoint_dir.joinpath('last_checkpoint')
        if last_checkpoint_path.exists() and last_checkpoint_path.read_text():
            self.checkpoint_path = last_checkpoint_path.read_text()
        else:
            self.checkpoint_path = None


class RetinaNet(Initialization):
    def __init__(self, **kwargs):
        super(RetinaNet, self).__init__(**kwargs)

        # Load model and optimizer (if checkpoint available, load prev state)
        self.retinanet, self.optimizer, self.scheduler, self.amp = self.initialize_training()

        # Define training datasets/dataloaders
        self.dataset_train = CocoDataset(root_dir='/home/adityakunapuli/data', set_name='train',
                                         transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        sampler_train = AspectRatioBasedSampler(self.dataset_train, batch_size=self.batch_size, drop_last=False)
        self.training_dataloader = DataLoader(self.dataset_train, num_workers=self.workers, collate_fn=collater,
                                              batch_sampler=sampler_train)
        # Define validation datasets/dataloaders
        self.dataset_val = CocoDataset(root_dir='/home/adityakunapuli/data', set_name='val',
                                       transform=transforms.Compose([Normalizer(), Resizer()]))
        sampler_val = AspectRatioBasedSampler(self.dataset_val, batch_size=1, drop_last=False)
        self.validation_dataloader = DataLoader(self.dataset_val, num_workers=self.workers, collate_fn=collater,
                                                batch_sampler=sampler_val)
        # Print out summary statistics of dataset
        self.print_summary_statistics()

    def initialize_training(self):
        # Initialize base model and optimizer
        _model = model.resnet18(num_classes=12, pretrained=True)
        _optimizer = optimizers.FusedAdam(params=_model.parameters(), lr=1e-3)
        _amp = amp
        # AMP Initialize
        _retinanet, _optimizer = _amp.initialize(_model.to(self.device), _optimizer, opt_level='O1', verbosity=1)
        # Load checkpoint state if checkpoint is available
        if self.checkpoint_path:
            print(f'CHECKPOINT FOUND AT: {self.checkpoint_path}')
            checkpoint = torch.load(self.checkpoint_path)
            _amp.load_state_dict(checkpoint['amp'])
            _retinanet.load_state_dict(checkpoint['model'])
            _optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print(colors.color('NO CHECKPOINTS--STARTING FROM SCRATCH', fg='yellow  ', style='bold+underline'))
        #  Define scheduler based off optimizer state
        _scheduler = optim.lr_scheduler.ReduceLROnPlateau(_optimizer, patience=3, verbose=True)
        return _model, _optimizer, _scheduler, _amp

    @staticmethod
    def make_checkpoint(Model, Optimizer, AMP):
        return {'model'    : Model.state_dict(),
                'optimizer': Optimizer.state_dict(),
                'amp'      : AMP.state_dict()}

    def save_checkpoint(self, Model, Optimizer, AMP, Epoch):
        checkpoint = self.make_checkpoint(Model, Optimizer, AMP)
        checkpoint_name = '_'.join([self.current_datetime, 'retinanet', str(Epoch) + '.pt'])
        checkpoint_path = str(self.checkpoint_dir.joinpath(checkpoint_name))
        torch.save(checkpoint, checkpoint_path)
        # note the last checkpoint saved in file 'last_checkpoint'
        last_checkpoint = self.checkpoint_dir.joinpath('last_checkpoint')
        last_checkpoint.write_text(checkpoint_path)

    def train_epoch(self, epoch_num):
        run_losses, cls_losses, box_losses = list(), list(), list()
        self.retinanet.training = True
        pbar = tqdm(total=len(self.dataset_train), file=sys.stdout, ncols=80)
        with open(self.loss_log, 'a') as log_file:
            for i, data in enumerate(self.training_dataloader, 1):
                try:
                    self.optimizer.zero_grad()

                    img_data = data['img'].to(self.device).float()
                    img_anno = data['annot'].to(self.device)
                    cls_loss, box_loss = self.retinanet([img_data, img_anno])

                    with self.amp.scale_loss(cls_loss + box_loss, self.optimizer) as scaled_loss:
                        scaled_loss.backward()

                    torch.nn.utils.clip_grad_norm_(self.amp.master_params(self.optimizer), 0.1)
                    self.optimizer.step()

                    cls_losses += cls_loss
                    box_losses += box_loss
                    run_losses += [float(scaled_loss)]
                    ls = print_loss(epoch=epoch_num, iteration=i, cls_loss=float(cls_loss),
                                    box_loss=float(box_loss), run_loss=float(scaled_loss))
                    if i % 5 == 0:
                        pbar.write(ls[0], nolock=True)
                    print(ls[1], file=log_file)
                    _vars = [img_data, img_anno, cls_loss, box_loss]
                    del _vars[:]
                    pbar.update(n=self.batch_size)
                except Exception as error:
                    print(error)
                    continue
            self.epoch_loss.append(run_losses)
            # coco_eval.evaluate_coco(self.dataset_val, self.retinanet)
            self.save_checkpoint(Model=self.retinanet, Optimizer=self.optimizer, AMP=self.amp, Epoch=epoch_num)
            pbar.close()

    def init_weights(self):
        _model = self.retinanet.state_dict()
        for name, param in _model.items():
            if 'weight' in name:
                _model[name] = torch.full_like(param, 0.01)
        self.retinanet.load_state_dict(_model)

    def print_summary_statistics(self):
        print(colors.yellow('Total number of training images = {0:,}'.format(len(self.dataset_train))))
        print(colors.yellow('Total number of annotations = {0:,}'.format(len(self.dataset_train.coco.anns))))
        print(colors.yellow('Total number of batches = {0:,}'.format(round(len(self.dataset_train) / self.batch_size))))

    def main(self):
        for epoch_num in range(self.epochs):
            self.retinanet.train()
            self.retinanet.freeze_bn()
            # Train over the dataset
            self.train_epoch(epoch_num=epoch_num)
            # Update scheduler
            self.scheduler.step(np.mean(self.epoch_loss))
            # # Validation
            # if True:
            #     coco_eval.evaluate_coco(self.dataset_val, self.retinanet)
            #     continue
        print('Evaluating dataset')
        coco_eval.evaluate_coco(self.dataset_val, self.retinanet)
        # Preform final model evaluation
        print(colors.color('Evaluating Final Model', fg='green'))
        coco_eval.evaluate_coco(self.dataset_val, self.retinanet)
        self.save_checkpoint(Model=self.retinanet, Optimizer=self.optimizer, AMP=self.amp, Epoch='End')


if __name__ == '__main__':
    warnings.simplefilter("ignore")
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('-b', '--batch', required=False, type=int, default=16)
    ap.add_argument('-r', '--resize', required=False, type=int, default=512)
    ap.add_argument('-s', '--stride', required=False, type=int, default=8)
    ap.add_argument('-e', '--epochs', required=False, type=int, default=10)
    ap.add_argument('-w', '--workers', required=False, type=int, default=0)
    ap.add_argument('-x', '--reset', default=False)
    args = ap.parse_args()
    if args.reset is True:
        flush_saved_files()
    # flush_saved_files()

    train_anno = 'annotations/train.json'
    val_anno = 'annotations/val.json'
    train_imgs = 'data/train/images'
    val_imgs = 'data/val/images'

    self = RetinaNet(training_images=train_imgs,
                     validation_images=val_imgs,
                     training_annotations=train_anno,
                     validation_annotations=val_anno,
                     batch_size=args.batch,
                     epochs=args.epochs,
                     workers=args.workers,
                     stride=args.stride,
                     resize=args.resize)

    self.main()
