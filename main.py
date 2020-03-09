from multithreaded_coco_eval import MultithreadedCOCOEval
from train_epoch import *


class TrainModel(RetinaNet):
    def __init__(self, **kwargs):
        super(TrainModel, self).__init__(**kwargs)
        self.initialize_training()
        self.initialize_dataloaders()
    
    def training(self):
        print(colors.color('  Beginning Training Job   ', style='negative'))
        for epoch_num in range(self.epoch, self.epochs):
            self.retinanet.training = True
            self.retinanet.train()
            self.retinanet.freeze_bn()
            self.train_epoch(dataloader=self.training_dataloader, epoch_num=epoch_num)
            self.save_checkpoint(epoch=epoch_num)
            self.delete_temp_checkpoint()
            self.scheduler.step(np.mean(self.epoch_loss))
    
    def validation(self):
        val = MultithreadedCOCOEval(dataset=self.validation_dataset, model=self.retinanet)
        val.__multiprocessor__()
        val.__evaluation__()


if __name__ == '__main__':
    import argparse
    
    defaults = {'reset': False, 'summarize': False, 'epochs': 10, 'workers': 10, 'batch': 8, 'resize': 512, 'stride': 2}
    
    ap = argparse.ArgumentParser()
    ap.add_argument('-x', '--reset', required=False, default=False)
    ap.add_argument('-p', '--summarize', required=False, default=True)
    ap.add_argument('-e', '--epochs', required=False, type=int, default=10)
    ap.add_argument('-w', '--workers', required=False, type=int, default=10)
    # ------------- HP -------------
    ap.add_argument('-b', '--batch', required=False, type=int, default=8)
    ap.add_argument('-s', '--stride', required=False, type=int, default=2)
    
    a = ap.parse_args()
    if a.reset is True:
        flush_saved_files()
    
    self = TrainModel(batch_size=a.batch, epochs=a.epochs, stride=a.stride, summarize=a.summarize)
    print('Model NMS threshold = 0.3')
    self.training()
    self.validation()
