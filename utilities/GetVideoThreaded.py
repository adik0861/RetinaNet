from concurrent.futures import ThreadPoolExecutor

import cv2
from tqdm import tqdm

from dataloader import *
from train_epoch import *
from train_epoch import RetinaNet


class GetVideo(RetinaNet):
    def __init__(self, sub_dir, **kwargs):
        super().__init__(**kwargs)
        if 'utilities' in os.getcwd():
            os.chdir('..')
            self.images_dir = Path.cwd().joinpath('savefiles/images')
        
        if self.validation_dataset is None:
            self.initialize_training()
            self.retinanet.to(self.device)
            self.dtype = self.get_dtype(self.retinanet)
            self.get_validation_dataloader(sub_dir=sub_dir, sort=True)
            self.bbox_colors = {'person': (51, 234, 48), 'other': (255, 0, 255), 'vehicle': (241, 196, 15)}
        
        self.frames_dict = dict()
        self.video_path = ''
        self.retinanet.eval()
        
        print(colors.color(f'Processing sub-directory {sub_dir}...', fg='magenta'))
        print('Required codecs:' + colors.color('\tsudo apt-get install ffmpeg x264 libx264-dev', fg='cyan'))
        
        # if self.training_dataset is None:
        #     self.get_training_dataloader()
        #     bbox_colors = {'vehicle': (241, 196, 15), 'person': (51, 234, 48), 'other': (255, 0, 255)}
        #     self.bbox_colors = {v['supercategory']: bbox_colors[v['supercategory']] for k, v in
        #                         self.training_dataset.coco.cats.items()}
        self.supercategories = {x['id']: x['supercategory'] for x in self.validation_dataset.categories}
        self.unique_supercats = list(set(self.supercategories.values()))
        
        self.video_path = os.path.join(self.root_dir, '.'.join([sub_dir, 'avi']))
        self.pbar = tqdm(total=len(self.validation_dataset.image_ids), position=0, leave=True, desc='Processing Frames')
        self.transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])
    
    def start(self):
        self.__multiprocessor__()
        self.make_movie()
    
    def __multiprocessor__(self):
        cores = os.cpu_count()
        image_ids = self.validation_dataset.image_ids
        with ThreadPoolExecutor(max_workers=cores * 2) as executor:
            executor.map(self.__partial__, image_ids, timeout=30)
        self.pbar.close()
    
    def __partial__(self, index):
        with torch.no_grad():
            image = self.validation_dataset[index]['img']
            plot_image = image.clone()
            img_data = image.permute(2, 0, 1).to(self.device, dtype=self.dtype).unsqueeze(dim=0)
            scores, labels, boxes = self.retinanet(img_data)
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()
            plot_image = cv2.normalize(np.array(plot_image), None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            idxs = np.where(scores.cpu() > 0.5)
            for j in range(idxs[0].shape[0]):
                bbox = boxes[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                predicted_idx = int(labels[idxs[0][j]])
                self.draw_caption(image=plot_image, box=(x1, y1, x2, y2), label_idx=predicted_idx)
        cv2.rectangle(img=plot_image, pt1=(0, 0), pt2=(160, 120), color=(255, 255, 255), thickness=-1)
        for i in range(len(self.unique_supercats)):
            _label = self.unique_supercats[i]
            cv2.putText(img=plot_image, text=_label.upper(), org=(10, 35 * (i + 1)), fontFace=cv2.FONT_HERSHEY_PLAIN,
                        fontScale=2.2,
                        color=self.bbox_colors[_label], thickness=3, lineType=cv2.LINE_AA)
        self.frames_dict[index] = plot_image
        self.pbar.update(n=1)
        del image
        del scores
        del labels
        del boxes
    
    def draw_caption(self, image, box, label_idx):
        b = np.array(box).astype(int)
        label_name = self.supercategories[label_idx]
        caption = label_name.upper()
        cv2.rectangle(img=image, pt1=(b[0], b[1]), pt2=(b[2], b[3]), color=self.bbox_colors[label_name], thickness=1)
    
 
    def make_movie(self, codec='XVID'):
        frame_idxs = sorted(self.frames_dict.keys())
        if len(frame_idxs) == 0:
            raise Exception('Empty frame list.')
        if os.path.exists(self.video_path):
            os.remove(self.video_path)
        height, width, layers = np.shape(self.frames_dict[0])
        video = cv2.VideoWriter(filename=self.video_path,
                                fourcc=cv2.VideoWriter_fourcc(*codec),
                                fps=15,
                                frameSize=(width, height))
        for idx in frame_idxs:
            frame = cv2.cvtColor(self.frames_dict[idx], cv2.COLOR_RGB2BGR)
            video.write(frame)
        cv2.destroyAllWindows()
        video.release()
        if os.path.exists(self.video_path):
            print(colors.color(f'[SUCCESS] Video Path: {self.video_path}', fg='green'))
        else:
            print(colors.color(f'[FAILED] Video Path: {self.video_path}', fg='red'))
    
    # def get_bbox_colors(self):
    #     supercats = {x['id']: x['supercategory'] for x in self.validation_dataset.categories}
    #     unique_supercats = list(set(supercats.values()))
    #     color_list = get_colors(len(unique_supercats))
    #     return {x['id']: color_list[unique_supercats.index(x['supercategory'])] for x
    #             in self.validation_dataset.categories}
    
    @staticmethod
    def get_dtype(_model):
        model_weights_dtype = [v.dtype for k, v in _model.state_dict().items() if 'weight' in k]
        model_weights_dtype = set(model_weights_dtype)
        if len(model_weights_dtype) != 1:
            return Exception('Too many dtypes returned from model weights.')
        return model_weights_dtype.pop()
    
    def output_to_file(self, idx, image_data):
        img_name = str(idx) + '.jpg'
        img_path = str(self.saved_images.joinpath(img_name))
        cv2.imwrite(filename=img_path, img=image_data)


if __name__ == '__main__':
    image_dir = '/home/adityakunapuli/data/val/images'
    subdirs = [x for x in os.listdir(image_dir) if os.path.isdir(f'{image_dir}/{x}')]
    for subdir in subdirs:
        GetVideo(sub_dir=subdir).start()
