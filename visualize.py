import cv2
from colors import yellow
from skimage import io, color
from PIL import Image
from dataloader import *
from train_epoch import *


class Visualizations(RetinaNet):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.initialize_training()
        self.images_dir = self.get_img_dir()
        self.unnormalize = UnNormalizer()
        self.frames = list()
        self.video_path = ''

        self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

    def validation(self, sub_dir):
        self.retinanet.eval()
        print(yellow(f'Processing sub-directory {sub_dir}...', style='bold'))
        self.get_validation_dataloader(sub_dir=sub_dir, sort=True)
        pbar = tqdm(total=len(self.validation_dataset), file=sys.stdout, ncols=80, unit=' images')
        for idx, data in enumerate(self.validation_dataloader):
            with torch.no_grad():
                scores, classification, transformed_anchors = self.retinanet(data['img'].to(self.device).float())
                img = np.array(255 * self.unnormalize(data['img'][0, :, :, :])).copy()
                del data
                img = self.process_image(img)
                idxs = np.where(scores.cpu() > 0.5)
                del scores
                for j in range(idxs[0].shape[0]):
                    predicted_idx = int(classification[idxs[0][j]])
                    bbox = transformed_anchors[idxs[0][j], :]
                    x1 = int(bbox[0])
                    y1 = int(bbox[1])
                    x2 = int(bbox[2])
                    y2 = int(bbox[3])
                    if predicted_idx in self.validation_dataset.labels:
                        label_name = self.validation_dataset.labels[predicted_idx]
                        self.draw_caption(img, (x1, y1, x2, y2), label_name)
                        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                del classification
                del transformed_anchors
            self.frames.append(img)
            self.output_to_file(idx, img)
            pbar.update()
        pbar.close()
        self.video_path = os.path.join(self.root_dir, '.'.join([sub_dir, 'avi']))

    @staticmethod
    def output_to_file(idx, image_data):
        img_name = str(idx) + '.jpg'
        img_path = str(self.saved_images.joinpath(img_name))
        cv2.imwrite(filename=img_path, img=image_data)

    @staticmethod
    def process_image(img):
        img[img < 0] = 0
        img[img > 255] = 255
        img = np.transpose(img, (1, 2, 0))
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
        return img

    @staticmethod
    def sorted_images(images_dir):
        images_dir = str(images_dir)
        images = [x for x in os.listdir(images_dir) if x.endswith('.jpg')]
        images.sort(key=lambda x: int(re.findall(r'(\d+).jpg', x)[0]))
        return [os.path.join(images_dir, x) for x in images]

    @staticmethod
    def get_img_dir():
        images_dir = Path.cwd().joinpath('savefiles/images')
        images_dir.mkdir(parents=True, exist_ok=True, mode=0o777)
        return images_dir

    @staticmethod
    def draw_caption(image, box, caption):
        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    def img_to_tensor(self, image_path):
        img = skimage.io.imread(image_path)
        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)
        np_img = Image.fromarray(img)
        image_tensor = self.transforms(np_img)
        new_shape = [1] + list(image_tensor.shape)
        return image_tensor.view(new_shape)

    # sudo apt-get install ffmpeg x264 libx264-dev
    def make_movie(self, codec='XVID'):
        height, width, layers = np.shape(self.frames[0])
        video = cv2.VideoWriter(filename=self.video_path,
                                fourcc=cv2.VideoWriter_fourcc(*codec),
                                fps=20,
                                frameSize=(width, height))
        for frame in self.frames:
            video.write(frame)
        cv2.destroyAllWindows()
        video.release()
        if os.path.exists(self.video_path):
            print(color(f'[SUCCESS] Video Path: {self.video_path}', fg='green'))
            self.frames = list()
        else:
            print(color(f'[FAILED] Video Path: {self.video_path}', fg='red'))


if __name__ == '__main__':
    self = Visualizations()
    directories = [x for x in os.listdir('/home/adityakunapuli/data/val/images')
                   if not x.endswith('.json') and not x.endswith('.avi')]
    for sub_dir in directories:
        if 'uav0000086_00000_v' in sub_dir:
            self.validation(sub_dir=sub_dir)
            self.make_movie()
            break
