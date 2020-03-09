from colors import color, green, yellow, red
import os
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image


# class MergeVisDroneCSVs:
#     def __init__(self, **kwargs):
#         self.column_names = ['x', 'y', 'w', 'h', 'score', 'category_id', 'truncation', 'occlusion']
#         self.csv_directory = kwargs.get('csv_directory',
#                                         os.path.join(os.path.expanduser('~'), 'data/{dataset}/annotations'))
#         self.save_path = kwargs.get('save_path',
#                                     os.path.join(os.path.expanduser('~'), 'data/{dataset}/{dataset}.txt'))
#
#     def get_dirs(self, d):
#         return self.csv_directory.format(dataset=d), self.save_path.format(dataset=d)
#
#     def combine_csv(self, d):
#         csv_dir, save_path = self.get_dirs(d=d)
#         annotations = [x for x in os.listdir(csv_dir) if x.endswith('.txt') or x.endswith('.csv')]
#         _df_list = []
#         for _csv in tqdm(annotations, desc=d):
#             _df = pd.read_csv(os.path.join(csv_dir, _csv), names=self.column_names)
#             _df.insert(0, 'file_name', _csv.replace('.txt', '.jpg'))
#             _df_list.append(_df)
#         df = pd.concat(_df_list, ignore_index=True)
#         df.to_csv(save_path, index=False)
#         return save_path


class CSVtoJSON:
    def __init__(self, csv_list, dataset, Head=None, image_dir=None, inference=False):
        self.csv_list = csv_list
        self.dataset = dataset
        self.head = Head
        self.image_dir = image_dir  # .format(dataset=Dataset)

        classes = [['person', 1, 'pedestrian'], ['person', 2, 'people'],
                   ['vehicle', 3, 'bicycle'], ['vehicle', 4, 'car'],
                   ['vehicle', 5, 'van'], ['vehicle', 6, 'truck'],
                   ['vehicle', 7, 'tricycle'], ['vehicle', 8, 'awning-tricycle'],
                   ['vehicle', 9, 'bus'], ['vehicle', 10, 'motor'],
                   ['other', 11, 'others'], ['other', 0, 'ignored']]
        self.classes = pd.DataFrame(classes, columns=['supercategory', 'category_id', 'category_name'])

        self.csv_header = ['file_name', 'target_id', 'x', 'y', 'w', 'h',
                           'score', 'category_id', 'truncation', 'occlusion']
        self.dtypes = {'file_name': str, 'target_id': int, 'x': int, 'y': int, 'w': int, 'h': int,
                       'score'    : int, 'category_id': int, 'truncation': int}
        self.inference = inference

    def coco(self):
        _df = self.csv_to_df()
        _dc = self.df_to_dict(Df=_df)
        _js = self.dict_to_json(Dict=_dc, Dataset=self.dataset)
        return _js

    def csv_to_df(self):
        df_list = list()
        for csv_path in tqdm(self.csv_list, desc=f'[{self.dataset}] csv → df'):
            df = pd.read_csv(open(csv_path, 'r'), names=self.csv_header, dtype=self.dtypes, index_col=False)
            # Only grab samples where score=1 for eval
            df = df.loc[df.score == 1] if self.inference is True else df
            # Fluff up the filenames
            df['file_name'] = df.file_name.apply(lambda x: '.'.join([x.rjust(7, '0'), 'jpg']))
            # Define the full path to the image
            df['full_path'] = df.file_name.apply(lambda x: '/'.join([self.get_image_dir(csv_path), x]))
            # Prefix the filenames with directory name
            df['file_name'] = df.file_name.apply(lambda x: os.path.join(csv_path.stem, x))
            # extract image size
            _img_dims = df.full_path.apply(lambda x: self.get_img_size(x)).tolist()
            df[['image_width', 'image_height']] = pd.DataFrame(_img_dims, index=df.index)
            # bbox column with nested list
            df['bbox'] = pd.Series(df.loc[:, ['x', 'y', 'w', 'h']].to_numpy().tolist())
            # merge/join class info
            df = df.merge(self.classes[['category_id', 'category_name', 'supercategory']], on=['category_id'])
            # Add the csv's df to the list to be concatenated at the end
            df_list.append(df)
        # Squash it all together
        df_concat = pd.concat([x for x in df_list])
        # Get numeric IDs for each image
        df_concat['image_id'] = list(df_concat.groupby('file_name').ngroup())
        # annotation IDs must be unique, any trying to do it with a df is a pita--well handle it in the df_to_dict
        df_concat['id'] = 0
        # df_concat['id'] = list(df_concat.index.astype(str).to_list())
        return df_concat

    def df_to_dict(self, Df):
        images_list, annotations_list, categories_list = list(), list(), list()
        anno_idx = 1
        for row in tqdm(Df.itertuples(), total=len(Df), desc=f'[{self.dataset}] df → dict'):
            if self.valid_bbox(row) is False:
                continue
            _dict = self.dict_contructor(row)
            images_list.append(_dict['images'])
            categories_list.append(_dict['categories'])

            _dict['annotations']['id'] = anno_idx
            anno_idx += 1
            annotations_list.append(_dict['annotations'])
        data_dict = {'images': images_list, 'annotations': annotations_list, 'categories': categories_list}
        for key in ['images', 'categories']:
            data_dict[key] = self.get_unique(Dict=data_dict[key])
        return data_dict

    def dict_to_json(self, Dict, Dataset, SaveTo=None):
        if SaveTo is None:
            _json_path = os.path.join(self.image_dir, Dataset + '.json')
        else:
            _json_path = SaveTo
        with open(_json_path, 'w', encoding='utf-8') as f:
            json.dump(Dict, f, ensure_ascii=False, indent=2)
        return _json_path

    @staticmethod
    def get_unique(Dict):
        return list(map(dict, set(tuple(sorted(d.items())) for d in Dict)))

    @staticmethod
    def get_image_dir(csv_path):
        csv_path = str(csv_path)
        csv_path = csv_path.replace('annotations', 'images')
        return csv_path.replace('.txt', '')

    @staticmethod
    def get_img_size(image_path):
        im = Image.open(image_path)
        return im.size  # w x h

    @staticmethod
    def dict_contructor(Row):
        return {'images'     : {'id'       : int(Row.image_id),
                                'width'    : int(Row.image_width),
                                'height'   : int(Row.image_height),
                                'file_name': Row.file_name},
                'annotations': {'id'         : 0,
                                'image_id'   : Row.image_id,
                                'category_id': Row.category_id,
                                'bbox'       : [Row.x, Row.y, Row.w, Row.h],
                                'iscrowd'    : 0,
                                'area'       : Row.w * Row.h},
                'categories' : {'supercategory': Row.supercategory,
                                'id'           : Row.category_id,
                                'name'         : Row.category_name}}

    @staticmethod
    def valid_bbox(row):
        if row.w <= 3:
            return False
        if row.h <= 3:
            return False
        return True


if __name__ == '__main__':
    # print(green('\nStarting validation subdirectory JSON conversion.'))
    # for dataset in ['val']:
    #     image_dir = f'/home/adityakunapuli/data/{dataset}/images'
    #     csv_dir = f'/home/adityakunapuli/data/{dataset}/annotations'
    #     val_list = [Path(csv_dir).joinpath(x) for x in os.listdir(csv_dir)]
    #     for sub_dir in val_list:
    #         print(yellow(f'{sub_dir.stem}'))
    #         save_to = str(sub_dir.parent.joinpath(sub_dir.stem, 'val.json')).replace('annotations', 'images')
    #         self = CSVtoJSON(csv_list=[sub_dir], image_dir=image_dir, inference=False, dataset=dataset)
    #         dataframe = self.csv_to_df()
    #         dictionary = self.df_to_dict(Df=dataframe)
    #         json_path = self.dict_to_json(Dict=dictionary, Dataset=self.dataset, SaveTo=save_to)

    print(green('\nStarting training and validation JSON conversion.'))
    i = 1
    for dataset in ['train']:
        image_dir = f'/home/adityakunapuli/data/{dataset}/images'
        csv_dir = f'/home/adityakunapuli/data/{dataset}/annotations'
        csv_list = [Path(csv_dir).joinpath(x) for x in os.listdir(csv_dir) if x.endswith('txt') or x.endswith('csv')]
        self = CSVtoJSON(csv_list=csv_list, image_dir=image_dir, inference=False, dataset=dataset)
        # json_path = self.coco()
        # print(f'JSON saved to:{json_path}\n')
