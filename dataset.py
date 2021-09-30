import os

import cv2
import numpy as np
import random
import torch
from torch.nn import functional as F
from torch.utils import data

IMG_SIZE = (640, 480)
TEMPLATE_SIZE = (124, 124)

class UIScreen(data.Dataset):
    def __init__(self,
                 dir='data',
                 mode = 'train',
                 category=['positive', 'negative'],
                 base_size=512,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225]):

        self.base_size = base_size
        self.category = category
        self.mean = mean
        self.std = std
        self.dir = dir
        self.mode = mode
        self.num_classes = len(category)

        self.dict_samples = {}
        for c in self.category:
            self.dict_samples[c] = [os.path.join(self.dir, mode, c, f) 
            for f in os.listdir(os.path.join(self.dir, mode, c))]

        print('\nFinish loading dataset, %d in total \n' % (self.__len__()))

    def __len__(self):
        # return len(self.source_files * self.num_classes)
        return sum(len(lst) for lst in self.dict_samples.values()) // 3

    def input_transform(self, image):
        image = image.astype(np.float32)[:, :, ::-1]
        image = image / 255.0
        image -= self.mean
        image /= self.std
        return image

    # def pad_to_img(self, img, template):
    #     h1, w1, c = img.shape
    #     h2, w2, c = template.shape
    #     top = int((h1 - h2) / 2)
    #     left = int((w1 - w2) / 2)
    #     bottom = h1 - top
    #     right = w1 - left
    #     pad_template = cv2.copyMakeBorder(
    #         template, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT)
    #     return pad_template

    def __getitem__(self, i):
        c_id = self.category[i % len(self.category)]
        f_id = i // len(self.category)
        
        img_path = os.path.join(self.dir, self.mode, c_id, "{}_img.jpg".format(f_id))
        msk_path = os.path.join(self.dir, self.mode, c_id, "{}_msk.jpg".format(f_id))
        tmp_path = os.path.join(self.dir, self.mode, c_id, "{}_tmp.jpg".format(f_id))

#        print(img_path)
#        print(msk_path)
#        print(tmp_path)

        img = cv2.imread(img_path)
        tmp = cv2.imread(tmp_path)
        msk = cv2.imread(msk_path)

        img = self.input_transform(img)
        tmp = self.input_transform(tmp)

        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        msk = msk.astype(np.float32) / 255.0

#        img = cv2.resize(img, dsize=IMG_SIZE, interpolation=cv2.INTER_LINEAR)
#        tmp = cv2.resize(tmp, dsize=TEMPLATE_SIZE, interpolation=cv2.INTER_LINEAR)

        img = img.transpose((2, 0, 1))
        tmp = tmp.transpose((2, 0, 1))      
        msk = msk[:, :, np.newaxis].transpose((2, 0, 1))    #.astype(np.int32)

        return img, tmp, msk


if __name__ == '__main__':
    ui_screen = UIScreen()
    dataset_iter = iter(ui_screen)
    for img, tmp, msk in dataset_iter:
        print(img.shape, tmp.shape, msk.shape)
