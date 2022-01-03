import os
import cv2
import numpy as np
import random

path_pool = './raw_data'
path_train = './data/train'
path_test = './data/test'


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
#            img = cv2.resize(img, dsize=(512,512), interpolation=cv2.INTER_AREA)
            img = cv2.resize(img, dsize=(480,640), interpolation=cv2.INTER_AREA)
            images.append(img)
    return images

def gen_pos_samples(path_pool, path_data, num_samples=10, min_sub_ratio=0.2, max_sub_ratio=0.4):
    images = load_images_from_folder(path_pool)
    num_images = len(images)
    
    for i in range(num_samples):
        index = random.randrange(num_images)
        
        img = images[index]
        img_h, img_w = img.shape[:2]
        
        min_sub_size = int(min(img_h, img_w) * min_sub_ratio)
        max_sub_size = int(min(img_h, img_w) * max_sub_ratio)

        sub_w = random.randrange(min_sub_size, max_sub_size)
        sub_h = random.randrange(min_sub_size, max_sub_size)
        sub_x = random.randrange(img_w - sub_w)
        sub_y = random.randrange(img_h - sub_h)

        sub_img = img[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]
        
        tmp_x = (int)(img_w/2)-(int)(sub_w/2) 
        tmp_y = (int)(img_h/2)-(int)(sub_h/2)
        tmp_w = sub_w
        tmp_h = sub_h

        tmp = np.zeros_like(img)
        tmp[tmp_y:tmp_y+tmp_h, tmp_x:tmp_x+tmp_w] = sub_img[0:sub_h, 0:sub_w]

        msk = np.zeros_like(img)
        msk = cv2.rectangle(msk, (sub_x, sub_y), (sub_x+sub_w, sub_y+sub_h), (255,255,255), -1)
#        msk = cv2.rectangle(img, (10, 10), (100, 100), (0,0,255), 10)
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('{}/positive/{}_img.jpg'.format(path_data, i), img)
        cv2.imwrite('{}/positive/{}_tmp.jpg'.format(path_data, i), tmp)
#        cv2.imwrite('{}/positive/{}_tmp.jpg'.format(path_data, i), sub_img)
        cv2.imwrite('{}/positive/{}_msk.jpg'.format(path_data, i), msk)


def gen_neg_samples(path_pool, path_data, num_samples=10, min_sub_ratio=0.2, max_sub_ratio=0.4):
    images = load_images_from_folder(path_pool)
    num_images = len(images)
    
    for i in range(num_samples):
        img_index = random.randrange(num_images)
        tmp_index = random.randrange(num_images-1)

        img = images[img_index]
        img2 = images[(img_index+tmp_index) % num_images]

        img2_h, img2_w = img2.shape[:2]
        
        min_sub_size = int(min(img2_h, img2_w) * min_sub_ratio)
        max_sub_size = int(min(img2_h, img2_w) * max_sub_ratio)

        sub_w = random.randrange(min_sub_size, max_sub_size)
        sub_h = random.randrange(min_sub_size, max_sub_size)
        sub_x = random.randrange(img2_w - sub_w)
        sub_y = random.randrange(img2_h - sub_h)

        sub_img2 = img2[sub_y:sub_y+sub_h, sub_x:sub_x+sub_w]

        tmp_x = (int)(img2_w/2)-(int)(sub_w/2) 
        tmp_y = (int)(img2_h/2)-(int)(sub_h/2)
        tmp_w = sub_w
        tmp_h = sub_h

        tmp = np.zeros_like(img)
        tmp[tmp_y:tmp_y+tmp_h, tmp_x:tmp_x+tmp_w] = sub_img2[0:sub_h, 0:sub_w]

        msk = np.zeros_like(img)
        msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)

        cv2.imwrite('{}/negative/{}_img.jpg'.format(path_data, i), img)
        cv2.imwrite('{}/negative/{}_tmp.jpg'.format(path_data, i), tmp)
#        cv2.imwrite('{}/negative/{}_tmp.jpg'.format(path_data, i), sub_img2)
        cv2.imwrite('{}/negative/{}_msk.jpg'.format(path_data, i), msk)


if __name__ == '__main__':
    gen_pos_samples(path_pool, path_train, 5000)
    gen_neg_samples(path_pool, path_train, 5000)

    gen_pos_samples(path_pool, path_test, 500)
    gen_neg_samples(path_pool, path_test, 500)
