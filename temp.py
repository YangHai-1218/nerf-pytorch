import os 
from os import path as osp
import random




if __name__ == '__main__':
    obj = '000006'
    with open(f'data/lm/image_lists/per_obj_image_lists/{obj}_train.txt', 'r') as f:
        train_images = list(map(lambda x:x.strip(), f.readlines()))
    with open(f'data/lm/image_lists/per_obj_image_lists/{obj}_test.txt', 'r') as f:
        test_images = list(map(lambda x: x.strip(), f.readlines()))

    new_train_images = []
    for image in train_images:
        seq, rgb, image_id =  image.split('/')
        new_train_images.append(image_id)
    train_images = new_train_images

    new_test_images = []
    for image in test_images:
        seq, rgb, image_id =  image.split('/')
        new_test_images.append(image_id)
    random.shuffle(new_test_images)
    valid_images = sorted(new_test_images[:100])
    test_images = sorted(new_test_images[100:200])
    
    
    with open(f'data/lm/nerf_image_lists/{obj}_train.txt', 'w') as f:
        f.writelines(map(lambda x:x+'\n', train_images))

    with open(f'data/lm/nerf_image_lists/{obj}_test.txt', 'w') as f:
        f.writelines(map(lambda x:x+'\n', test_images)) 
    
    with open(f'data/lm/nerf_image_lists/{obj}_val.txt', 'w') as f:
        f.writelines(map(lambda x:x+'\n', valid_images)) 
    