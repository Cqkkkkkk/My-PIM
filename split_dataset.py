import os
import pandas as pd
import shutil
import argparse

import pdb

args_parser = argparse.ArgumentParser()
args_parser.add_argument('--dir', type=str, help='Original dataset path')

args = args_parser.parse_args()

img_path = os.path.join(args.dir, 'images.txt')
images = pd.read_csv(img_path, names=['img_id', 'filepath'],  sep=' ')

img_class_labels_path = os.path.join(args.dir, 'image_class_labels.txt')
image_class_labels = pd.read_csv(img_class_labels_path, names=['img_id', 'target'], sep=' ')

train_test_split_path = os.path.join(args.dir, 'train_test_split.txt')
train_test_split = pd.read_csv(train_test_split_path, names=['img_id', 'is_training_img'], sep=' ')


# images = pd.read_csv('images.txt', names=['img_id', 'filepath'],  sep=' ')
# image_class_labels = pd.read_csv('image_class_labels.txt', names=['img_id', 'target'], sep=' ')
# train_test_split = pd.read_csv('train_test_split.txt', names=['img_id', 'is_training_img'], sep=' ')
# print(images)

data = images.merge(image_class_labels, on='img_id')
data = data.merge(train_test_split, on='img_id')

train = data[data.is_training_img == 1]
test = data[data.is_training_img == 0]


for row in train.iterrows():
    file_path = row[1].filepath
    file_dir = file_path.split('/')[0]
    file_name = file_path.split('/')[-1]
    file_path = os.path.join(args.dir, 'images', file_path)
    # dst_dir = 'sortedImages/train/{}'.format(file_dir)
    dst_dir = os.path.join(args.dir, 'sortedImages', 'train', file_dir)
    dst_path = os.path.join(dst_dir, file_name)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.copyfile(file_path, dst_path)

for row in test.iterrows():
    file_path = row[1].filepath
    file_dir = file_path.split('/')[0]
    file_name = file_path.split('/')[-1]
    file_path = os.path.join(args.dir, 'images', file_path)
    dst_dir = os.path.join(args.dir, 'sortedImages', 'test', file_dir)
    # dst_dir = 'sortedImages/test/{}'.format(file_dir)
    dst_path = os.path.join(dst_dir, file_name)

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    shutil.copyfile(file_path, dst_path)


