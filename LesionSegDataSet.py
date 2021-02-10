import os
import torch
import pandas as pd
from skimage import color
import matplotlib.pyplot as plt
import numpy as np
import cv2
from torch.utils import data
from PIL import Image


def bb2mask(bb_boxes, out_size):
    # util function generate a mask from BB data
    # in:
    # bb_boxes is a DF of all BB in file
    # out_size the mask's out size
    # out:
    # img_mask - size of [out_size,2] first chan is lesion 2nd is back
    img_mask = np.zeros([*out_size, 2],dtype='uint8')
    for i in range(len(bb_boxes)):
        bb_box_i = [bb_boxes.iloc[i]['xmin'], bb_boxes.iloc[i]['ymin'],
                    bb_boxes.iloc[i]['xmax'], bb_boxes.iloc[i]['ymax']]
        img_mask[int(bb_box_i[1]):int(bb_box_i[3]), int(bb_box_i[0]):int(bb_box_i[2]), 0] = 1.
    img_mask[:, :, 1] = 1 - img_mask[:, :, 0]
    return img_mask


class LesionSegDataSet(data.Dataset):

    def __init__(self, root_dir, csv_file, data_set=None, transform=None):
        df_files = pd.read_csv(os.path.join(root_dir, csv_file), header=0, index_col='index')
        if not (data_set is None):
            df_files = df_files[df_files['set'] == data_set]
        self.landmarks_frame = df_files
        self.root_dir = root_dir
        self.transform = transform
        self.out_size = (int(224), int(224))
        self.ind_array = self.landmarks_frame.index.unique()

    def __len__(self):
        return len(self.landmarks_frame.index.unique())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = self.ind_array[idx]
        img_name = os.path.join(self.root_dir,
                                self.landmarks_frame['File_Path'].iloc[
                                    self.landmarks_frame['File_Path'].index == idx].iloc[0])
        image = cv2.imread(img_name)
        orig_img_size = np.shape(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.out_size)

        bb_boxes = self.landmarks_frame.iloc[self.landmarks_frame.index == idx].reset_index()

        bb_boxes['xmin'] = np.round(bb_boxes['xmin'] / orig_img_size[1] * self.out_size[1])
        bb_boxes['xmax'] = np.round(bb_boxes['xmax'] / orig_img_size[1] * self.out_size[1])
        bb_boxes['ymin'] = np.round(bb_boxes['ymin'] / orig_img_size[0] * self.out_size[0])
        bb_boxes['ymax'] = np.round(bb_boxes['ymax'] / orig_img_size[0] * self.out_size[0])
        bb_boxes['Area'] = (bb_boxes['xmax'] - bb_boxes['xmin']) * (bb_boxes['ymax'] - bb_boxes['ymin'])
        mask = bb2mask(bb_boxes, self.out_size)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        sample = {'image': image, 'mask': mask}



        return sample

    def show_item(self, idx):
        sample = self[idx]
        plt.subplot(3, 1, 1)
        plt.imshow(sample['image'])
        plt.subplot(3, 1, 2)
        plt.imshow(color.label2rgb(sample['mask'][:, :, 0], sample['image'], bg_label=0))
        plt.subplot(3, 1, 3)
        plt.imshow(color.label2rgb(sample['mask'][:, :, 1], sample['image'], bg_label=0))
        plt.show()

# source_directory = r'D:\downloads\neuralNetwork'
# csv_file = 'labels_lesion.csv'
# lessionSegDataset1 = LesionSegDataSet(source_directory, csv_file,data_set='test')
# lessionSegDataset1.__len__()
# lessionSegDataset1[55]
# lessionSegDataset1.show_item(55)
# a = 1
