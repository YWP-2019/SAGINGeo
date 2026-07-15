import cv2
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import random
import copy
import torch
import os
import glob
from PIL import Image
import torchvision.transforms.functional as TF
cv2.setNumThreads(2)


class MultiDatasetTrain(Dataset):

    def __init__(self,
                 data_folder,
                 task=None,
                 split=80,
                 transforms_square=None,
                 transforms_wide=None,
                 prob_flip=0.0,
                 prob_rotate=0.0,
                 img_size=256,
                 ):

        super().__init__()
        self.data_folder = data_folder
        self.task = task
        self.transforms_square = transforms_square
        self.transforms_wide = transforms_wide
        self.prob_flip = prob_flip
        self.prob_rotate = prob_rotate
        self.img_size = img_size

        csv_file = f"{split}%train_RSI_SVI_UAV_VGI.csv"
        self.df = pd.read_csv(os.path.join(self.data_folder, "RSI_SVI_UAV_VGI", csv_file), header=None)
        self.samples = self.df.values.tolist()
        self.MAX_FRAMES = 54

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        keys = ["RSI", "SVI", "UAV", "VGI"]
        sample = {}

        # Decide once whether to flip or not for this sample
        do_flip = random.random() < self.prob_flip

        for i, key in enumerate(keys):
            if key not in self.task:
                continue

            img_path = os.path.normpath(os.path.join(self.data_folder, row[i].replace('\\', '/')))


            if key == "UAV":
                uav_id = os.path.splitext(os.path.basename(img_path))[0]
                uav_folder = os.path.normpath(os.path.join(self.data_folder, "UAV_preprocessed", uav_id))
                img_files = sorted(glob.glob(os.path.join(uav_folder, "*.jpg")))

                if len(img_files) > 0:
                    last_img_path = img_files[-1]
                    img = np.array(Image.open(last_img_path).convert("RGB"))

                    if self.transforms_square:
                        img = self.transforms_square(image=img)['image']

                    if do_flip:
                        img = TF.hflip(img)
                else:
                    img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)

                sample[key] = img


            else:
                img = np.array(Image.open(img_path).convert("RGB"))

                if key in ["RSI", "VGI"]:
                    if self.transforms_square:
                        img = self.transforms_square(image=img)['image']
                else:
                    if self.transforms_wide:
                        img = self.transforms_wide(image=img)['image']

                if do_flip:
                    img = TF.hflip(img)

                sample[key] = img

        return sample



class MultiDatasetEval(Dataset):

    def __init__(self,
                 data_folder,
                 task=None,
                 split=20,  # percentage of test set
                 transforms_square=None,
                 transforms_wide=None,
                 img_size=256,
                 train=False
                 ):

        super().__init__()

        self.data_folder = data_folder
        self.task = task
        self.transforms_square = transforms_square  # for RSI and VGI
        self.transforms_wide = transforms_wide      # for SVI
        if train:
            csv_file = f"{100-split}%train_RSI_SVI_UAV_VGI.csv"
        else:
            csv_file = f"{split}%test_RSI_SVI_UAV_VGI.csv"
        self.df = pd.read_csv(os.path.join(data_folder, "RSI_SVI_UAV_VGI", csv_file), header=None)
        self.samples = self.df.values.tolist()
        self.MAX_FRAMES = 54
        self.img_size = img_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        row = self.samples[index]
        keys = ["RSI", "SVI", "UAV", "VGI"]
        sample = {}

        for i, key in enumerate(keys):
            if key not in self.task:
                continue

            img_path = os.path.normpath(os.path.join(self.data_folder, row[i].replace('\\', '/')))

            if key == "UAV":
                uav_id = os.path.splitext(os.path.basename(img_path))[0]
                uav_folder = os.path.normpath(os.path.join(self.data_folder, "UAV_preprocessed", uav_id))
                img_files = sorted(glob.glob(os.path.join(uav_folder, "*.jpg")))

                if len(img_files) > 0:
                    last_img_path = img_files[-1]
                    img = np.array(Image.open(last_img_path).convert("RGB"))

                    if self.transforms_square:
                        img = self.transforms_square(image=img)['image']
                else:
                    img = torch.zeros((3, self.img_size, self.img_size), dtype=torch.float32)

                sample[key] = img


            else:
                img = np.array(Image.open(img_path).convert("RGB"))

                if key in ["RSI", "VGI"]:
                    if self.transforms_square:
                        img = self.transforms_square(image=img)['image']
                else:
                    if self.transforms_wide:
                        img = self.transforms_wide(image=img)['image']

                sample[key] = img

        label = torch.tensor(index, dtype=torch.long)

        return sample, label






