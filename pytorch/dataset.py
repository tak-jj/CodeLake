import os
import glob
import cv2
import pandas as pd
from torch.utils.data import Dataset

# annotation csv가 다음과 같이 주어졌을 때.
# annotation csv is given like below.
'''
annotation_file.csv
---
,file_name,label
0,01.png,0
1,02.png,1
.
.
.
---
'''
class CustomDataset(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_label = pd.read_csv(annotation_file, names=['file_name', 'label'], skiprows=[0])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = os.path.join(self.img_dir, self.img_label.iloc[index, 0])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = self.img_label.iloc[index, 1]
        if self.transform:
            image = self.transform(image)['image']
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

    def __len__(self):
        return len(self.img_label)

# 이미지-폴더가 다음과 같이 구성되었을 때.
# image-folder constructed like below.
'''
folder
- train
-- label
--- image.png
'''
def get_label_dict(path):
    label_dict = {}
    for i, label in enumerate(os.listdir(path)):
        label_dict[label] = i
    
    return label_dict

class ImgFolderDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.file_path = glob.glob(os.path.join(img_dir, '*', '*.png'))
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path = self.file_path[index]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # temp_label = img_path.split('\\')[-2] window 환경, window env
        temp_label = img_path.split('/')[-2]
        label = label_dict[temp_label]
        if self.transform:
            image = self.transform(image)['image']
        if self.target_transform:
            label = self.target_transform(label)

        return image, label
    
    def __len__(self):
        return len(self.file_path)