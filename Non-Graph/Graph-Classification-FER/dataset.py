import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import os


def make_dataset(image_list, label_list):
    len_ = len(image_list)
    images = [(image_list[i].strip(),  label_list[i]) for i in range(len_)]
    return images


def pil_loader(path):
    with open(path, 'rb') as f:
        # with Image.open(f) as img:
        #     return img.convert('RGB')
        img = Image.open(f)
        img.load()
        return img


def default_loader(path):
    return pil_loader(path)


class FER2013(Dataset):
    def __init__(self, root_path, train=True,  transform=None, crop_size = 224, stage=1, loader=default_loader):

        assert stage>0 and stage <=2, 'The stage num must be restricted from 1 to 2'
        self._root_path = root_path
        self._train = train
        self._stage = stage
        self._transform = transform
        self.crop_size = crop_size
        self.loader = loader
        self.img_folder_path = os.path.join(root_path,'img')
        if self._train:
            # img
            train_image_list_path = os.path.join(root_path, 'list', 'train_img' +'.txt')
            train_image_list = open(train_image_list_path).readlines()
            # img labels
            train_label_list_path = os.path.join(root_path, 'list', 'train_label' + '.txt')
            train_label_list = np.loadtxt(train_label_list_path)

        
            self.data_list = make_dataset(train_image_list, train_label_list)

        else:
            # img
            test_image_list_path = os.path.join(root_path, 'list', 'test_img' + '.txt')
            test_image_list = open(test_image_list_path).readlines()

            # img labels
            test_label_list_path = os.path.join(root_path, 'list', 'test_label' + '.txt')
            test_label_list = np.loadtxt(test_label_list_path)
            self.data_list = make_dataset(test_image_list, test_label_list)

    def __getitem__(self, index):
        if self._stage == 2 and self._train:
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            if self._transform is not None:
                img = self._transform(img)
            return img, label
        else:
            img, label = self.data_list[index]
            img = self.loader(os.path.join(self.img_folder_path, img))

            if self._train:
                if self._transform is not None:
                    img = self._transform(img)
            else:
                if self._transform is not None:
                    img = self._transform(img)
            return img, label

    def __len__(self):
        return len(self.data_list)