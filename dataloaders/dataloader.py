import os

import numpy as np
from PIL import Image
import scipy.io as sio
import torch
from torch.utils.data import ConcatDataset, Dataset, DataLoader
import torchvision.transforms as transforms

import dataloaders.custom_transforms as tr

class SBDDataset(Dataset):
    def __init__(self, params, data_dir="../../Dataset/SBD/dataset/img", mask_dir="../../Dataset/SBD/dataset/cls", dataset_type="../../Dataset/SBD/dataset/train_aug.txt"):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type
        self.dataset = []
        self.base_size = params.base_size
        self.crop_size = params.crop_size
        with open(self.dataset_type, 'r') as f:
            for line in f.readlines():
                self.dataset.append('{}.mat'.format(line.strip()))
        self.mask_filenames = os.listdir(mask_dir)
        self.mask_filenames = [os.path.join(mask_dir, f) for f in self.dataset]

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask = sio.loadmat(self.mask_filenames[idx])
        mask = Image.fromarray(mask['GTcls'][0]['Segmentation'][0])
        image = Image.open(self.mask_filenames[idx].replace('mat', 'jpg').replace(self.mask_dir, self.data_dir))
        sample = {'image': image, 'label': mask}
        return self.transform(sample)
    
    def transform(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

class VOCDataset(Dataset):
    def __init__(self, split, params, data_dir="../../Dataset/VOC2012/JPEGImages", mask_dir="../../Dataset/VOC2012/SegmentationClass", dataset_type="../../Dataset/VOC2012/ImageSets/Segmentation/"):
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.dataset_type = dataset_type
        self.dataset_type = os.path.join(self.dataset_type, split + '.txt')
        self.split =split
        self.base_size = params.base_size
        self.crop_size = params.crop_size
        self.dataset = []
        with open(self.dataset_type, 'r') as f:
            for line in f.readlines():
                self.dataset.append('{}.png'.format(line.strip()))
        self.mask_filenames = os.listdir(mask_dir)
        self.mask_filenames = [os.path.join(mask_dir, f) for f in self.dataset]

    def __len__(self):
        return len(self.mask_filenames)

    def __getitem__(self, idx):
        mask = Image.open(self.mask_filenames[idx])
        image = Image.open(self.mask_filenames[idx].replace('png', 'jpg').replace(self.mask_dir, self.data_dir))
        sample = {'image': image, 'label': mask}

        if self.split == "train":
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.base_size, crop_size=self.crop_size),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

def fetch_dataloader(types, params):
    dataloaders={}
    for split in ['train', 'val', 'test']:
        if split in types:
            if split == 'train':
                VOC = VOCDataset(split, params)
                SBD = SBDDataset(params)
                dl = DataLoader(ConcatDataset([VOC, SBD]), batch_size=params.batch_size, shuffle=True,
                                num_workers=params.num_workers, pin_memory=params.cuda, drop_last=False)
            else:
                dl = DataLoader(VOCDataset(split, params), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_workers, pin_memory=params.cuda, drop_last=False)
            dataloaders[split] = dl
    return dataloaders

if __name__ == '__main__':
    import torch
    import utils.utils as utils
    json_path = os.path.join('experiments', 'params.json')
    params = utils.Params(json_path)
    params.cuda = torch.cuda.is_available()
    dataloader = fetch_dataloader(['train'], params)
    train_dl = dataloader['train']
    for ii, sample in enumerate(train_dl):
        img = sample['image'].numpy()
        gt = sample['label'].numpy()

        if ii == 10:
            break