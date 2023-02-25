import torch
from torch.utils.data import Dataset
import os
from image import *
#from image_sr import *
import random

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4, phase='1024'):
        if train:
            random.shuffle(root)

        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.phase = phase


    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        # print(self.phase)
        img_path = self.lines[index]
        fname = os.path.basename(img_path)
        img,target,kpoint,sigma_map= load_data(img_path,self.train)
        '''data augmention'''
        if self.train==True:
            if random.random() > 0.5:
                target = np.fliplr(target)
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                kpoint = np.fliplr(kpoint)
        target = target.copy()
        kpoint = kpoint.copy()
        img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        target = torch.from_numpy(target).cuda()

        if self.train == True:
            if self.phase == 'x2':
                read_path = '/data/xiejiahao/Crowd_SR/train/train_1024/' + fname
            if self.phase == 'x4':
                read_path = '/data/xiejiahao/Crowd_SR/train/train_2048/' + fname
            img_sr = Image.open(read_path).convert('RGB')

            if self.transform is not None:
                img_sr = self.transform(img_sr)

            return img, target, kpoint, fname, img_sr
        else:
            return img, target, kpoint, fname
