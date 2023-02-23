import os
import cv2
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from constant import CONSTANT

class MyDataset(Dataset):
    def __init__(self, data_path, paths):
        self.data_path = data_path
        self.paths = paths
        self.imgs = []
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        for p in tqdm(paths):
            pics = os.listdir(os.path.join(data_path,p))
            for pic in pics:
                img = cv2.resize(cv2.imread(os.path.join(data_path,p,pic),cv2.IMREAD_GRAYSCALE),(64,64), interpolation=cv2.INTER_NEAREST)
                self.imgs.append([str(os.path.join(data_path,p,pic)),img])
        self.imgs= pd.DataFrame(self.imgs, columns=['Name','Image'])

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        return torch.flatten(self.transform(self.imgs['Image'][idx])).squeeze(), self.imgs['Name'][idx]
    
    def show_image(self, idx):
        img = self.imgs['Image'][idx].reshape(64,64)
        plt.imshow(img)
        plt.show()
        return img

class MyDataloader():
    def __init__(self):
        super().__init__()
        self.C = CONSTANT()
        data_path = self.C.data_path
        data_path_test = self.C.data_path_test

        pids_all = os.listdir(data_path)
        pids_test = os.listdir(data_path_test)

        allpaths = {}
        allpaths['train'] = pids_all
        allpaths['val'] = pids_test
        allpaths['test'] = pids_test
        self.allpaths = allpaths

        # K-means
        pids_kmeans = pids_all
        self.kpaths = pids_kmeans
        
    def setup_all(self):
        print('Loading Data...')
        self.train_dataset = MyDataset(self.C.data_path, self.allpaths['train'])
        self.train_loader = self.loader_prepare(self.train_dataset, True)
        del self.train_dataset

        self.valid_dataset = MyDataset(self.C.data_path, self.allpaths['val'])
        self.valid_loader = self.loader_prepare(self.valid_dataset, False)
        del self.valid_dataset
        print('Preparation Done!')
    
    def setup_test(self):
        print('Loading Data...')
        self.test_dataset = MyDataset(self.C.data_path, self.allpaths['test']) 
        self.test_loader = self.loader_prepare(self.test_dataset, False)
        del self.test_dataset
        print('Preparation Done!')

    def setup_kmeans(self):
        print('Loading Data...')
        self.kmeans_dataset = MyDataset(self.C.data_path, self.kpaths) 
        self.kmeans_loader = self.loader_prepare(self.kmeans_dataset, False)
        img_name = self.kmeans_dataset.imgs['Name']
        del self.kmeans_dataset
        print('Preparation Done!')
        return img_name

    def loader_prepare(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.C.bs,
            num_workers=self.C.nw,
            shuffle=shuffle,
            pin_memory=self.C.pm,
        )

if __name__ == '__main__':
    dataloaders = MyDataloader()
    dataloaders.setup_all()

    for x,y in dataloaders.train_loader:
        print(x.shape,len(y))
        break