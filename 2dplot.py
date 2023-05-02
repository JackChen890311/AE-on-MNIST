import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from model import Autoencoder, VariationalAutoencoder
from dataloader import MyDataloader
from constant import CONSTANT

C = CONSTANT()
NoB = 30

def pass_model(path):
    model = VariationalAutoencoder(C.in_size, C.latent_size, C.hidden_dims)
    model.load_state_dict(torch.load(path))
    model = model.to(C.device)
    model.eval()
    dataloaders = MyDataloader()
    dataloaders.setup_all()
    train_loader = dataloaders.train_loader

    minx1, maxx1 = 1e5, -1e5
    minx2, maxx2 = 1e5, -1e5
    for x,y in train_loader:
        x = x.to(C.device)
        result = model.encoder(x).detach().cpu().numpy()
        minx1 = min(minx1, min(result[:,0]))
        minx2 = min(minx2, min(result[:,1]))
        maxx1 = max(maxx1, max(result[:,0]))
        maxx2 = max(maxx2, max(result[:,1]))

    x1list = np.linspace(minx1, maxx1, NoB)
    x2list = np.linspace(minx2, maxx2, NoB)

    return x1list, x2list, model


def plot(x1list, x2list, model):
    pics = np.zeros((NoB*28,NoB*28))
    for id1, x1 in enumerate(x1list):
        for id2, x2 in enumerate(x2list):
            latent = torch.tensor((x1,x2),dtype=torch.float,device=C.device)
            pic = model.decoder(latent).detach().cpu().numpy()
            pic = pic.reshape((28,28))
            pics[int(id1*28):int((id1+1)*28), int(id2*28):int((id2+1)*28)] = pic*255
    cv2.imwrite('output/2dplot.png',pics)

if __name__ == '__main__':
    x1list, x2list, model = pass_model('output/2023-05-02~19:36:20/model')
    plot(x1list, x2list, model)
    
      