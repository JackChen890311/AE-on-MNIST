import os
import cv2
import time
import torch
import random
import numpy as np
import pickle as pk
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import Autoencoder
from dataloader import MyDataloader
from constant import CONSTANT

C = CONSTANT()

def train(model, train_loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for x,y in train_loader:
        x = x.to(C.device)
        optimizer.zero_grad()
        result = model(x)
        loss = loss_fn(x,result)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss/len(train_loader)


def valid(model, valid_loader, loss_fn):
    model.eval()
    total_loss = 0
    for x,y in valid_loader:
        x = x.to(C.device)
        result = model(x)
        loss = loss_fn(x,result)
        total_loss += loss.item()
    return total_loss/len(valid_loader)


def reconstruction(model, test_loader, epoch, time, img_dim):
    model.eval()
    for x,y in test_loader:
        x = x.to(C.device)
        idx = random.randint(0,len(y)-1)
        result = model(x).cpu().detach().numpy()[idx].reshape((img_dim,img_dim))
        origin = x.cpu().detach().numpy()[idx].reshape((img_dim,img_dim))
        concatenated = np.concatenate([origin*255,result*255,np.round(result)*255],axis=1)
        cv2.imwrite('output/%s/recon/reconstructed_%d.png'%(time,epoch),concatenated)
        return

def reconstruction_multiple(model, test_loader, time, name, nob, img_dim):
    model.eval()
    cnt = 0
    images = []
    for x,y in test_loader:
        x = x.to(C.device)
        idx = random.randint(0,len(y)-1)
        result = model(x).cpu().detach().numpy()[idx].reshape((img_dim,img_dim))
        origin = x.cpu().detach().numpy()[idx].reshape((img_dim,img_dim))
        images.append(np.concatenate([origin*255,result*255,np.round(result)*255],axis=1))
        cnt += 1
        if cnt == nob:
            break
    concatenated = np.concatenate(images,axis=0)
    cv2.imwrite('output/%s/recon_multi/reconstructed_%s.png'%(time,name),concatenated)
    return

def main():
    model = Autoencoder(C.in_size, C.latent_size, C.hidden_dims)
    model = model.to(C.device)
    dataloaders = MyDataloader()
    dataloaders.setup_all()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=C.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = C.milestones, gamma = C.gamma)
    img_dim = C.image_dim

    start_time = str(time.strftime("%Y-%m-%d~%H:%M:%S", time.localtime()))
    if not os.path.exists('output'):
        os.mkdir('output')
    os.mkdir('output/%s'%start_time)
    os.mkdir('output/%s/recon'%start_time)
    os.mkdir('output/%s/recon_multi'%start_time)
    train_losses = []
    valid_losses = []
    p_cnt = 0
    best_valid_loss = 1e5

    for e in tqdm(range(1,1+C.epochs)):
        train_loss = train(model, dataloaders.train_loader, optimizer, loss_fn)
        valid_loss = valid(model, dataloaders.valid_loader, loss_fn)
        scheduler.step()
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print('Epoch = ',e, 'Train / Valid Loss = %f / %f'%(train_loss,valid_loss))
        if e % 10 == 0:
            reconstruction(model, dataloaders.valid_loader, e, start_time, img_dim)

        if valid_loss < best_valid_loss:
            p_cnt = 0
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), 'output/%s/model'%start_time)
        else:
            p_cnt += 1
            if p_cnt == C.patience:
                print('Early Stopping at epoch',e)
                break
        
        if e % 100 == 0:
            reconstruction_multiple(model, dataloaders.valid_loader, start_time, 'epoch%d'%e, 30, img_dim)
            print('Plotting Loss at epoch', e)
            x_axis = list(range(e))
            plt.plot(x_axis, train_losses, label='Train')
            plt.plot(x_axis, valid_losses, label='Valid')
            plt.legend()
            plt.savefig('output/%s/loss.png'%start_time)
            plt.clf()

            plt.plot(x_axis[-100:], train_losses[-100:], label='Train')
            plt.plot(x_axis[-100:], valid_losses[-100:], label='Valid')
            plt.legend()
            plt.savefig('output/%s/loss_last100.png'%start_time)
            plt.clf()

        with open('output/%s/losses.pickle'%start_time, 'wb') as file:
            pk.dump([train_losses, valid_losses, best_valid_loss], file)
        
    print('Ending at epoch',e, '. Best valid loss:',best_valid_loss)

def test(path):
    model = Autoencoder(C.in_size, C.latent_size, C.hidden_dims)
    model.load_state_dict(torch.load(path))
    model = model.to(C.device)
    model.eval()
    dataloaders = MyDataloader()
    dataloaders.setup_test()
    time = path.split('/')[1]
    reconstruction_multiple(model, dataloaders.test_loader, time, 'test', 30, C.image_dim)


if __name__ == '__main__':
    main()
    # test('output/2023-03-30~22:14:50/model')

