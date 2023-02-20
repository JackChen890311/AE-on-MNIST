import os
import cv2
import torch
import random
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from model import Autoencoder
from dataloader import MyDataloader
from constant import CONSTANT

C = CONSTANT()
path = 'output/2023-02-15~15:17:27/model'

model = Autoencoder(C.in_size, C.latent_size, C.hidden_dims)
model.load_state_dict(torch.load(path))
model = model.to(C.device)
dataloaders = MyDataloader()
img_name = dataloaders.setup_kmeans()
kmeans_vector = np.zeros((len(img_name),32))
# print(img_name[0])
print(kmeans_vector.shape)


cnt = 0
model.eval()
for x,y in dataloaders.kmeans_loader:
    x = x.to(C.device)
    latent = model.encoder(x)
    # print(latent.shape)
    kmeans_vector[C.bs*cnt:C.bs*(cnt+1),:] = latent.cpu().detach().numpy()
    cnt += 1
# print(kmeans_vector)


# K-means
kmeans = KMeans(n_clusters=20, random_state=0, n_init=10)
kmeans.fit(kmeans_vector)
# print(kmeans_vector[0])
# print(kmeans.labels_[:10])
# print(kmeans.cluster_centers_[0])


# Reconstruct cluster center vector
cluster_center = np.array(kmeans.cluster_centers_)
# print(cluster_center.shape)
result = model.decoder(torch.tensor(cluster_center,dtype=torch.float).to(C.device))
# print(result.shape)
for i in range(20):
    # print(result[i].shape)
    reconstructed = (result.cpu().detach().numpy()[i].reshape((64,64)))*255
    cv2.imwrite('output/k-center/cluster_%d.png'%i,reconstructed)


# TSNE dimension reduction
tsne_vector = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=3).fit_transform(kmeans_vector)
# print(tsne_vector.shape)

cmap = plt.get_cmap('tab20')

fig, ax = plt.subplots()
for i in range(20):
    ax.scatter(tsne_vector[kmeans.labels_ == i, 0], tsne_vector[kmeans.labels_ == i, 1], label='Cluster' + str(i), color=cmap(i))

# ax.legend()
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
plt.savefig('output/k-center/cluster_tsne.png')
