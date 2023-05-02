import os
import cv2
import torch
import random
import shutil
import numpy as np
import pandas as pd
import pickle as pk
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

from model import Autoencoder, VariationalAutoencoder
from dataloader import MyDataloader
from constant import CONSTANT

C = CONSTANT()

class Kmeans():
    def __init__(self, path):
        self.path = path
        kpaths = ['/k-center','/k-example','/k-center-real']
        for i in kpaths:
            if os.path.exists(path+i):
                shutil.rmtree(path+i)
            os.mkdir(path+i)
        model = VariationalAutoencoder(C.in_size, C.latent_size, C.hidden_dims)
        model.load_state_dict(torch.load(path+'/model'))
        self.model = model.to(C.device)
        self.dataloaders = MyDataloader()
        self.img_name = self.dataloaders.setup_kmeans()
        self.kmeans_vector = np.zeros((len(self.img_name),C.latent_size))
        self.truth = np.zeros((len(self.img_name)))
        # print(img_name[0])
        print(self.kmeans_vector.shape)

    def pass_model(self):
        self.model.eval()
        for cnt, (x,y) in tqdm(enumerate(self.dataloaders.kmeans_loader)):
            x = x.to(C.device)
            latent = self.model.encoder(x)
            # print(latent.shape)
            self.kmeans_vector[C.bs*cnt:C.bs*(cnt+1),:] = latent.cpu().detach().numpy()
            self.truth[C.bs*cnt:C.bs*(cnt+1)] = np.array(y)
        print('Passed Autoencoder Model.')
        # print(self.truth)
        # print(self.img_name)

    def pass_kmeans(self, K):
        self.K = K
        self.kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
        self.kmeans.fit(self.kmeans_vector)
        # print(kmeans_vector[0])
        # print(kmeans.labels_[:10])
        # print(kmeans.cluster_centers_[0])


    def pass_pca(self):
        pca = PCA(n_components=2)
        self.pc = pca.fit_transform(self.kmeans_vector)
        print(self.pc.shape)


    def elbow(self, Ks):
        sse = {}
        for k in tqdm(Ks):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            kmeans.fit(self.kmeans_vector)
            sse[k] = kmeans.inertia_ 
        # print(sse)
        plt.figure()
        plt.plot(list(sse.keys()), list(sse.values()))
        plt.xlabel("Number of cluster")
        plt.ylabel("SSE")
        plt.savefig(self.path+'/k-elbow.png')

    def reconstruct_cluster_center(self):
        # Reconstruct cluster center vector
        cluster_center = np.array(self.kmeans.cluster_centers_)
        # print(cluster_center.shape)
        result = self.model.decoder(torch.tensor(cluster_center,dtype=torch.float).to(C.device))
        # print(result.shape)
        for i in range(self.K):
            reconstructed = (result.cpu().detach().numpy()[i].reshape((C.image_dim,C.image_dim)))*255
            cv2.imwrite(self.path+'/k-center/cluster_%d.png'%i,reconstructed)

    def reconstruct_real_center(self):
        def find_medoids(X, labels):
            """
            Find the medoid of each cluster based on the pairwise distance between points.
            :param X: numpy array of shape (n_samples, n_features) representing the data
            :param labels: numpy array of shape (n_samples,) representing the cluster labels
            :return: numpy array of shape (n_clusters,) representing the indices of the medoids
            """
            medoids = []
            for label in np.unique(labels):
                indices = np.where(labels == label)[0]
                distances = np.sum(np.abs(X[np.newaxis, indices, :] - X[indices, np.newaxis, :]), axis=2)
                medoid_index = indices[np.argmin(np.sum(distances, axis=1))]
                medoids.append(medoid_index)
            return np.array(medoids)
        medoids = find_medoids(self.kmeans_vector,self.kmeans.labels_)
        for cnt, i in enumerate(medoids):
            frame = cv2.imread(self.img_name[i])
            cv2.imwrite(self.path+'/k-center-real/cluster_%d.png'%cnt,frame)

    def cluster_sample(self, nob):
        for i in range(self.K): # for each cluster
            cnt = 0
            all_img = np.zeros((C.image_dim*(nob//10),C.image_dim*10,3))
            indexes = [j for j in range(len(self.kmeans.labels_)) if self.kmeans.labels_[j] == i]
            indexes = random.sample(indexes, nob)
            for j,cnt in enumerate(indexes):
                img = cv2.imread(self.img_name[cnt])
                all_img[(j//10)*C.image_dim:(j//10+1)*C.image_dim,(j%10)*C.image_dim:(j%10+1)*C.image_dim,:] = img
            cv2.imwrite(self.path+'/k-example/cluster_%d.png'%i,all_img)
    
    def tsne_plot(self):
        if self.K > 20:
            print('Not supporting K > 20 yet.')
            return
        # TSNE dimension reduction
        print('TSNE started!')
        tsne_vector = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=1000).fit_transform(self.kmeans_vector)
        print('TSNE Done!')
        # print(tsne_vector.shape)
        cmap = plt.get_cmap('tab20')
        fig, ax = plt.subplots()
        for i in range(self.K):
            ax.scatter(tsne_vector[self.kmeans.labels_ == i, 0], tsne_vector[self.kmeans.labels_ == i, 1], label='Cluster' + str(i), color=cmap(i))
        plt.savefig(self.path+'/k-center/cluster_tsne.png')

    def dump_result(self):
        dict_pk = {
            'cluster':self.kmeans.labels_,
            'centers':self.kmeans.cluster_centers_,
            'img_name': self.img_name
        }
        with open(self.path+'/k-center/k_means.pk', 'wb') as f:
            pk.dump(dict_pk,f)

if __name__ == '__main__':
    path = 'output/2023-04-04~22:15:10'
    kmeans = Kmeans(path)
    kmeans.pass_model()
    # kmeans.elbow(range(2,40,2))
    kmeans.pass_kmeans(K = 10)
    # kmeans.pass_pca()
    kmeans.dump_result()
    kmeans.reconstruct_cluster_center()
    kmeans.reconstruct_real_center()
    kmeans.cluster_sample(500)
    kmeans.tsne_plot()