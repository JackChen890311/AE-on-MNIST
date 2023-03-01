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
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt

from model import Autoencoder
from dataloader import MyDataloader
from constant import CONSTANT

C = CONSTANT()

class Kmeans():
    def __init__(self):
        self.reset_folder()
        self.setup()
        self.pass_model()

    def reset_folder(self):
        paths = ['output/k-center','output/k-example','output/k-center-real']
        for i in paths:
            if os.path.exists(i):
                shutil.rmtree(i)
            os.mkdir(i)

    def setup(self):
        path = 'output/2023-02-23~11:44:24/model'
        model = Autoencoder(C.in_size, C.latent_size, C.hidden_dims)
        model.load_state_dict(torch.load(path))
        self.model = model.to(C.device)
        self.dataloaders = MyDataloader()
        self.img_name = self.dataloaders.setup_kmeans()
        self.kmeans_vector = np.zeros((len(self.img_name),32))
        # print(img_name[0])
        print(self.kmeans_vector.shape)

    def pass_model(self):
        self.model.eval()
        for cnt, (x,y) in tqdm(enumerate(self.dataloaders.kmeans_loader)):
            x = x.to(C.device)
            latent = self.model.encoder(x)
            # print(latent.shape)
            self.kmeans_vector[C.bs*cnt:C.bs*(cnt+1),:] = latent.cpu().detach().numpy()
        # print(kmeans_vector)
        print('Passed Autoencoder Model.')

    def pass_kmeans(self, K):
        self.K = K
        self.kmeans = KMeans(n_clusters=K, random_state=0, n_init=10)
        # self.kmeans = MiniBatchKMeans(n_clusters=K, random_state=0, n_init=10, batch_size=20480)
        self.kmeans.fit(self.kmeans_vector)
        # print(kmeans_vector[0])
        # print(kmeans.labels_[:10])
        # print(kmeans.cluster_centers_[0])
    
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
        plt.savefig('output/k-elbow.png')

    def reconstruct_cluster_center(self):
        # Reconstruct cluster center vector
        cluster_center = np.array(self.kmeans.cluster_centers_)
        # print(cluster_center.shape)
        result = self.model.decoder(torch.tensor(cluster_center,dtype=torch.float).to(C.device))
        # print(result.shape)
        for i in range(self.K):
            # print(result[i].shape)
            reconstructed = (result.cpu().detach().numpy()[i].reshape((64,64)))*255
            cv2.imwrite('output/k-center/cluster_%d.png'%i,reconstructed)

    def reconstruct_real_center(self):
        def find_medoids(X, labels):
            """
            根據k-means算法的結果找到每個聚類的medoid
            :param X: 樣本數據，以numpy array表示
            :param labels: k-means算法的結果，即每個樣本所屬的聚類
            :return: 每個聚類的medoid
            """
            medoids = []
            # D = squareform(pdist(X))
            # for label in sorted(set(labels)):
            #     indices = np.where(labels == label)[0]
            #     distances = np.sum(D[indices][:, indices], axis=1)
            #     medoid_index = indices[np.argmin(distances)]
            #     medoids.append(medoid_index)
            for label in sorted(set(labels)):
                indices = np.where(labels == label)[0]
                distances = cdist(X[indices], X[indices])
                min_distances = np.min(distances, axis=1)
                medoid_index = indices[np.argmin(min_distances)]
                medoids.append(medoid_index)
            return np.array(medoids)
        medoids = find_medoids(self.kmeans_vector,self.kmeans.labels_)
        for cnt, i in enumerate(medoids):
            frame = cv2.imread(self.img_name[i])
            cv2.imwrite('output/k-center-real/cluster_%d.png'%cnt,frame)

    def cluster_sample(self, nob):
        for i in range(self.K): # for each cluster
            cnt = 0
            all_img = np.zeros((64*(nob//10),640,3))
            for j in range(nob):
                while self.kmeans.labels_[cnt] != i:
                    cnt += 1
                    continue
                img = cv2.resize(cv2.imread(self.img_name[cnt]),(64,64), interpolation=cv2.INTER_NEAREST)
                all_img[(j//10)*64:(j//10+1)*64,(j%10)*64:(j%10+1)*64,:] = img
                cnt += 1
            cv2.imwrite('output/k-example/cluster_%d.png'%i,all_img)
    
    def tsne_plot(self):
        if self.K > 20:
            print('Not supporting K > 20 yet.')
            return
        # TSNE dimension reduction
        print('TSNE started!')
        tsne_vector = TSNE(n_components=2, learning_rate='auto',init='random', perplexity=50).fit_transform(self.kmeans_vector)
        print('TSNE Done!')
        # print(tsne_vector.shape)
        cmap = plt.get_cmap('tab20')
        fig, ax = plt.subplots()
        for i in range(self.K):
            ax.scatter(tsne_vector[self.kmeans.labels_ == i, 0], tsne_vector[self.kmeans.labels_ == i, 1], label='Cluster' + str(i), color=cmap(i))
        plt.savefig('output/k-center/cluster_tsne.png')

    def dump_result(self):
        dict_pk = {
            'cluster':self.kmeans.labels_,
            'centers':self.kmeans.cluster_centers_,
            'img_name': self.img_name
        }
        with open('output/k-center/k_means.pk', 'wb') as f:
            pk.dump(dict_pk,f)

if __name__ == '__main__':
    kmeans = Kmeans()
    # kmeans.elbow(range(2,40,2))
    kmeans.pass_kmeans(K = 10)
    kmeans.reconstruct_cluster_center()
    kmeans.reconstruct_real_center()
    kmeans.cluster_sample(50)
    kmeans.tsne_plot()
    # kmeans.dump_result()
    
