import torch

class CONSTANT():
    def __init__(self):
        self.epochs = 1000
        self.lr = 1e-5
        self.bs = 512
        self.nw = 8
        self.pm = True
        self.milestones = [50,500,5000]
        self.gamma = 0.5
        self.patience = 20
        self.device = torch.device('cuda:0')

        self.data_path = 'data/training'
        self.data_path_test = 'data/testing'
        self.num_of_folder = 10
        self.num_of_folfer_kmeans = 10
        # self.train_portion = 1
        # self.valid_test_portion = 1
        '''
        For example, If the above numbers are 250, 200, 0.8, 0.1
        Then the indexes of used folders are:
        Train: 0 ~ 200
        Valid: 200 ~ 250
        Test: 225 ~ 250
        K-means Clustering: Randomly choose 200 folders from 632 folders
        Maximum is 632 folders
        '''

        self.hidden_dims = [2048, 512, 128]
        self.in_size = 4096
        self.latent_size = 32

        