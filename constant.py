import torch

class CONSTANT():
    def __init__(self):
        self.epochs = 3000
        self.lr = 1e-5
        self.bs = 512
        self.nw = 8
        self.pm = True
        self.milestones = [25,50,100,200,400,800,1600]
        self.gamma = 0.5
        self.patience = 20
        self.device = torch.device('cuda:0')

        self.data_path = '../../NBA/data/res_img/'
        self.num_of_folder = 250
        self.num_of_folfer_kmeans = 500
        self.train_portion = 0.8
        self.valid_test_portion = 0.1
        '''
        For example, If the above numbers are 250, 500, 0.8, 0.1
        Then the indexes of used folders are:
        Train: 0 ~ 200
        Valid: 200 ~ 250
        Test: 225 ~ 250
        K-means Clustering: 250 ~ 500
        Maximum is 632 folders
        '''

        self.hidden_dims = [2048, 512, 128]
        self.in_size = 4096
        self.latent_size = 32

        