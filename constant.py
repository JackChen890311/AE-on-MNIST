import torch

class CONSTANT():
    def __init__(self):
        self.epochs = 1000
        self.lr = 5e-3
        self.bs = 64
        self.nw = 8
        self.pm = True
        self.milestones = [50, 100, 500, 5000]
        self.gamma = 0.5
        self.patience = 100
        self.device = torch.device('cuda:0')

        self.data_path = 'data/training'
        self.data_path_test = 'data/testing'

        self.hidden_dims = [8192, 2048, 512, 128, 32]
        # self.hidden_dims = [4096, 1024, 256, 64, 16]
        self.image_dim = 28
        self.in_size = 784
        self.latent_size = 2

        