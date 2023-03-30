import torch

class CONSTANT():
    def __init__(self):
        self.epochs = 500
        self.lr = 1e-3
        self.bs = 32
        self.nw = 8
        self.pm = True
        self.milestones = [50,500,5000]
        self.gamma = 0.5
        self.patience = 20
        self.device = torch.device('cuda:0')

        self.data_path = 'data/testing'
        self.data_path_test = 'data/testing'

        self.hidden_dims = [392, 196, 98]
        self.image_dim = 28
        self.in_size = 784
        self.latent_size = 8

        