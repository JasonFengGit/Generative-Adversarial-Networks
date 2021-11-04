import torch
import torch.nn as nn
import torch.nn.functional as F

class Reshape(nn.Module):
    def __init__(self, shape):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

class Generator(nn.Module):
    def __init__(self, z_dim, num_class):
        super(Generator, self).__init__()
        prod = 128 * 7 * 7
        self.net = nn.Sequential(
            nn.Linear(z_dim + num_class, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, prod),
            nn.BatchNorm1d(prod),
            nn.LeakyReLU(0.2),
        
            Reshape((128, 7, 7)),

            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid()
        )

        # init weights
    
    def forward(self, z):
        x = self.net(z)
        return x.view(-1, 1, 28, 28)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        prod = 128 * 5 * 5
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2),
            
            Reshape((prod,)),
            
            nn.Linear(prod, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )

        # init weights

    def forward(self, x):
        return self.net(x)

class Encoder(nn.Module):
    def __init__(self, z_dim, num_class):
        super(Encoder, self).__init__()
        prod = 128 * 5 * 5
        self.z_dim = z_dim
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2),
            
            Reshape((prod,)),
            
            # Fully connected layers
            torch.nn.Linear(prod, 1024),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(1024, z_dim + num_class)
        )
    
    def forward(self, x):
        z = self.net(x)
        z = z.view(z.shape[0], -1)
        z_n = z[:, :self.z_dim]
        z_class_raw = z[:, self.z_dim:]
        z_class = F.softmax(z_class_raw, dim=1)
        return z_n, z_class, z_class_raw  