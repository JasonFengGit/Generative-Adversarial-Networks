import torch
import torch.nn as nn

# Generator model
class Generator(nn.Module):
    def __init__(self, device, n_z=128):
        super(Generator, self).__init__()
        self.device = device
        self.hidden_size = 1024 if self.device == 'cuda' else 128
        self.n_z = n_z
        self.mlp = nn.Sequential(
            nn.Linear(self.n_z, self.hidden_size),
            nn.LeakyReLU(0.2), #Leaky ReLU

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2),

            nn.Linear(self.hidden_size, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.mlp(x).view(-1, 1, 28, 28)

    def backprop(self, loss_func, optimizer, Z, discriminator):
        optimizer.zero_grad()
        output = discriminator(Z)
        loss = loss_func(output, torch.ones(Z.size(0), 1, device=self.device))
        loss.backward()
        optimizer.step()
        return loss


# Discriminator model
class Discriminator(nn.Module):
    def __init__(self, device):
        super(Discriminator, self).__init__()
        self.device = device
        self.hidden_size = 1024 if self.device == 'cuda' else 128
        self.n_input = 784
        self.mlp = nn.Sequential(
            nn.Linear(self.n_input, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.mlp(x.view(-1, 784))

    def backprop(self, loss_func, optimizer, X, Z):
        optimizer.zero_grad()
        loss_x = loss_func(self.forward(X), torch.ones(X.size(0), 1, device=self.device))
        loss_z = loss_func(self.forward(Z), torch.zeros(Z.size(0), 1, device=self.device))

        loss_x.backward()
        loss_z.backward()
        optimizer.step()

        return loss_z + loss_x