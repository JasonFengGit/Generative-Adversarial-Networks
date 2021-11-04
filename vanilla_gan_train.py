import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.variable import Variable
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
import imageio
from tqdm import tqdm
from vanilla_gan_model import Generator, Discriminator
import numpy as np
from matplotlib import pyplot as plt

def noise(len_img, n_z):
    return torch.randn(len_img, n_z).to(device)

def final_results(imgs, losses):
    # save imgs as gif
    imgs = [np.array(transforms.ToPILImage()(img)) for img in imgs]
    imageio.mimsave('outputs/generator_results.gif', imgs)

    # show trend of losses
    plt.figure()
    plt.plot(losses[0], label='Generator loss')
    plt.plot(losses[1], label='Discriminator Loss')
    plt.legend()
    plt.savefig('outputs/loss.png')

batch_size = 512
num_epochs = 500
sample_size = 64
n_z = 128 # noise vector size
k = 1 # discriminator's k in algorithm 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# prepare training data
training_data = MNIST( 
    root='data/',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,)),
    ])
)
dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

# initialize Generator and Discriminator
generator = Generator(device, n_z).to(device)
discriminator = Discriminator(device).to(device)

# optimizers for G and D
g_adam = optim.Adam(generator.parameters(), lr=0.0002)
d_adam = optim.Adam(discriminator.parameters(), lr=0.0002)

# loss function
loss_func = nn.BCELoss()
epoch_g_loss, epoch_d_loss = 0, 0
noise_for_display = noise(sample_size, n_z)
imgs = []
losses = [[], []] # g_losses, d_losses

epochs = tqdm(range(num_epochs))

for epoch in epochs:
    epoch_g_loss, epoch_d_loss = 0, 0
    for data in dataloader:
        image, _ = data # only need the image
        image = image.to(device)
        for step in range(k):
            Z = generator(noise(len(image), n_z)).detach()
            X = image
            epoch_d_loss += discriminator.backprop(loss_func, d_adam, X, Z)
        Z = generator(noise(len(image), n_z))
        epoch_g_loss += generator.backprop(loss_func, g_adam, Z, discriminator)
    epoch_g_loss = epoch_g_loss / len(dataloader)
    epoch_d_loss = epoch_d_loss / len(dataloader)
    losses[0].append(epoch_g_loss)
    losses[1].append(epoch_d_loss)
    
    epochs.set_description("epoch_g_loss: {:.5f}, epoch_d_loss: {:.5f}".format(epoch_g_loss, epoch_d_loss))
    if epoch % 10 == 0: # save imgs for every 10 epoch
        generated_img = generator(noise_for_display).cpu().detach()
        generated_img = make_grid(generated_img)
        save_image(generated_img, "outputs/img_{}.png".format(epoch))
        imgs.append(generated_img)
    
final_results(imgs, losses)