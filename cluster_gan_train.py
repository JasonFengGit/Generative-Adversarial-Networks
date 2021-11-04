
import torch

from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image

from itertools import chain as ichain
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
import imageio

from cluster_gan_model import Generator, Encoder, Discriminator


def sample_z(shape=100, z_dim=30, n_class=10, device="cuda", display=False):
    z_n = torch.randn(shape, z_dim).to(device)
    z_c_val = torch.empty(shape, dtype=torch.long).random_(n_class).to(device)
    if display :
        z_c_val = torch.tensor(list(range(10))*10).to(device)
    z_c = torch.zeros(shape, n_class, device=device).scatter_(1, z_c_val.unsqueeze(1), 1.)

    return z_n, z_c, z_c_val 

def final_results(imgs, losses):
    # save imgs as gif
    imgs = [np.array(transforms.ToPILImage()(img)) for img in imgs]
    imageio.mimsave('cluster_outputs/generator_results.gif', imgs)

    # show trend of losses
    plt.figure(1)
    plt.plot(losses[0], label='Generator-Encoder loss')
    plt.plot(losses[1], label='Discriminator Loss')
    plt.legend()
    plt.savefig('cluster_outputs/loss.png')

    plt.figure(2)
    plt.plot(losses[2], label='enc_mse_loss')
    plt.legend()
    plt.savefig('cluster_outputs/enc_mse_loss.png')

    plt.figure(3)
    plt.plot(losses[3], label='enc_cross_entropy_loss')
    plt.legend()
    plt.savefig('cluster_outputs/enc_cross_entropy_loss.png')


n_epochs = 200
batch_size = 100
lr = 0.0001
k = 5

img_size = 28
z_dim = 30
n_class = 10
beta_n = 10
beta_c = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

generator = Generator(z_dim, n_class).to(device)
encoder = Encoder(z_dim, n_class).to(device)
discriminator = Discriminator().to(device)

bce_loss = torch.nn.BCELoss()
ce_loss = torch.nn.CrossEntropyLoss()
mse_loss = torch.nn.MSELoss()

training_data = FashionMNIST(
    root="data/",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    )
)

dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

ge_chain = ichain(generator.parameters(), encoder.parameters())
optimizer_GE = torch.optim.Adam(
    ge_chain, lr=lr, betas=(0.5, 0.9), weight_decay=2.5*1e-5)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=lr, betas=(0.5, 0.9))
losses = [[], [], [], []] # ge_losses, d_losses
imgs = []
sample_for_display = sample_z(shape=100, z_dim=z_dim, n_class=n_class, device=device)

epochs = tqdm(range(n_epochs))
for epoch in epochs:
    epoch_ge_loss, epoch_d_loss = 0, 0
    for b_i, data in enumerate(dataloader):
        generator.train()
        encoder.train()
        
        image, _ = data  # only need the image
        real_imgs = image.to(device)
        z_n, z_c, z_c_val = sample_z(shape=real_imgs.shape[0], z_dim=z_dim, n_class=n_class, device=device)
        
        # G & E
        if b_i % k == 0: 
            optimizer_GE.zero_grad()
            fake_imgs = generator(torch.cat((z_n, z_c), 1))
            D_fake = discriminator(fake_imgs)
            enc_z_n, enc_z_class, enc_z_class_raw = encoder(fake_imgs)
            zn_loss = mse_loss(enc_z_n, z_n)
            zc_loss = ce_loss(enc_z_class_raw, z_c_val)
            d_loss = bce_loss(D_fake, torch.ones(
                D_fake.size(0), 1, device=device))
            ge_loss = d_loss + beta_n * zn_loss + beta_c * zc_loss

            ge_loss.backward(retain_graph=True)
            optimizer_GE.step()
            epoch_ge_loss += ge_loss.item()

        # D
        optimizer_D.zero_grad()
        fake_imgs = generator(torch.cat((z_n, z_c), 1))
        D_fake = discriminator(fake_imgs)
        D_real = discriminator(real_imgs)
        
        real_loss = bce_loss(D_real, torch.ones(
            D_real.size(0), 1, device=device))
        fake_loss = bce_loss(D_fake, torch.zeros(
            D_real.size(0), 1, device=device))
        real_loss.backward()
        fake_loss.backward()
        optimizer_D.step()
        epoch_d_loss += (real_loss + fake_loss).item()

    epoch_ge_loss = epoch_ge_loss / len(dataloader)
    epoch_d_loss = epoch_d_loss / len(dataloader)
    losses[0].append(epoch_ge_loss)
    losses[1].append(epoch_d_loss)
    epochs.set_description("epoch_ge_loss: {:.5f}, epoch_d_loss: {:.5f}".format(epoch_ge_loss, epoch_d_loss))
    generator.eval()
    encoder.eval()
    if epoch % 10 == 0: # save imgs for every 10 epoch
        generated_img = generator(torch.cat((sample_for_display[0], sample_for_display[1]), 1)).cpu().detach()
        generated_img = make_grid(generated_img)
        save_image(generated_img, "cluster_outputs/img_{}.png".format(epoch))
        imgs.append(generated_img)

    n_samp = 100
    zn_samp, zc_samp, zc_samp_val = sample_z(shape=100, z_dim=z_dim, n_class=n_class, device=device)
    x_samp = generator(torch.cat((zn_samp, zc_samp), 1))

    zn_enc, zc_enc, zc_enc_val = encoder(x_samp)

    lat_mse_loss = mse_loss(zn_enc, zn_samp)
    lat_ce_loss = ce_loss(zc_enc_val, zc_samp_val)

    losses[2].append(lat_mse_loss.item())
    losses[3].append(lat_ce_loss.item())

final_results(imgs, losses)