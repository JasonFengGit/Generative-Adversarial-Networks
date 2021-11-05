# Generative Adversarial Networks(GANs)
1. [**Vanilla GAN**](https://github.com/JasonFengGit/Generative-Adversarial-Networks#vanilla-gan)
2. [**ClusterGAN**](https://github.com/JasonFengGit/Generative-Adversarial-Networks#clustergan)

## Vanilla GAN

### Model Structure

<p align="center">
  <img src="https://raw.githubusercontent.com/JasonFengGit/Generative-Adversarial-Networks/3139e2dcd4cd21bbf7e768db0c311a111e3ebab7/imgs/vanilla_GAN.svg" alt/>
</p>

#### Final Generator Structure

- A MLP with 2 hidden layers of hidden_size=1024

- A LeakyReLU of slope=0.2 is used for each layer for activation

- ```python
  nn.Sequential(
      nn.Linear(self.n_z, self.hidden_size),
      nn.LeakyReLU(0.2),
  
      nn.Linear(self.hidden_size, self.hidden_size),
      nn.LeakyReLU(0.2),
  
      nn.Linear(self.hidden_size, self.hidden_size),
      nn.LeakyReLU(0.2),
  
      nn.Linear(self.hidden_size, 784),
      nn.Tanh(),
  )
  ```

#### Final Discriminator Structure

- A MLP with 2 hidden layers of hidden_size=1024

- A LeakyReLU of slope=0.2 is used for each layer for activation

- A Dropout of rate=0.3 is used for each layer

- ```python
  nn.Sequential(
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
  ```

> I tried several settings of Generator and Discriminator for 200 epochs to determine the final structures, including activations, Dropout rate, and # of hidden layers. 

### Training Losses

> *Trained for 500 epochs on a GPU*


<p align="center">
  <img src="https://github.com/JasonFengGit/Generative-Adversarial-Networks/blob/master/imgs/vanilla_gan_losses.png?raw=true" alt/>
</p>

### Generated Results During Training

> *Trained for 500 epochs on a GPU*

<p align="center">
  <img height="450px" src="https://github.com/JasonFengGit/Generative-Adversarial-Networks/blob/master/imgs/vanilla_gan_results.gif?raw=true" alt/>
</p>

## ClusterGAN

### Model Structure

![](https://github.com/JasonFengGit/Generative-Adversarial-Networks/blob/master/imgs/cluster_gan.png?raw=true)

> Adding an Encoder: X - > Z

#### Final Generator Structure

```python
nn.Sequential(
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
```

#### Final Encoder Structure

```python
nn.Sequential(
    nn.Conv2d(1, 64, 4, stride=2, bias=True),
    nn.LeakyReLU(0.2),
    nn.Conv2d(64, 128, 4, stride=2, bias=True),
    nn.LeakyReLU(0.2),

    Reshape((prod,)),

    torch.nn.Linear(prod, 1024),
    nn.LeakyReLU(0.2),
    torch.nn.Linear(1024, z_dim + num_class)
)
```

#### Final Discriminator Structure

```python
nn.Sequential(
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
```

### Training Losses

> *Trained for 500 epochs on a GPU*
>
> *Generator&Encoder seem to be not strong enough*

![](https://github.com/JasonFengGit/Generative-Adversarial-Networks/blob/master/imgs/cluster_gan_loss.png?raw=true)

![](https://github.com/JasonFengGit/Generative-Adversarial-Networks/blob/master/imgs/enc_mse_loss.png?raw=true)

![](https://github.com/JasonFengGit/Generative-Adversarial-Networks/blob/master/imgs/enc_cross_entropy_loss.png?raw=true)

### Generated Results During Training

> *Trained for 500 epochs on a GPU*
>
> *For Cluster GAN, I make the noise for display ordered by 10 classes*

![](https://github.com/JasonFengGit/Generative-Adversarial-Networks/blob/master/imgs/cluster_gan_results.gif?raw=true)



## Reference
- [GPAM+14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil
Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in
neural information processing systems, 27, 2014.
- [MALK19] Sudipto Mukherjee, Himanshu Asnani, Eugene Lin, and Sreeram Kannan. Clustergan:
Latent space clustering in generative adversarial networks. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 33, pages 4610–4617, 2019.
- https://github.com/eriklindernoren/PyTorch-GAN from PapersWithCode
