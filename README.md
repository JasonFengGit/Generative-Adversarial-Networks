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

TODO

## Reference
- [GPAM+14] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil
Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial nets. Advances in
neural information processing systems, 27, 2014.
- [MALK19] Sudipto Mukherjee, Himanshu Asnani, Eugene Lin, and Sreeram Kannan. Clustergan:
Latent space clustering in generative adversarial networks. In Proceedings of the AAAI
Conference on Artificial Intelligence, volume 33, pages 4610–4617, 2019.
