import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, channels_num, n_latent, iterations_num=10):
        super().__init__()
        self.n_latent = n_latent
        self.iterations_num = iterations_num
        self.channels_num = channels_num
        self.device = torch.device('cuda')
        self.encoder = nn.Sequential(
            # nn.BatchNorm2d(c),
            nn.Conv2d(channels_num, 256, kernel_size=1, stride=1, padding=0),  # 32, 16, 16
            # nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 64, kernel_size=1, stride=1, padding=0),  # 32, 8, 8
            # nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 16, kernel_size=1, stride=1, padding=0),  # 32, 8, 8
            # nn.BatchNorm2d(16),
            nn.LeakyReLU(),
        ).to(self.device)
        self.z_mean = nn.Conv2d(16, self.n_latent, kernel_size=1, stride=1, padding=0).to(self.device)
        self.z_var = nn.Conv2d(16, self.n_latent, kernel_size=1, stride=1, padding=0).to(self.device)
        self.decoder = nn.Sequential(
            nn.Conv2d(self.n_latent, 16, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(16, 64, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(256, channels_num, kernel_size=1, stride=1, padding=0),
            # CenterCrop(h,w),
            # nn.Sigmoid()
        ).to(self.device)

    def sample_z(self, mean, logvar):
        stddev = torch.exp(0.5 * logvar) + 1e-6
        noise = torch.randn(stddev.size(), device=stddev.device)
        return 0*(noise * stddev) + mean

    def encode(self, x):
        x = self.encoder(x)
        mean = self.z_mean(x)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        out = self.decoder(z)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar

def vae_loss(output, input, mean, logvar, indices_pos):
    recon_loss = torch.sum( (input[:, :, indices_pos[:, 0], indices_pos[:, 1]] - output[:, :, indices_pos[:, 0], indices_pos[:, 1]])**2 ) / indices_pos.shape[0]
    kl_loss = 0.0*torch.mean(0.5 * torch.sum(
        torch.exp(logvar[:, :, indices_pos[:, 0], indices_pos[:, 1]]) + mean[:, :, indices_pos[:, 0], indices_pos[:, 1]] ** 2 - 1. - logvar[:, :, indices_pos[:, 0], indices_pos[:, 1]], 1))
    return recon_loss + kl_loss


# Loss
# Function
#
# The
# VAE
# loss
# function
# combines
# reconstruction
# loss(e.g.Cross
# Entropy, MSE) with KL divergence.
#
# def vae_loss(output, input, mean, logvar, loss_func):
#     recon_loss = loss_func(output, input)
#     kl_loss = torch.mean(0.5 * torch.sum(
#         torch.exp(logvar) + mean ** 2 - 1. - logvar, 1))
#     return recon_loss + kl_loss
#
# Model
#
# An
# example
# implementation in PyTorch
# of
# a
# Convolutional
# Variational
# Autoencoder.
#
# class VAE(nn.Module):
#     def __init__(self, in_shape, n_latent):
#         super().__init__()
#         self.in_shape = in_shape
#         self.n_latent = n_latent
#         c, h, w = in_shape
#         self.z_dim = h // 2 ** 2  # receptive field downsampled 2 times
#         self.encoder = nn.Sequential(
#             nn.BatchNorm2d(c),
#             nn.Conv2d(c, 32, kernel_size=4, stride=2, padding=1),  # 32, 16, 16
#             nn.BatchNorm2d(32),
#             nn.LeakyReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32, 8, 8
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(),
#         )
#         self.z_mean = nn.Linear(64 * self.z_dim ** 2, n_latent)
#         self.z_var = nn.Linear(64 * self.z_dim ** 2, n_latent)
#         self.z_develop = nn.Linear(n_latent, 64 * self.z_dim ** 2)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1),
#             CenterCrop(h, w),
#             nn.Sigmoid()
#         )
#
#     def sample_z(self, mean, logvar):
#         stddev = torch.exp(0.5 * logvar)
#         noise = Variable(torch.randn(stddev.size()))
#         return (noise * stddev) + mean
#
#     def encode(self, x):
#         x = self.encoder(x)
#         x = x.view(x.size(0), -1)
#         mean = self.z_mean(x)
#         var = self.z_var(x)
#         return mean, var
#
#     def decode(self, z):
#         out = self.z_develop(z)
#         out = out.view(z.size(0), 64, self.z_dim, self.z_dim)
#         out = self.decoder(out)
#         return out
#
#     def forward(self, x):
#         mean, logvar = self.encode(x)
#         z = self.sample_z(mean, logvar)
#         out = self.decode(z)
#         return out, mean, logvar
#
# Training
#
# def train(model, loader, loss_func, optimizer):
#     model.train()
#     for inputs, _ in loader:
#         inputs = Variable(inputs)
#
#         output, mean, logvar = model(inputs)
#         loss = vae_loss(output, inputs, mean, logvar, loss_func)
#
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#
