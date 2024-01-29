import torch
from torch import nn
from torch.nn import functional as F

VAE_ENCODING_DIM = 64
IMG_HEIGHT, IMG_WIDTH = 24, 24

# Define the Variational Encoder
class VarEncoder(nn.Module):
    def __init__(self, encoding_dim):
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        super(VarEncoder, self).__init__()
        # TODO: implement the encoder
        
        self.encoding_dim = encoding_dim
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)                                          # (H, W) -> (H/2, W/2)
        self.linear_mu = nn.Linear(128 * (IMG_HEIGHT // 4) * (IMG_WIDTH // 4), encoding_dim)
        self.linear_var = nn.Linear(128 * (IMG_HEIGHT // 4) * (IMG_WIDTH // 4), encoding_dim)  


    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the forward pass
        x = F.relu(self.pool(self.conv1(x)))  # (3, H, W) -> (32, H/2, W/2)
        x = F.relu(self.pool(self.conv2(x)))  # (32, H/2, W/2) -> (64, H/4, W/4)
        x = F.relu(self.conv3(x))             # (64, H/4, W/4) -> (128, H/4, W/4)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)             # Flatten to (Batch_size, 128 * H/4 * W/4)

        # Pass the flattened output through the fully connected layer
        mu = self.linear_mu(x)
        log_var = self.linear_var(x)
        
        return mu, log_var

# Define the Decoder
class VarDecoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarDecoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        # TODO: implement the decoder

        self.encoding_dim = encoding_dim
        self.linear_back = nn.Linear(encoding_dim, 128 * (IMG_HEIGHT // 4) * (IMG_WIDTH // 4))
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')


    def forward(self, v):
        '''
        v: latent vector, dim: (Batch_size, encoding_dim)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        '''
        
        # TODO: implement the forward pass
        v = self.linear_back(v)

        # Reconstruct to (128, H/4, W/4)
        v = v.view(-1, 128, IMG_HEIGHT // 4, IMG_WIDTH // 4)

        x = F.relu(self.deconv1(v))                             # (128, H/4, W/4) -> (64, H/4, W/4)
        x = F.relu(self.upsample(self.deconv2(x)))              # (64, H/4, W/4)  -> (32, H/2, W/2)
        x = torch.sigmoid(self.upsample(self.deconv3(x)))       # (32, H/2, W/2)  -> (3, H, W) in [0,1]

        return x

# Define the Variational Autoencoder
class VarAutoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(VarAutoencoder, self).__init__()
        self.encoder = VarEncoder(encoding_dim)
        self.decoder = VarDecoder(encoding_dim)

    @property
    def name(self):
        return "VAE"

    def reparameterize(self, mu, log_var):
        '''
        mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        return v: sampled latent vector, dim: (Batch_size, encoding_dim)
        '''
        
        # TODO: implement the reparameterization trick to sample v
        sigma = torch.exp(0.5 * log_var)
        
        epsilon = torch.randn_like(sigma)
        
        v = mu + epsilon * sigma

        return v
        
    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return x: reconstructed images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return mu: mean of the distribution, dim: (Batch_size, encoding_dim)
        return log_var: log of the variance of the distribution, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the forward pass
        mu, log_var = self.encoder(x)

        v = self.reparameterize(mu, log_var)

        x = self.decoder(v)
    
        return x, mu, log_var

# Loss Function
def VAE_loss_function(outputs, images):
    '''
    outputs: (x, mu, log_var)
    images: input/original images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
    return loss: the loss value, dim: (1)
    '''
    # TODO: implement the loss function for VAE
    x_reconstructed, mu, log_var = outputs
    
    # Reconstruction loss (mean squared error)
    reconstruction_loss = F.mse_loss(x_reconstructed, images, reduction='sum')
    
    # KL divergence regularization term
    kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    # Total VAE loss
    loss = reconstruction_loss + kl_divergence
    
    return loss


