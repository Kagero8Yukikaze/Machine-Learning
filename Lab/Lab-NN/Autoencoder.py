import torch
import torch.nn as nn
import torch.nn.functional as F

AE_ENCODING_DIM = 64
IMG_WIDTH = 24
IMG_HEIGHT = 24

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Encoder, self).__init__()
        '''
        encoding_dim: the dimension of the latent vector produced by the encoder
        '''
        # TODO: implement the encoder

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)                                          # (H, W) -> (H/2, W/2)
        self.encoding_dim = encoding_dim
        self.linear_to = nn.Linear(128 * (IMG_HEIGHT // 4) * (IMG_WIDTH // 4), encoding_dim)    # fully-connected layer
        

    def forward(self, x):
        '''
        x: input images, dim: (Batch_size, 3, IMG_WIDTH, IMG_HEIGHT)
        return v: latent vector, dim: (Batch_size, encoding_dim)
        '''
        # TODO: implement the forward pass    
        x = F.relu(self.pool(self.conv1(x)))  # (3, H, W) -> (32, H/2, W/2)
        x = F.relu(self.pool(self.conv2(x)))  # (32, H/2, W/2) -> (64, H/4, W/4)
        x = F.relu(self.conv3(x))             # (64, H/4, W/4) -> (128, H/4, W/4)

        # Flatten the output for the fully connected layer
        x = x.view(x.size(0), -1)             # Flatten to (Batch_size, 128 * H/4 * W/4)

        # Pass the flattened output through the fully connected layer
        v = self.linear_to(x)                 # Latent vector with size (Batch_size, encoding_dim)

        return v


# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Decoder, self).__init__()
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


# Combine the Encoder and Decoder to make the autoencoder
class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(encoding_dim)
        self.decoder = Decoder(encoding_dim)

    def forward(self, x):
        v = self.encoder(x)
        x = self.decoder(v)
        return x
    
    @property
    def name(self):
        return "AE"

