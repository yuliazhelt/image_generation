import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_size=128, depths=64):
        super(Generator, self).__init__()

        self.depths = depths
        self.latent_size = latent_size


        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=latent_size, out_channels=depths * 8, kernel_size=4, stride=1, padding=0, bias=False),

            nn.BatchNorm2d(num_features=depths * 8),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=depths * 8, out_channels=depths * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depths * 4),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=depths * 4, out_channels=depths * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depths * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(depths * 2, out_channels=depths, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depths),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=depths, out_channels=3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.conv(x)
    

class Discriminator(nn.Module):
    def __init__(self, depths=64):
        super(Discriminator, self).__init__()

        self.depths = depths

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=depths, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depths),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=depths, out_channels=depths * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depths * 2),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=depths * 2, out_channels=depths * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depths * 4),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(depths * 4, out_channels=depths * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=depths * 8),
            nn.LeakyReLU(negative_slope=0.2),

            nn.Conv2d(in_channels=depths * 8, out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.conv(x)