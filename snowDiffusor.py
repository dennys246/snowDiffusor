import os, atexit, argparse, shutil, re, cv2, torch, argparse
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from pipeline import pipeline
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms


class snowDiffusor(nn.Module):

    """
    This class will builds and trains a diffusor model using pytorch to analyze
    snowpack data collected during the 2024-2025 Winter season.
 
    Attributes:
        path (str) - Path to save the models or containing a pre-trained model to load
        resolution (set of int) - Resolution to resample snow images too
        new (bool) - Boolean whether to reinitialize models weights before training
    
    Methods:
        build() - Build the diffusor architecture
        load() - Load in snowpack data and format for training
        train() - Train the diffusor using the loaded data
        plot_history() - Plot the diffusor history
    
    """

    def __init__(self, path = 'snowdiffusor/', resolution = (512, 512), new = False):
        super().__init__()

        self.data_directory = '/Users/dennyschaedig/Scripts/AvalancheAI/snow-profiles/magnified-profiles'

        # Check if model folder rebuilt requested
        if new and os.path.exists(path):
            # Double check with user model is to be rebuilt
            response = input("New model requested, are you sure you would like to delete the current model? (y/n)\n")
            if response == 'y': # Rebuild
                shutil.rmtree(path)
            else: # Cancel rebuild
                print("Model rebuilding canceled, reverting to loading old pre-trained model")
                new = False

        self.path = path # Handle the model path
        if not os.path.exists(self.path):
            os.makedirs(f"{self.path}synthetic_images/", exist_ok = True)

        # Check if model could be loaded
        if path and new == False:
            if os.path.exists(f"{self.path}diffusor.keras"):
                self.load_model()
        
        # Define training runtime parameters
        self.resolution = resolution # Resolution to resample images to
        self.batch_size = 1 # Number of images to load in per training batch
        self.epochs = 50 # Number of training epochs per training batch
        self.synthetics = 10 # Number of synthetic images to generate after training

        self.t = 1000  # total diffusion steps

        # Initialize data loader
        self.dataloader = self.load(resolution)

        # Initialize pipeline and the generator/discriminator
        #self.pipe = pipeline(self.path, self.resolution)
        self.loss = []

        # Create a noise schedule to follow
        self.schedule_noise()

        self.build()
        self.optimizer = torch.optim.Adam(self.parameters(), lr = 1e-4)

        torch.no_grad()

        atexit.register(self.save_model)
        device = torch.device("mps")  # Use MPS device
        self.to(device)
        return

    def build(self, in_channels = 3, out_channels = 3, features = [4, 8, 16, 32]):
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        def get_num_groups(num_channels, divisor = 32):
            for g in reversed(range(1, divisor + 1)):
                if num_channels % g == 0:
                    return g
            return 1  # Fallback to 1 group

        # Build encoder through downsampling
        for feature in reversed(features):
            ch_count = feature
            num_groups = get_num_groups(ch_count)

            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, 3, padding=1),
                nn.GroupNorm(num_groups, feature),
                nn.ReLU(),
                nn.Conv2d(feature, feature, 3, padding=1),
                nn.GroupNorm(num_groups, feature),
                nn.ReLU(),
            ))

            in_channels = feature

        # Bottleneck (middle block)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(get_num_groups(in_channels * 2), in_channels * 2),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1),
            nn.GroupNorm(get_num_groups(in_channels), in_channels),
            nn.ReLU(),
        )

        # Build the upsampling blocks
        for feature in features:
            ch_count = feature
            num_groups = get_num_groups(ch_count)

            self.ups.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(feature * 2, feature, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, feature),
                nn.ReLU(),
                nn.Conv2d(feature, feature, kernel_size=3, padding=1),
                nn.GroupNorm(num_groups, feature),
                nn.ReLU(),
            ))

            in_channels = feature

        self.final = nn.Conv2d(in_channels, out_channels, 1)


    def load(self, resolution):
        transform = transforms.Compose([
            transforms.Resize(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # Scale to [-1, 1], common in diffusion
        ])

        dataset = ImageFolder(root = self.data_directory, transform = transform)
        dataloader = DataLoader(dataset, 
                                batch_size=4,
                                shuffle=True, 
                                num_workers=0) # Set workers to 0 due to weird M2 MPS behavior
        return dataloader

    def forward(self, x, t):
        skips = []  # Save features for skip connections
        device = torch.device("mps")  # Use MPS device

        # Downsample through the encoder
        for down in self.downs:
            x = down(x)
            x.to(device)
            skips.append(x)

        # Bottleneck (middle block)
        x = self.bottleneck(x)
        x.to(device)
        print(f"Bottleneck output shape: {x.shape}")
        

        # Decode (Upsampling with skip connections)
        for idx in range(len(self.ups)):
            skip_connection = skips[-(idx + 1)]  # Get corresponding skip
            print(f"Skip connection {idx} shape: {skip_connection.shape}")
            print(f"x.device: {x.device}, skip_connection.device: {skip_connection.device}")

            # Resize skip_connection to match the spatial dimensions of x
            if x.shape[2:] != skip_connection.shape[2:]:
                skip_connection = F.interpolate(skip_connection, size=x.shape[2:], mode="bilinear", align_corners=False)

            # Ensure skip_connection channels match x's channels
            if skip_connection.shape[1] != x.shape[1]:
                skip_connection = self.adjust_skip_channels(skip_connection, x)

            # Concatenate skip connection with x
            x = torch.cat((x, skip_connection), dim=1)
            x.to(device)
            print(f"After concatenation: {x.shape}")

            # Apply upsampling block
            print(f"Before upsampling, x shape: {x.shape}")
            x = self.ups[idx](x)
            x.to(device)
            print(f"After upsampling, x shape: {x.shape}")

            # Before applying the next convolution layer, adjust channels if necessary
            x = self.adjust_channels_after_upsample(x, required_channels = 16)
            x.to(device)

        # Final output layer
        x = self.final(x)
        x.to(device)
        return 

    def adjust_skip_channels(self, skip_connection, x):
        device = x.device
        skip_connection = skip_connection.to(device)  # Ensure skip_connection is on the same device as x
        
        # Check the number of channels and adjust if necessary
        if skip_connection.shape[1] != x.shape[1]:
            print(f"Adjusting channels from {skip_connection.shape[1]} to {x.shape[1]}")
            # Use a 1x1 convolution to adjust channels
            skip_connection = nn.Conv2d(skip_connection.shape[1], x.shape[1], kernel_size=1).to(device)(skip_connection)
        
        return skip_connection
    
    def adjust_channels_after_upsample(self, x, required_channels):
        if x.shape[1] != required_channels:
            print(f"Adjusting channels from {x.shape[1]} to {required_channels}")
            # Apply a 1x1 convolution to adjust the number of channels
            x = nn.Conv2d(x.shape[1], required_channels, kernel_size=1)(x)
        return x
        
    def linear_beta_schedule(self, timesteps):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, timesteps)
    
    def schedule_noise(self):
        self.betas = self.linear_beta_schedule(self.t)
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod[t])[:, None, None, None]
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def train_model(self, _device = None):

        if _device is None:
            _device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        self.to(_device)  # Move model to the chosen device
        torch.autocast(_device)
        self.train()  # set model to training mode
        
        # Disable GradScaler/autocast unless CUDA is available
        use_amp = torch.cuda.is_available()

        if use_amp: # if CUDA available
            scaler = torch.cuda.amp.GradScaler() # Set up CUDA scaler

        for epoch in range(self.epochs):
            for batch in self.dataloader:
                x, _ = batch  # if using ImageFolder, this returns (image, label)
                x = x.to(_device)
                t = torch.randint(0, self.t, (x.size(0), ), device = _device).long()
                noise = torch.randn_like(x)
                x_noisy = self.q_sample(x, t, noise)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        noise_pred = self(x_noisy, t)
                        loss = F.mse_loss(noise_pred, noise)

                    self.optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    noise_pred = self(x_noisy, t)
                    loss = F.mse_loss(noise_pred, noise)

                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {loss.item():.4f}")
            self.loss.append(loss.item())

    @torch.no_grad()
    def generate(self, batch_size=16, channels=3, _device='mps'):
        self.eval()
        self.to(_device)

        x = torch.randn((batch_size, channels, self.resolution[0], self.resolution[1])).to(_device)

        for t in reversed(range(self.t)):
            t_batch = torch.full((batch_size,), t, dtype = torch.long, device= _device)

            predicted_noise = self(x, t_batch)

            beta_t = self.betas[t].to(_device)
            alpha_t = self.alphas[t].to(_device)
            alpha_cumprod_t = self.alphas_cumprod[t].to(_device)
            alpha_cumprod_t_prev = (
                self.alphas_cumprod[t - 1].to(_device) if t > 0 else torch.tensor(1.0, device = _device)
            )

            coef1 = (1 / torch.sqrt(alpha_t)) * (
                x - (beta_t / torch.sqrt(1 - alpha_cumprod_t)) * predicted_noise
            )

            if t > 0:
                noise = torch.randn_like(x)
                coef2 = torch.sqrt(1 - alpha_cumprod_t_prev)
                x = coef1 + coef2 * noise
            else:
                x = coef1

        return x
    
    def save_samples(self, samples, step=0):
        grid = torchvision.utils.make_grid(samples, normalize=True)
        torchvision.utils.save_image(grid, f"{self.path}/synthetic_images/sample_{step}.png")
    
    def save_model(self):
        torch.save(self.state_dict(), os.path.join(self.path, "diffusor.pth"))

    def load_model(self):
        self.load_state_dict(torch.load(os.path.join(self.path, "diffusor.pth")))
    
if __name__ == "__main__": # Add in command line functionality

    # Initialize the parser for accepting arugments into a command line call
    parser = argparse.ArgumentParser(description = "The snowDiffusor model is used to train a PyTorch based Diffusor model on a dataset of snow samples magnified on a crystal card. You can define how the model runs by the number of epochs, batch sizes and other parameters. You can also pass in a path to a pre-trained snowDiffusor to accomplish transfer learning on new Diffusion tasks!")

    # Add command-line arguments
    parser.add_argument('--path', type = str, default = "snowdiffusor/", help = "Path to a pre-trained model or directory to save results (defaults to snowgan/)")
    parser.add_argument('--resolution', type = set, default = (1024, 1024), help = 'Resolution to downsample images too (Default set to (1024, 1024))')
    parser.add_argument('--batches', type = int, default = None, help = 'Number of batches to run (Default to max available)')
    parser.add_argument('--batch-size', type = int, default = 8, help = 'Batch size (Defaults to 8)')
    parser.add_argument('--epochs', type = int, default = 100, help = 'Training epochs per image (Defaults to 100)')
    parser.add_argument('--new', type = bool, default = False, help = 'Whether to rebuild model from scratch (defaults to False)')

    # Parse the arguments
    args = parser.parse_args()

    # Create the snowGAN object with the parsed arguments
    snowdiff = snowDiffusor(path = args.path, resolution = args.resolution, new = args.new)
    snowdiff.batch_size = args.batch_size
    snowdiff.epochs = args.epochs

    # Train the model
    snowdiff.train()

    # Generate a final batch of images for viewing
    snowdiff.generate()
