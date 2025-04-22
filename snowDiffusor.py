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

    def __init__(self, path = 'snowdiffusor/', resolution = (1024, 1024), new = False):
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
        self.batch_size = 8 # Number of images to load in per training batch
        self.epochs = 50 # Number of training epochs per training batch
        self.synthetics = 10 # Number of synthetic images to generate after training

        self.t = 1000  # total diffusion steps

        # Initialize data loader
        self.dataloader = self.load(resolution)

        # Initialize pipeline and the generator/discriminator
        self.pipe = pipeline(self.path, self.resolution)
        self.loss = []

        # Create a noise schedule to follow
        self.schedule_noise()

        self.build()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        atexit.register(self.save_model)
        return

    def build(self, in_channels = 3, out_channels = 3, features = [64, 128, 256, 512, 1024, 1024]):
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        # Down sampling
        for feature in features:
            self.downs.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature, feature, 3, padding=1),
                nn.ReLU(),
            ))
            in_channels = feature
        
        # Up sampling
        for feature in reversed(features):
            self.ups.append(nn.Sequential(
                nn.Conv2d(in_channels, feature, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(feature, feature, 3, padding=1),
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
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
        return dataloader

    def forward(self, x, t):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.avg_pool2d(x, 2)

        for up in self.ups:
            x = F.interpolate(x, scale_factor = 2, mode = 'nearest')
            skip_x = skip_connections.pop()
            
            x = torch.cat((x, skip_x), dim=1)
            x = up(x)

        return self.final(x)
        
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
    
    def train_model(self, device = 'cuda'):
        self.train()  # set model to training mode
        scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.epochs):
            for batch in self.dataloader:
                x, _ = batch  # if using ImageFolder, this returns (image, label)
                x = x.to(device)
                t = torch.randint(0, self.t, (x.size(0), ), device = device).long()
                noise = torch.randn_like(x)
                x_noisy = self.q_sample(x, t, noise)

                with torch.cuda.amp.autocast():
                    noise_pred = self(x_noisy, t)  # 🔥 use self for forward pass
                    loss = F.mse_loss(noise_pred, noise)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()

            print(f"[Epoch {epoch+1}/{self.epochs}] Loss: {loss.item():.4f}")
            self.loss.append(loss.item())

    @torch.no_grad()
    def generate(self, batch_size=16, channels=3, device='cuda'):
        self.eval()

        image_size = self.resolution

        x = torch.randn((batch_size, channels, image_size, image_size)).to(device)

        for t in reversed(range(self.t)):
            t_batch = torch.full((batch_size,), t, dtype = torch.long, device=device)

            predicted_noise = self(x, t_batch)

            beta_t = self.betas[t].to(device)
            alpha_t = self.alphas[t].to(device)
            alpha_cumprod_t = self.alphas_cumprod[t].to(device)
            alpha_cumprod_t_prev = (
                self.alphas_cumprod[t - 1].to(device) if t > 0 else torch.tensor(1.0, device = device)
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
