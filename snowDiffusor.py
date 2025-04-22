import os, atexit, argparse, shutil, re, cv2, torch
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

        # Initialize pipeline and the generator/discriminator
        self.pipe = pipeline(self.path, self.resolution)
        self.loss = []

        # Create a noise schedule to follow
        self.schedule_noise()

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

    def load(self, data_folder):
        transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3),  # Scale to [-1, 1], common in diffusion
        ])

        dataset = ImageFolder(root = data_folder, transform = transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
        return dataloader

    def forward(self, x, t):
        skip_connections = []
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = F.avg_pool2d(x, 2)

        for up in self.ups:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
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
    
    def train(self, model, dataloader, optimizer, device):
        model.train()
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            for x in dataloader:
                x = x.to(device)
                t = torch.randint(0, self.t, (x.size(0),), device=device).long()
                noise = torch.randn_like(x)
                x_noisy = self.q_sample(x, t, noise)
                noise_pred = model(x_noisy, t)
                loss = F.mse_loss(noise_pred, noise)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    @torch.no_grad()
    def sample(self, model, image_size, batch_size=16, channels=3):
        model.eval()
        x = torch.randn((batch_size, channels, image_size, image_size)).to(device)

        for t in reversed(range(self.t)):
            t_batch = torch.full((batch_size,), t, dtype=torch.long)
            predicted_noise = model(x, t_batch)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alphas_cumprod[t]
            alpha_cumprod_t_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0)

            coef1 = (1 / torch.sqrt(alpha_t)) * (x - beta_t / torch.sqrt(1 - alpha_cumprod_t) * predicted_noise)
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
    