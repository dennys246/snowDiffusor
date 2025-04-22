import os, cv2, atexit, shutil
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

class pipeline:

	def __init__(self, path, resolution):
		"""
		The pipeline object serves as a loading and preprocessing pipeline for the snowGAN to 
		pass in trainable data from pictures of snowpack and snow cores. 

		Class attributes:
			path (str) - Path to load and save details about image loading history
			resolution (set of int) - 2D resolution to resample images too

		Class methods:
			load_batch() - Load in a batch of images for training the snowgan
			load_image() - Load and preprocess a single image
			resample() - Resample a preprocessed image by flipping it
			save_loaded() - Save the state and history of the pipeline
			read_loaded() - Load the state and history of the pipeline
		"""
		# Assign passed in class attributes
		self.path = path
		self.profile_path = '/Users/dennyschaedig/Scripts/AvalancheAI/snow-profiles/magnified-profiles/'
		self.resolution = resolution

		# Load in history of images loaded if it exists for the current model
		self.images_loaded = self.read_loaded()
		if self.images_loaded:
			print(f"Previous images model trained on: {', '.join(self.images_loaded)}")
		else:
			print("No previously trained images detected...")

		# Gather available photos
		self.avail_photos = [image for image in glob(f"{self.profile_path}*.JPG") if image not in self.images_loaded]
		print(f"Pipeline loaded with {len(self.avail_photos)} available photos to load...\n")
		

	def load_batch(self, count, resample = False):
		"""
		Load a batch of subjects based on count requested and whether resampling requested

		Function arguments:
			count (int) - Number of images to load in
			resample (bool) - Whether to resample images loaded in through rotation
		"""
		# Initialize array to hold images
		x = np.array([None])

		# Save already loaded images
		self.save_loaded()

		# Iterate through image pool and load in until count is satisfied
		while self.avail_photos and x.shape[0] < count:
			# Pop next image and add to images loaded
			image_filename = self.avail_photos.pop(0)
			self.images_loaded.append(image_filename)

			# Process image and append to x train variable
			image = self.load_image(image_filename)
			if len(x.shape) > 1 and image.shape[0] == x.shape[1]:
				image = image.reshape((1, image.shape[0], image.shape[1], 3))
			else:
				image = image.reshape((1, image.shape[1], image.shape[0], 3))
			if len(x.shape) <= 1:
				x = image
			else:
				x = np.append(x, image, axis = 0)
			if resample:
				x = np.append(x, self.resample(image), axis = 0)
		return x


	# Load and preprocess images
	def load_image(self, filepath):
		"""
		Load image and preprocess to prepare for training through resampling,
		normalization and adjusting dtype resolution. Finally return end product.

		Function arguments:
			filepath (str) - Path of image to load
		"""
		print(f"Loading {filepath}")
		image = cv2.imread(filepath)  # Read image
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, self.resolution)  # Resize
		image = (image / 127.5) - 1  # Normalize to -1, 1 for tanh
		image = np.array(image, dtype = np.float32)
		print(f"Image shape {image.shape} | max {image.max()} | deviation {image.std()}")
		return image

	def resample(self, image):
		"""
		Slip the image 180 degrees to easily resample the snowpack sample

		Function arguments:
			image (3D numpy array) - Numpy array containing a single RGB image 
		"""
		synthetic_images = np.flip(image) # Flip the image 180 degrees
		return synthetic_images

	def save_loaded(self):
		"""
		Save the images loaded to a text file for remember which files have been loaded
		"""
		# Remove the path if it exists
		if os.path.exists(f"{self.path}images_loaded.txt"):
			os.remove(f"{self.path}images_loaded.txt")

		# Open a new file and output the images trained on
		with open(f"{self.path}images_loaded.txt", 'w') as file:
			for image in self.images_loaded:
				file.write(image+'\n')

	def read_loaded(self):
		"""
		Read in the images that have been loaded previously
		"""
		if os.path.exists(f"{self.path}images_loaded.txt"):
			with open(f"{self.path}images_loaded.txt", 'r') as file:
				return [line.split('\n')[0] for line in file.readlines()]
		else: # If no saved record of trained images exists
			return [] # Return an empty list
