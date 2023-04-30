import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

layer = torch.nn.Sequential()

image = Image.open('images/sagiri_ss.png')
transform = transforms.ToTensor()
tensor = transform(image)

# Print the tensor shape and data type
print(tensor.shape, tensor.dtype)

pca = torch.pca_lowrank(tensor)
height, width, channels = tensor.shape
print(height, width, channels)

# Iterate through each pixel
for i in range(height):
    for j in range(width):
        for k in range(channels):
            print(f"Pixel ({i}, {j}, {k}) in image 1: {tensor[i,j,k]}")
            print(f"Pixel ({i}, {j}, {k}) in image 2: {pca[0][i,j,k]}")
