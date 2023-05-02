import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

layer = torch.nn.Sequential()

image = Image.open('images/sagiri_ss.png')
transform = transforms.ToTensor()
resized_image = image.resize((35, 35))

tensor = transform(resized_image)
# Print the tensor shape and data type

pca = torch.pca_lowrank(tensor)
pca1, pca2, pca3 = pca


image1 = transforms.ToPILImage(pca1)
image2 = transforms.ToPILImage(pca2)
image3 = transforms.ToPILImage(pca3)

# Display the images
print(pca1.shape, pca1.dtype)
print(pca2.shape, pca2.dtype)
print(pca3.shape, pca3.dtype)
# tensor = np.transpose(pca1, (2, 1, 0))

for a, b in zip(tensor, pca1):
    for elemA, elemB in zip(a, b):
        print(f"regular shape:  {elemA.shape} vs pca A: {elemB.shape}")
        print("Element 1:", elemA)
        print("Element 2:", elemB)

# print(tensor.shape, tensor.dtype)
# plt.imshow(tensor)
# plt.show()