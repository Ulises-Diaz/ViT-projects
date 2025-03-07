import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import os



device = "cuda" if torch.cuda.is_available() else "cpu"

# Add Dataset
train_dir = ''
test_dir = ''

# DataLoaders 
'''
Prepares Data to a format that ViT accept
'''

NUM_WORKERS = os.cpu_count()

def create_dataloader (
    train_dir : str, 
    test_dir : str,
    transform : transforms.Compose, 
    batch_size : int, 
    num_workers : int = NUM_WORKERS
): 
    
    # Use ImageFolder to create Datasets 
    train_data = datasets.ImageFolder(train_dir, transform= transform)
    test_data = datasets.ImageFolder(test_dir, transform= transform)
    
    # Get class Names 
    class_names = train_data.classes 

    # Turn images into data Loaders
    train_dataloader = DataLoader(
        train_data, 
        batch_size= batch_size, 
        shuffle= True,
        num_workers= num_workers, 
        pin_memory= True
    )
    
    
    test_dataloader = DataLoader(
        test_data, 
        batch_size= batch_size, 
        shuffle= False,
        num_workers= num_workers, 
        pin_memory= True
    )
    
    return train_dataloader, test_dataloader, class_names


# Create Image size
IMG_SIZE = 224 

#  Create Transform pipeline manually

# Transforms all image into a 224 format. 
manual_transform = transforms.Compose ([
    transforms.Resize ((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor ()
    ])

print(f'Manually created Transforms : {manual_transform}')



# Set Batch Size 

# Use 32 if using a GPU
BATCH_SIZE = 16 # This can be modified according to ur computer. Large Batch Size means a bigger model. 


# Create DataLoaders 

train_dataloader, test_dataloader, class_names = create_dataloader(
    train_dir= train_dir ,
    test_dir= test_dir,
    transform= manual_transform,
    batch_size= BATCH_SIZE
)

print(train_dataloader, test_dataloader, class_names) # this wont print bcs there's no datasets 

'''
# Test if the image is loaded properly 

# Get  Batch Image

image_batch, label_batch = next(iter(train_dataloader))

# Get an image from the batch
image, label = image_batch[0], label_batch[0]

# View the batch shape

print(image.shape, label)

# Plot the image

plt.imshow(image.permute(1, 2, 0)) # Rearrange dimensions for matplotlib
plt.title(class_names[0])
plt.axis(False)

'''


# Train the model

