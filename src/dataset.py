from collections import defaultdict
import os
from pathlib import Path
import random
import json

# Is the path to the dataset 
SOURCE_DIR = Path("C:\\Users\\vvrao\\Porfolio\\Handwriting-Verification\\dataset")

if not SOURCE_DIR.exists():  # Check if the dataset folder exists
    raise Exception(f"Dataset folder not found: {SOURCE_DIR}")

# When making a dictionary, if the key doesn't exist, it will create a new list for that key
writer_map = defaultdict(list)

for file in os.listdir(SOURCE_DIR): # Loop through all files in the dataset folder
    # writer_id will be same for the same writer, so we can use it to group images by writer
    writer_id = file[:4]  # writer_id is the first 4 characters of the filename (example: 0001a.png)
    path = os.path.join(SOURCE_DIR, file) # Create the full path to the file, by joining the dataset folder path and the filename
    writer_map[writer_id].append(path)  # makes a key and appends the path to the list of images for that writer_id

os.makedirs("..\\data", exist_ok=True)
with open("..\\data\\writer_map.json", "w") as f: # makes a new file called writer_map.json and opens it in write mode, so we can write the writer_map dictionary to it in JSON format
    json.dump(
        {k: [p for p in v] for k, v in writer_map.items()}, # makes a new dictionary with the same keys as writer_map, but the values are lists of paths (instead of defaultdict objects)
        f,
        indent=4
    )






import random
import torch
from torch.utils.data import Dataset
from PIL import Image

# Purpose of the SiameseDataset class is to create a dataset that can be used for training a Siamese network. 
# A Siamese network is a type of neural network that learns to differentiate between two inputs, often used for tasks like signature verification or face recognition. 
# The dataset generates pairs of images, either from the same writer (positive pair) or from different writers (negative pair), along with a label indicating whether the pair is positive or negative.
# Basically: Generate Positive and Negative pairs of images for training a Siamese Network. Load the image files, apply any specified transformations, and return them to pytorch

class SiameseDataset(Dataset): # Inherit from the Dataset class, which is a base class for all datasets in PyTorch. This allows us to create a custom dataset that can be used with PyTorch's DataLoader for training and evaluation.
    # Pytorch requires that we implement the __init__, __len__, and __getitem__ methods for a custom dataset class.
    def __init__(self, writer_map, transform=None, num_samples=100000): # runs once when class is created
        """
        writer_map: dict -> {writer_id: [img_path1, img_path2, ...]}
        transform: torchvision transforms
        num_samples: virtual dataset size (since pairs are generated dynamically)
        """
        # Can be used anywhere in the class, but is only accessible within the class. It is used to store the writer_map, transform, and num_samples parameters passed to the constructor.
        self.writer_map = writer_map
        self.transform = transform
        self.num_samples = num_samples

        self.writers = list(writer_map.keys())

        # optional sanity filter: remove writers with <2 images
        self.writers = [w for w in self.writers if len(writer_map[w]) >= 2]

    def __len__(self): 
        # returns the number of samples in the dataset. This is used by PyTorch to determine how many batches to create during training or evaluation.
        return self.num_samples
  
    # def _load_image(self, path): 
    #     # helper function to load an image from a given path, convert it to grayscale, and apply any specified transformations. 
    #     # This is used in the __getitem__ method to load the images for each pair.
    #     img = Image.open(path).convert("RGB")  # grayscale is typical for handwriting
    #     if self.transform: # if transform is not None, apply the transform to the image. 
    #         img = self.transform(img) # calls the method and passes the image to it, which applies the transformations defined in the transform parameter (like resizing, normalization, etc.)
    #     return img
    def _load_image(self, path):
        img = Image.open(path)

        # print("Before convert:", img.mode)

        img = img.convert("RGB")

        # print("After convert:", img.mode)

        if self.transform:
            img = self.transform(img)

        return img

    def __getitem__(self, idx): # This method is called by PyTorch when it needs to get a sample from the dataset. It takes an index (idx) as input, but in this case, the index is not used because pairs are generated dynamically.
        # decide: positive or negative pair
        same = random.random() < 0.5

        if same:
            writer = random.choice(self.writers) # randomly select a writer from the list of writers
            img1, img2 = random.sample(self.writer_map[writer], 2) # randomly select 2 different images from the list of images for that writer
            label = 1 # label of 1 indicates that the images are from the same writer (positive pair)
        else:
            w1, w2 = random.sample(self.writers, 2) # randomly select 2 different writers from the list of writers, assigning them to w1 and w2
            # randomly select 1 image from the list of images for writer w1 and w2, and assign them to img1 and img2 respectively
            img1 = random.choice(self.writer_map[w1]) 
            img2 = random.choice(self.writer_map[w2])
            label = 0 # label of 0 indicates that the images are from different writers (negative pair)

        # applies the transformations defined in the transform parameter (like resizing, normalization, etc.) to the images
        img1 = self._load_image(img1)
        img2 = self._load_image(img2)

        # returns a tuple containing the two images and the label, which can be used for training or evaluation in a Siamese network. The label is converted to a PyTorch tensor of type float32, which is the expected format for loss functions in PyTorch.
        return img1, img2, torch.tensor(label, dtype=torch.float32) 
    

# dataset[42]
#         │
#         ▼
# __getitem__(42)
#         │
#         ▼
# 50% chance same writer?
#         │
#    ┌────┴────┐
#    │         │
#  Yes        No
#    │         │
# Pick 1      Pick 2
# writer      writers
#    │         │
# Choose      Choose
# 2 images    1 image from each
#    │         │
# Load images with PIL
#         │
# Apply transforms
#         │
# Convert to tensors
#         │
# Return:
# (img1, img2, label)





from torchvision import transforms
from torch.utils.data import DataLoader

# Define image preprocessing
your_transforms = transforms.Compose([
    transforms.Resize((224, 224)), # Resize all images
    transforms.ToTensor(), # Convert PIL Image to PyTorch Tensor
    transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
) # Normalize RGB image, by subtracting the mean and dividing by the standard deviation. This helps the model learn better by ensuring that the input data has a consistent scale and distribution.
])

# Create the dataset
dataset = SiameseDataset(
    writer_map=writer_map,
    transform=your_transforms,
    num_samples=100000
)

# Create the DataLoader, which asks the dataset for batches of data during training or evaluation. It handles shuffling, batching, and parallel loading of data. 
loader = DataLoader(
    dataset, # Whenever it needs a batch of data, it will call the __getitem__ method of the dataset to get a sample. 
    # The DataLoader will then combine these samples into batches and return them to the training loop.
    # it will call it similar to this: dataset[0], dataset[1], ..., dataset[batch_size-1] which actually runs the __getitem__ method of the dataset class, which generates a pair of images and a label indicating whether they are from the same writer or not.
    # and the number passed in doesn't matter since the dataset is virtual and generates pairs dynamically, but it is used to determine how many batches to create during training or evaluation.

    batch_size=32, # Number of samples per batch to load. 32 is a common choice, but can be adjusted.

    # doesn't do much since __getitem__ generates pairs dynamically
    shuffle=True # Whether to shuffle the data at every epoch. Shuffling is important for training to ensure that the model does not learn any order-based patterns in the data.
)




# Checking if the dataset and dataloader are working correctly by printing the shapes of the images and labels for a single sample and a batch of samples.

img1, img2, label = dataset[0] # basically called __getitem__(0) to get the first sample from the dataset, which returns a pair of images and a label indicating whether they are from the same writer or not.

print("Description of the an image in the dataset:")
# format is [channels, height, width] (C, H, W) for PyTorch tensors. For grayscale images, channels = 1.
print(img1.shape) 
print(img2.shape)
print(f"Label (.0 means different writers, and .1 means same writer): {label}")

print("\n")

for img1, img2, labels in loader: 
    # used the loader which batches the data into 32 samples per batch, and returns a tuple of 3 tensors: img1, img2, and labels. Each tensor has a shape of [batch_size, channels, height, width] for the images, and [batch_size] for the labels.
    print(img1.shape)
    print(img2.shape)
    print(labels.shape)
    break # without this it would print the shapes of all batches, but we only want to see the first batch for verification.
    
print("\nMore Data :)")
for i in range(5): # does the same thing as the above loop
    img1, img2, label = dataset[i]
       
    print(f"Sample {i}")
    print("Image 1 shape:", img1.shape)
    print("Image 2 shape:", img2.shape)
    print(f"Label: {label.item()} - {'Same writer' if label.item() == 1 else 'Different writers'}")
    print("-" * 30) 


