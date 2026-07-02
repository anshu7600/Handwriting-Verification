import sys
from pathlib import Path
from torchvision import transforms
from torch.utils.data import DataLoader

root_path = Path.cwd().parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from src.dataset import *
from src.test import *

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
