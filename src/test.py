from collections import defaultdict
import os
from pathlib import Path
import random
import json

# Is the path to the dataset 
SOURCE_DIR = Path(r"C:\Users\Pratham\OneDrive\Desktop\Portfolio\Handwriting-Verification\dataset")

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