from collections import defaultdict
import os
import random
from pathlib import Path 
import pandas as pd

SOURCE_DIR = Path("C:/Users/vvrao/PycharmProjects/Handwriting-Verification/data")
if not SOURCE_DIR.exists():
    raise Exception(f"Dataset folder not found: {SOURCE_DIR}")

writer_map = defaultdict(list)

for file in os.listdir(SOURCE_DIR):
    writer_id = file[:4]  # assumes 0001a.png style
    path = os.path.join(SOURCE_DIR, file)
    writer_map[writer_id].append(path)

print(writer_map["0001"])

writers = list(writer_map.keys())

def get_positive_pair():
    w = random.choice(writers)
    imgs = writer_map[w]
    return random.sample(imgs, 2), 1

def get_negative_pair():
    w1, w2 = random.sample(writers, 2)
    return (random.choice(writer_map[w1]),
            random.choice(writer_map[w2])), 0



rows = []

for writer, files in writer_map.items():
    for f in files:
        rows.append([writer, f])

df = pd.DataFrame(rows, columns=["writer_id", "filepath"])
df.to_csv("writer_index.csv", index=False)