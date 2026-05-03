import os
import pandas as pd
import shutil

# paths
metadata_path = "HAM10000_metadata.csv"

image_folders = [
    "HAM10000_images_part_1",
    "HAM10000_images_part_2"
]

output_folder = "dataset"

# create output folders
os.makedirs(f"{output_folder}/melanoma", exist_ok=True)
os.makedirs(f"{output_folder}/non_melanoma", exist_ok=True)

# load metadata
df = pd.read_csv(metadata_path)

# function to find image in both folders
def find_image(image_name):
    for folder in image_folders:
        path = os.path.join(folder, image_name)
        if os.path.exists(path):
            return path
    return None

# loop through dataset
for _, row in df.iterrows():
    img_name = row['image_id'] + ".jpg"
    label = row['dx']
    
    src = find_image(img_name)
    
    if src is None:
        continue
    
    if label == "mel":
        dst = os.path.join(output_folder, "melanoma", img_name)
    else:
        dst = os.path.join(output_folder, "non_melanoma", img_name)
    
    shutil.copy(src, dst)

print("Dataset organized successfully!")