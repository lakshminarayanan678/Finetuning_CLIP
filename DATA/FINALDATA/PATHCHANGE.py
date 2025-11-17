# This script updates image file paths in a CSV file to point to a new 
# directory while retaining the original filenames.

import pandas as pd
import os
import re

csv_file = "/content/drive/MyDrive/COLAB/balanced_dataset.csv"
output_csv = "/content/drive/MyDrive/COLAB/updated_data.csv"

image_column = "ImagePath"
new_image_folder = "/content/drive/MyDrive/COLAB/images"


df = pd.read_csv(csv_file)

def extract_filename(path):
    path = str(path)
    # split by both types of slashes
    parts = re.split(r'[\\/]', path)
    return parts[-1]  # last item is the filename


def make_new_path(old_value):
    filename = extract_filename(old_value)
    return os.path.join(new_image_folder, filename)

df[image_column] = df[image_column].apply(make_new_path)

df.to_csv(output_csv, index=False)

print("DONE. Corrected paths saved to:", output_csv)
