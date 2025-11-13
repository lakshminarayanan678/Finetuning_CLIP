import os
import shutil
import pandas as pd
from glob import glob

# === USER CONFIG ===
BASE_DIR = r"C:\Users\DELL\Desktop\CLIP\Data"
CSV_PATH = os.path.join(BASE_DIR, "Data_Entry_2017.csv")
IMAGE_DIRS = [os.path.join(BASE_DIR, f"images_{i:03d}", "images") for i in range(1, 13)]
OUTPUT_DIR = r"C:\Users\DELL\Desktop\CLIP\Finetuning_CLIP\DATA\FINALDATA"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "balanced_dataset.csv")

SELECTED_CLASSES = ["Atelectasis", "Effusion", "Infiltration", "Mass", "Nodule"]
VIEW_POSITION = "PA"

os.makedirs(IMAGES_DIR, exist_ok=True)

# --- Step 1: Load metadata ---
df = pd.read_csv(CSV_PATH)
print(f"Total entries in dataset: {len(df)}")

# --- Step 2: Keep only single-label entries ---
df = df[~df["Finding Labels"].str.contains(r"\|", regex=True, na=False)]
print(f"After keeping only single-label images: {len(df)}")

# --- Step 3: Keep only selected classes ---
df = df[df["Finding Labels"].isin(SELECTED_CLASSES)]
print(f"After filtering to selected classes: {len(df)}")

# --- Step 4: Keep only PA images ---
df_pa = df[df["View Position"] == VIEW_POSITION]
print(f"After filtering to PA view only: {len(df_pa)}")

# --- Step 5: Keep one PA image per patient ---
df_pa = df_pa.sort_values("Follow-up #").drop_duplicates(subset="Patient ID", keep="first")
print(f"After keeping one PA image per patient: {len(df_pa)}")

# --- Step 6: Map full image paths ---
print("\nMapping image filenames to full paths...")
image_map = {}
for d in IMAGE_DIRS:
    for img_path in glob(os.path.join(d, "*")):
        image_map[os.path.basename(img_path)] = img_path

df_pa["Image Path"] = df_pa["Image Index"].map(image_map)
missing_paths = df_pa["Image Path"].isna().sum()
df_pa = df_pa.dropna(subset=["Image Path"])
print(f"Dropped {missing_paths} rows with missing images. Remaining: {len(df_pa)}")

# --- Debug info for mismatched filenames ---
if missing_paths > 0:
    print("\n⚠️ Example missing image names:")
    print(df_pa[df_pa["Image Path"].isna()]["Image Index"].head().tolist())

print(f"\n✅ Found {len(df_pa)} valid images after mapping.")

# --- Step 7: Gender distribution per class ---
gender_summary = df_pa.groupby(["Finding Labels", "Patient Gender"]).size().unstack(fill_value=0)
print("\nGender distribution per class:")
print(gender_summary)

# --- Step 8: Balance dataset ---
min_per_class_gender = gender_summary.min(axis=1).min()
print(f"\nBalancing dataset to {min_per_class_gender} samples per gender per class...")

balanced_dfs = []
for cls, group in df_pa.groupby("Finding Labels"):
    males = group[group["Patient Gender"] == "M"].sample(min_per_class_gender, random_state=42)
    females = group[group["Patient Gender"] == "F"].sample(min_per_class_gender, random_state=42)
    balanced_dfs.append(pd.concat([males, females]))

balanced_df = pd.concat(balanced_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"\nTotal balanced samples: {len(balanced_df)} "
      f"({len(SELECTED_CLASSES)} classes × 2 genders × {min_per_class_gender} each)")

# --- Step 9: Create captions (domain-specific) ---
balanced_df["Caption"] = (
    "Posteroanterior view chest X-ray showing " + balanced_df["Finding Labels"].str.lower() + "."
)
print("\nExample captions:")
print(balanced_df["Caption"].head().tolist())

# --- Step 10: Save CSV ---
balanced_df[["Image Path", "Caption"]].to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved CSV with captions to: {OUTPUT_CSV}")

# --- Step 11: Copy images ---
print("\nCopying balanced images...")
for i, row in balanced_df.iterrows():
    src = row["Image Path"]
    dst = os.path.join(IMAGES_DIR, os.path.basename(src))
    shutil.copy(src, dst)
print(f"Copied {len(balanced_df)} images to {IMAGES_DIR}")