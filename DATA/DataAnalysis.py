# FINAL CLASSES are Atelectasis, Effusion, Infiltration, Mass, Nodule (atleast 300 images in both Male and Female)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
DATA_CSV = r"C:\Users\DELL\Desktop\CLIP\Finetuning_CLIP\DATA\Data_Entry_2017.csv"
SELECTED_CLASSES = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural_Thickening",
    "Hernia"
]

# === READ CSV ===
df = pd.read_csv(DATA_CSV)

# --- Filter to single-class only ---
df = df[~df["Finding Labels"].str.contains(r"\|")]

# --- Keep only selected 5 classes ---
df = df[df["Finding Labels"].isin(SELECTED_CLASSES)]

# --- One image per patient (choose first occurrence) ---
df = df.sort_values("Follow-up #").drop_duplicates(subset="Patient ID", keep="first")

print(f"Total samples after filtering: {len(df)}")
print("\nUnique View Positions:", df["View Position"].unique())

# === BASIC SUMMARIES ===

# Gender distribution
gender_summary = df.groupby(["Finding Labels", "Patient Gender"]).size().unstack(fill_value=0)

# View position distribution
view_summary = df.groupby(["Finding Labels", "View Position"]).size().unstack(fill_value=0)


# Print summaries
print("\n--- Gender distribution per class ---")
print(gender_summary)
print("\n--- View Position distribution per class ---")
print(view_summary)


# === SAVE SUMMARIES ===
gender_summary.to_csv("gender_distribution.csv")
view_summary.to_csv("view_distribution.csv")

# === VISUALIZATIONS ===
sns.set(style="whitegrid", font_scale=1.2)

# 1. Gender distribution per class
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Finding Labels", hue="Patient Gender", palette="coolwarm")
plt.title("Gender Distribution per Disease Class (1 Image per Patient)")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("gender_distribution_plot.png", dpi=300)
plt.show()

# 2. View position distribution per class
plt.figure(figsize=(8,5))
sns.countplot(data=df, x="Finding Labels", hue="View Position", palette="viridis")
plt.title("View Position Distribution per Disease Class (1 Image per Patient)")
plt.xlabel("Disease Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("view_distribution_plot.png", dpi=300)
plt.show()


print("\n✅ Analysis complete.")
print("Saved:")
print("  → gender_distribution.csv")
print("  → view_distribution.csv")
print("  → gender_distribution_plot.png")
print("  → view_distribution_plot.png")