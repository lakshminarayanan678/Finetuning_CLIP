import pandas as pd

# === CONFIG ===
DATA_CSV = r"C:\Users\DELL\Desktop\CLIP\Data\Data_Entry_2017.csv"
SELECTED_CLASSES = ["Atelectasis", "Effusion", "Infiltration", "Mass", "Nodule"]
VIEW_POSITION = "PA"

# === READ CSV ===
df = pd.read_csv(DATA_CSV)

# --- Keep only single-class entries ---
df = df[~df["Finding Labels"].str.contains(r"\|")]

# --- Keep only selected classes ---
df = df[df["Finding Labels"].isin(SELECTED_CLASSES)]

# --- Keep only chosen view position ---
df = df[df["View Position"] == VIEW_POSITION]

# --- One image per patient ---
df = df.sort_values("Follow-up #").drop_duplicates(subset="Patient ID", keep="first")

# --- Basic sanity check ---
print(f"Total samples after filtering: {len(df)}")
print(f"Unique view positions: {df['View Position'].unique()}")

# --- Count males/females per class ---
gender_counts = df.groupby(["Finding Labels", "Patient Gender"]).size().unstack(fill_value=0)
print("\nGender count per class:")
print(gender_counts)

# --- Find the number of usable samples per class (balanced M/F) ---
# We'll take the minimum between male and female counts for each class
gender_counts["usable_per_class"] = gender_counts.min(axis=1)
min_balanced = gender_counts["usable_per_class"].min()

print("\nBalanced samples possible per class (equal M/F):")
print(gender_counts[["usable_per_class"]])
print(f"\n✅ You can safely sample {min_balanced*2} images (M+F) per class "
      f"({min_balanced} male + {min_balanced} female).")

# --- Optional summary CSV for inspection ---
gender_counts.to_csv("gender_balance_summary.csv")

print("\nSummary saved to 'gender_balance_summary.csv'.")
