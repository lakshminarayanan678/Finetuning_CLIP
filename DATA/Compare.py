import pandas as pd

DATA_CSV = r"C:\Users\DELL\Desktop\CLIP\Finetuning_CLIP\DATA\Data_Entry_2017.csv"

SELECTED_CLASSES_ALL = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
    "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema","Fibrosis",
    "Pleural_Thickening","Hernia"
]
SELECTED_CLASSES_TARGET = ["Atelectasis", "Effusion", "Infiltration", "Mass", "Nodule"]

df_orig = pd.read_csv(DATA_CSV)

# Normalize whitespace in key columns to be safe
for col in ["Finding Labels", "View Position", "Patient Gender"]:
    if col in df_orig.columns:
        df_orig[col] = df_orig[col].astype(str).str.strip()

# Common pre-filter: single-label rows only
df_single = df_orig[~df_orig["Finding Labels"].str.contains(r"\|", regex=True, na=False)]

# PIPE A: your first approach (kept many classes, no view filter, one image per patient)
df_a = df_single[df_single["Finding Labels"].isin(SELECTED_CLASSES_ALL)].copy()
df_a = df_a.sort_values("Follow-up #").drop_duplicates(subset="Patient ID", keep="first")

# PIPE B: your second approach (selected 5 classes, PA view, one image per patient)
df_b = df_single[df_single["Finding Labels"].isin(SELECTED_CLASSES_TARGET)].copy()
df_b = df_b[df_b["View Position"] == "PA"]
df_b = df_b.sort_values("Follow-up #").drop_duplicates(subset="Patient ID", keep="first")

# Counts per gender for Effusion in both pipelines
def gender_counts_for_effusion(df, label="Effusion"):
    g = df[df["Finding Labels"] == label].groupby("Patient Gender").size()
    # ensure F and M keys exist
    return {"F": int(g.get("F", 0)), "M": int(g.get("M", 0))}

print("Effusion counts (PIPE A):", gender_counts_for_effusion(df_a))
print("Effusion counts (PIPE B):", gender_counts_for_effusion(df_b))

# Which Patient IDs appear for Effusion in each pipeline?
pats_a = set(df_a[df_a["Finding Labels"] == "Effusion"]["Patient ID"].astype(str))
pats_b = set(df_b[df_b["Finding Labels"] == "Effusion"]["Patient ID"].astype(str))

only_in_a = pats_a - pats_b
only_in_b = pats_b - pats_a
in_both = pats_a & pats_b

print(f"\nEffusion Patient IDs only in PIPE A: {len(only_in_a)}")
print(f"Effusion Patient IDs only in PIPE B: {len(only_in_b)}")
print(f"Effusion Patient IDs in both: {len(in_both)}")

# Print a few example rows for the differing patients so we can inspect why they differ
def show_rows_for_ids(df, ids, n=10):
    ids_list = list(ids)[:n]
    if not ids_list:
        print("  (none)")
        return
    display = df[df["Patient ID"].astype(str).isin(ids_list)][
        ["Patient ID", "Image Index", "Finding Labels", "View Position", "Patient Gender", "Follow-up #"]
    ]
    print(display.to_string(index=False))

print("\nExamples of Effusion rows only in PIPE A:")
show_rows_for_ids(df_a, only_in_a, n=10)

print("\nExamples of Effusion rows only in PIPE B:")
show_rows_for_ids(df_b, only_in_b, n=10)

# Extra helpful checks
# 1) Count of Effusion before deduplication (per view and gender)
print("\nEffusion counts BEFORE deduplication (by View Position and Gender):")
print(df_single[df_single["Finding Labels"] == "Effusion"].groupby(["View Position","Patient Gender"]).size().unstack(fill_value=0))

# 2) Any NaN/empty view positions for Effusion?
effusion_views = df_single[df_single["Finding Labels"] == "Effusion"]["View Position"].isna().sum()
print(f"\nNumber of Effusion rows with NaN View Position: {effusion_views}")
