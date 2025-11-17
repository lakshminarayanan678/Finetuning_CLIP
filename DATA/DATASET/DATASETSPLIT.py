import pandas as pd
from sklearn.model_selection import train_test_split

input_csv = r"C:\Users\DELL\Desktop\CLIP\Finetuning_CLIP\DATA\FINALDATA\balanced_dataset.csv"
train_csv = r"C:\Users\DELL\Desktop\CLIP\Finetuning_CLIP\DATA\FINALDATA\train.csv"
val_csv   = r"C:\Users\DELL\Desktop\CLIP\Finetuning_CLIP\DATA\FINALDATA\val.csv"
test_csv  = r"C:\Users\DELL\Desktop\CLIP\Finetuning_CLIP\DATA\FINALDATA\test.csv"


# For 80-10-10 split
train_ratio = 0.8
val_ratio = 0.5  # 50% of the remaining 20% for validation
random_state = 42  

df = pd.read_csv(input_csv)

# First split: 80% for training, 20% for temp (validation + test)
train_df, temp_df = train_test_split(
    df,
    test_size=1 - train_ratio,
    random_state=random_state,
    shuffle=True
)

# Second split: 10% for validation, 10% for test from the temp_df
val_df, test_df = train_test_split(
    temp_df,
    test_size=val_ratio, # This makes val_df and test_df each 10% of original df
    random_state=random_state,
    shuffle=True
)

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"Train samples: {len(train_df)}")
print(f"Val samples:   {len(val_df)}")
print(f"Test samples:  {len(test_df)}")
print("Saved:")
print(" →", train_csv)
print(" →", val_csv)
print(" →", test_csv)
