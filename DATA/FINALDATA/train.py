import clip
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import wandb

wandb.init(project="Finetuning_CLIP", name="new_run_clip_ft")
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------
# Dataset
# -----------------------------------------------------
class ChestXray(Dataset):
    def __init__(self, csv_path, preprocess):
        self.data = pd.read_csv(csv_path)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row["ImagePath"]).convert("RGB") # cuz, clip expect 3 channel images, and not simplt greyscale
        img = self.preprocess(img) #preprocess defined before during trial of zero-shot

        caption = row["Caption"]
        return img, caption

train_csv = r"/content/drive/MyDrive/COLAB/train.csv" # later adapt it to cli
val_csv   = r"/content/drive/MyDrive/COLAB/val.csv"

# -----------------------------------------------------
# Add a trainable projection head
# -----------------------------------------------------
class CLIPFineTune(nn.Module):
    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        self.clip = clip_model
        self.proj = nn.Linear(512, 512)   # trainable head

    def forward(self, images, text_tokens):
        with torch.no_grad():
            img_f = self.clip.encode_image(images)
            txt_f = self.clip.encode_text(text_tokens)

        img_f = img_f / img_f.norm(dim=1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=1, keepdim=True)
        img_f = img_f.float()
        txt_f = txt_f.float()


        # Only projection head is trainable
        img_f = self.proj(img_f)

        logits = img_f @ txt_f.t()
        return logits

model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

ft_model = CLIPFineTune(model).to(device) 
state_dict = torch.load("/content/drive/MyDrive/COLAB/clip_epoch_80.pt", map_location=device) # INCLUDE IF STATEMENT
ft_model.load_state_dict(state_dict)
for p in ft_model.clip.parameters():
    p.requires_grad = False

train_loader = DataLoader(ChestXray(train_csv, preprocess), batch_size=32, shuffle=False)
val_loader   = DataLoader(ChestXray(val_csv, preprocess), batch_size=32, shuffle=False) #already shuffled in Prepr pipeline


from torch.optim.lr_scheduler import ReduceLROnPlateau
optimizer = torch.optim.AdamW(ft_model.proj.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)

# -----------------------------------------------------
# Training loop
# -----------------------------------------------------
patience_es = 10        # Number of epochs to wait for improvement
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

for epoch in range(100):
    ft_model.train()
    total_loss = 0

    for imgs, caps in train_loader:
        imgs = imgs.to(device)
        tok = clip.tokenize(list(caps)).to(device) #cuz tokenize expects list of strings #UNDERSTAND

        logits = ft_model(imgs, tok)
        # correct labels → diagonal alignment
        labels = torch.arange(len(imgs)).to(device)
        loss = (loss_fn(logits, labels) + loss_fn(logits.t(), labels)) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # After finishing all batches of an epoch
    current_lr = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+81} — Current LR: {current_lr}")

    avg_train_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+81} Train Loss: {total_loss/len(train_loader):.4f}")

    # Validation
    ft_model.eval()
    val_loss = 0

    with torch.no_grad():
        for imgs, caps in val_loader:
            imgs = imgs.to(device)
            tok = clip.tokenize(list(caps)).to(device)

            logits = ft_model(imgs, tok)
            labels = torch.arange(len(imgs)).to(device)
            loss = (loss_fn(logits, labels) + loss_fn(logits.t(), labels)) / 2
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch} Val Loss: {val_loss/len(val_loader):.4f}")
    print("--------------------------------------------------")

    wandb.log({
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "epoch": epoch
    })

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        # Save best model
        torch.save(ft_model.state_dict(), '/content/drive/MyDrive/COLAB/clip_ft_best.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience_es:
            print(f"Early stopping triggered after {epoch} epochs!")
            early_stop = True
            break

    # -----------------------------
    #  SAVE CHECKPOINT EVERY 5 EPOCHS
    # -----------------------------
    if (epoch + 1) % 5 == 0:
        save_path = f"/content/drive/MyDrive/COLAB/clip_epoch_{epoch}.pt"
        torch.save(ft_model.state_dict(), save_path)
        print(f"✅ Saved checkpoint: {save_path}")

print("Model saving")
torch.save(ft_model.state_dict(), '/content/drive/MyDrive/COLAB/clip_ft4.pt') 