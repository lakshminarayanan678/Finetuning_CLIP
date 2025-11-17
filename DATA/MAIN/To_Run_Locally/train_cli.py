import argparse
import clip
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import wandb
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class ChestXray(Dataset):
    """
    Custom Dataset for Chest X-ray images and their corresponding captions.
    """
    def __init__(self, csv_path, preprocess):
        """
        Args:
            csv_path (str): Path to the CSV file containing ImagePath and Caption columns.
            preprocess (callable): CLIP's image preprocessing function.
        """
        # Load the data from the specified CSV file
        self.data = pd.read_csv(csv_path)
        self.preprocess = preprocess

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves the image and caption at the specified index."""
        row = self.data.iloc[idx]
        
        img = Image.open(row["ImagePath"]).convert("RGB") 
        img = self.preprocess(img) 

        caption = row["Caption"]
        return img, caption


class CLIPFineTune(nn.Module):
    """
    A fine-tuning model for CLIP where only the projection head is trainable.
    The CLIP image and text encoders are frozen.
    """
    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        # Load the frozen CLIP model
        self.clip = clip_model
        # Trainable projection head (matches CLIP's embedding dimension)
        self.proj = nn.Linear(embed_dim, embed_dim) 
        
        # Freeze the entire base CLIP model
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images, text_tokens):
        """
        Forward pass for image-text pair.
        """
        # Encode features with the frozen CLIP model
        with torch.no_grad():
            img_f = self.clip.encode_image(images)
            txt_f = self.clip.encode_text(text_tokens)

        # Normalize features
        img_f = img_f / img_f.norm(dim=1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=1, keepdim=True)
        
        # Ensure correct type for the projection head
        img_f = img_f.float()
        txt_f = txt_f.float()

        # Apply the trainable projection head only to image features
        img_f = self.proj(img_f)

        # Calculate logits (similarity matrix)
        logits = img_f @ txt_f.t()
        return logits

# --- Main Training Function ---

def train_model(args):
    """
    Runs the main training and validation loop.

    Args:
        args (argparse.Namespace): Command-line arguments containing file paths and hyperparameters.
    """
    print(f"--- Starting Fine-tuning on {DEVICE} ---")
    
    # 1. Load CLIP Model and Preprocessor
    model, preprocess = clip.load(args.model_name, device=DEVICE, jit=False)

    # 2. Instantiate Fine-tune Model and Load Pre-trained Checkpoint
    ft_model = CLIPFineTune(model).to(DEVICE) 
    
    if args.checkpoint_path and os.path.exists(args.checkpoint_path):
        print(f"Loading checkpoint from: {args.checkpoint_path}")
        state_dict = torch.load(args.checkpoint_path, map_location=DEVICE)
        ft_model.load_state_dict(state_dict)
    else:
        print("Starting training from initial fine-tuned state (no checkpoint loaded).")
    
    # 3. Setup Data Loaders
    train_dataset = ChestXray(args.train_csv_path, preprocess)
    val_dataset = ChestXray(args.val_csv_path, preprocess)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) 

    # 4. Setup Optimizer, Loss, and Scheduler
    optimizer = torch.optim.AdamW(ft_model.proj.parameters(), lr=args.learning_rate)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_factor, patience=args.lr_patience)
    
    # 5. Training Loop Setup
    best_val_loss = float('inf')
    epochs_no_improve = 0
    start_epoch = args.start_epoch # Starting epoch number for logging/display
    
    # Optional: Initialize W&B here based on args
    if args.use_wandb:
        wandb.init(project="Finetuning_CLIP", name=args.run_name, config=args)
        wandb.watch(ft_model)

    # 6. Main Loop
    for epoch in range(start_epoch, start_epoch + args.num_epochs):
        # --- Training Phase ---
        ft_model.train()
        total_loss = 0

        for batch_idx, (imgs, caps) in enumerate(train_loader):
            imgs = imgs.to(DEVICE)
            # Tokenize captions
            tok = clip.tokenize(list(caps), truncate=True).to(DEVICE) 

            logits = ft_model(imgs, tok)
            
            # Loss Calculation: Contrastive Loss
            # Correct labels are on the diagonal (image i matches text i)
            labels = torch.arange(len(imgs)).to(DEVICE)
            
            # Symmetric loss (Image-to-Text and Text-to-Image)
            loss = (loss_fn(logits, labels) + loss_fn(logits.t(), labels)) / 2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        

        avg_train_loss = total_loss / len(train_loader)
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} — LR: {current_lr:.1e}, Train Loss: {avg_train_loss:.4f}")

        # --- Validation Phase ---
        ft_model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, caps in val_loader:
                imgs = imgs.to(DEVICE)
                tok = clip.tokenize(list(caps), truncate=True).to(DEVICE)

                logits = ft_model(imgs, tok)
                labels = torch.arange(len(imgs)).to(DEVICE)
                loss = (loss_fn(logits, labels) + loss_fn(logits.t(), labels)) / 2
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss) # Update LR based on validation loss

        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")
        print("--------------------------------------------------")
        
        # Log to W&B
        if args.use_wandb:
             wandb.log({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "epoch": epoch,
                "learning_rate": current_lr
            })

        # --- Checkpointing and Early Stopping ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save best model
            save_path = os.path.join(args.save_dir, 'clip_ft_best.pt')
            torch.save(ft_model.state_dict(), save_path)
            print(f"New best model saved to {save_path}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience_es:
                print(f"Early stopping triggered after {epoch} epochs with no improvement!")
                break

        # Save checkpoint every N epochs
        if (epoch + 1) % args.checkpoint_frequency == 0:
            save_path = os.path.join(args.save_dir, f"clip_epoch_{epoch}.pt")
            torch.save(ft_model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

    print("--- Training finished. ---")
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLIP Fine-Tuning Script.")

    # --- File Paths ---
    parser.add_argument("--train_csv_path", type=str, required=True, 
                        help="Path to the training CSV file (must contain 'ImagePath' and 'Caption').")
    parser.add_argument("--val_csv_path", type=str, required=True, 
                        help="Path to the validation CSV file.")
    parser.add_argument("--save_dir", type=str, default="checkpoints", 
                        help="Directory to save model checkpoints and the best model.")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Path to a pre-trained fine-tune checkpoint to resume training from.")

    # --- Hyperparameters ---
    parser.add_argument("--model_name", type=str, default="ViT-B/32", 
                        help="Name of the CLIP model to use (e.g., 'ViT-B/32').")
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training and validation.")
    parser.add_argument("--learning_rate", type=float, default=1e-3, 
                        help="Initial learning rate for the optimizer.")
    parser.add_argument("--num_epochs", type=int, default=100, 
                        help="Total number of epochs to train for.")
    parser.add_argument("--start_epoch", type=int, default=0, 
                        help="Starting epoch number (useful for resumed training).")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of data loader workers.")
    
    # --- Checkpointing and Early Stopping ---
    parser.add_argument("--patience_es", type=int, default=10, 
                        help="Early stopping patience (epochs without improvement).")
    parser.add_argument("--checkpoint_frequency", type=int, default=5, 
                        help="Save a checkpoint every N epochs.")

    # --- LR Scheduler ---
    parser.add_argument("--lr_factor", type=float, default=0.1, 
                        help="Factor by which the LR will be reduced on plateau.")
    parser.add_argument("--lr_patience", type=int, default=3, 
                        help="Patience (epochs) for LR scheduler.")
    
    # --- W&B Logging ---
    parser.add_argument("--use_wandb", action="store_true", 
                        help="Flag to enable Weights & Biases logging.")
    parser.add_argument("--run_name", type=str, default="new_run_clip_ft", 
                        help="Name for the Weights & Biases run.")


    args = parser.parse_args()
    
    # Create save directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Execute the training
    train_model(args)
