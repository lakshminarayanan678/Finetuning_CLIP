import os
import sys
import argparse
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import clip

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    print("Warning: Matplotlib, Seaborn, or Scikit-learn not found. Plotting and detailed metrics will be skipped.")

# --- Configuration Constants ---
# Disease classes we're working with
SELECTED_CLASSES = ["atelectasis", "mass", "effusion", "infiltration", "nodule"]

# Text descriptions for zero-shot classification (must be ordered to match SELECTED_CLASSES)
TEXT_PROMPTS = [
    "Posteroanterior view chest X-ray showing atelectasis.",
    "Posteroanterior view chest X-ray showing mass.",
    "Posteroanterior view chest X-ray showing effusion.",
    "Posteroanterior view chest X-ray showing infiltration.",
    "Posteroanterior view chest X-ray showing nodule."
]

# Mapping for extracting ground truth class from caption
CAPTION_TO_CLASS_MAP = {
    "Posteroanterior view chest X-ray showing atelectasis.": "atelectasis",
    "Posteroanterior view chest X-ray showing mass.": "mass",
    "Posteroanterior view chest X-ray showing effusion.": "effusion",
    "Posteroanterior view chest X-ray showing infiltration.": "infiltration",
    "Posteroanterior view chest X-ray showing nodule.": "nodule"
}

# --- Utility Functions ---

def get_ground_truth_class(caption: str) -> str:
    """Extracts the ground truth class from a given caption."""
    for prompt_key, cls_name in CAPTION_TO_CLASS_MAP.items():
        if prompt_key.lower() in caption.lower():
            return cls_name
    return "UNKNOWN" # Default if no match found

def plot_confusion_matrix(y_true: list, y_pred: list, classes: list, title: str, output_path: str):
    """Generate and save confusion matrix visualization."""
    if not HAS_PLOTTING:
        print("Cannot plot confusion matrix: plotting libraries not available.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix plot saved to {output_path}")


class ChestXray(Dataset):
    """Custom Dataset for Chest X-ray images and their corresponding captions."""
    def __init__(self, csv_path, preprocess):
        self.data = pd.read_csv(csv_path)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img = Image.open(row["ImagePath"]).convert("RGB")
        img = self.preprocess(img)
        caption = row["Caption"]
        return img, caption

class CLIPFineTune(nn.Module):
    """
    Fine-tuning model for CLIP with a trainable projection head (self.proj).
    The base CLIP model remains frozen.
    """
    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        self.clip = clip_model
        self.proj = nn.Linear(embed_dim, embed_dim)
        for p in self.clip.parameters():
            p.requires_grad = False

    def forward(self, images, text_tokens):
        # Encode features with the frozen CLIP model
        with torch.no_grad():
            img_f = self.clip.encode_image(images)
            txt_f = self.clip.encode_text(text_tokens)

        # Normalize features
        img_f = img_f / img_f.norm(dim=1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=1, keepdim=True)
        img_f = img_f.float()
        txt_f = txt_f.float()

        # Apply the projection head
        img_f = self.proj(img_f)

        # Calculate logits (similarity matrix)
        logits = img_f @ txt_f.t()
        return logits

# --- Prediction Function ---

@torch.no_grad()
def predict_image_with_ft_model(image_path: str, model_ft: nn.Module, preprocess_fn, tokenizer_fn, device: str):
    """
    Predicts the disease class for a single X-ray image using the fine-tuned CLIP model.
    Returns predicted class, confidence score, and probability for all classes.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess_fn(image).unsqueeze(0).to(device)

    # Tokenize the predefined text prompts for classification
    # These prompts serve as the classification targets
    text_inputs = tokenizer_fn(TEXT_PROMPTS).to(device)

    # Get logits (similarity) from the fine-tuned model
    # We use the fine-tuned model's logic for inference
    logits = model_ft(image_input, text_inputs[0].unsqueeze(0)) # Pass dummy text features to satisfy forward signature
    
    # Rerunning the core CLIP logic for image-to-text similarity for classification
    # The actual fine-tuning was done on the image encoder head.
    
    # Encode the text features separately for zero-shot style prediction
    with torch.no_grad():
        text_features = model_ft.clip.encode_text(text_inputs).float()
        
    # Get image feature after projection head
    image_feature = model_ft.proj(model_ft.clip.encode_image(image_input).float())

    # Calculate final logits (similarity) between the single image feature and all text features
    # Normalized image and text features are crucial for cosine similarity
    image_feature /= image_feature.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Final similarity calculation (logits)
    logits = (100.0 * image_feature @ text_features.T) # CLIP often scales logits by 100

    # Convert logits to probabilities
    probs = logits.softmax(dim=-1).cpu().numpy()[0]

    # Get the predicted class (highest probability)
    pred_idx = int(np.argmax(probs))
    return SELECTED_CLASSES[pred_idx], float(probs[pred_idx]), probs.tolist()

# --- Main Evaluation Function ---

def main(args):
    """Loads the model, runs inference on the test set, and calculates metrics."""
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    print(f"Loading CLIP model: {args.clip_model_name}...")

    # Load the base CLIP model and its preprocessing function
    model, preprocess = clip.load(args.clip_model_name, device=DEVICE, jit=False)

    # Check if a fine-tuned path was provided
    if args.model_path:
        # --- Fine-Tuned Mode ---
        print("Attempting to load fine-tuned model checkpoint...")
        
        # Instantiate the fine-tuned model wrapper
        ft_model = CLIPFineTune(model).to(DEVICE)
        
        # Load the fine-tuned state dictionary
        try:
            if not os.path.exists(args.model_path):
                # If path is given but file doesn't exist, raise error and exit
                raise FileNotFoundError(f"Model checkpoint not found at {args.model_path}")
                
            ft_model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
            print(f"✅ Successfully loaded fine-tuned model from {args.model_path}")
            EVAL_MODEL = ft_model # Use the fine-tuned model
            MODE = "Fine-Tuned"
            
        except Exception as e:
            print(f"❌ Error loading model state dictionary: {e}. Aborting.")
            sys.exit(1) # Exit if loading fails

    else:
        # --- Zero-Shot Mode ---
        print("Running in ZERO-SHOT mode: No model path provided.")
        print(f"Using base CLIP model ({args.clip_model_name}) directly.")
        ft_model = CLIPFineTune(model).to(DEVICE) 
        
        # Ensure the base model is frozen and set it as the evaluation model
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        EVAL_MODEL = model # Use the base CLIP model
        MODE = "Zero-Shot"

    # Set model to evaluation mode
    EVAL_MODEL.eval()

    # Load test data
    if not os.path.exists(args.test_csv_path):
        print(f"Error: Test CSV not found at {args.test_csv_path}. Aborting evaluation.")
        sys.exit(1)
        
    test_df = pd.read_csv(args.test_csv_path)
    print(f"Loaded {len(test_df)} samples from {args.test_csv_path}")

    y_true_labels = []
    y_pred_labels = []

    print("\nStarting inference on test dataset...")
    for idx, row in test_df.iterrows():
        img_path = row["ImagePath"]
        true_caption = row["Caption"]

        ground_truth_class = get_ground_truth_class(true_caption)

        if not os.path.exists(img_path):
            # print(f"Skipping image {img_path}: File not found.")
            continue
        
        # Only evaluate samples for which we can find a ground truth label
        if ground_truth_class not in SELECTED_CLASSES:
             # print(f"Skipping sample {idx}: Unknown ground truth class.")
             continue

        try:
            predicted_class, confidence, probabilities = predict_image_with_ft_model(
                img_path, ft_model, preprocess, clip.tokenize, DEVICE
            )

            y_true_labels.append(ground_truth_class)
            y_pred_labels.append(predicted_class)

            if (idx + 1) % 100 == 0 or (idx + 1) == len(test_df):
                print(f"Processed {idx + 1}/{len(test_df)} test samples...")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    print("\nInference complete.")

    # --- Evaluation Metrics ---
    if y_true_labels and y_pred_labels:
        # Calculate and print metrics
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        
        print(f"\n--- Evaluation Results ({len(y_true_labels)} Valid Samples) ---")
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        
        print("\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels, labels=SELECTED_CLASSES, zero_division=0))

        # Plot confusion matrix
        if HAS_PLOTTING:
            os.makedirs(args.results_dir, exist_ok=True)
            output_filename = f'confusion_matrix_{os.path.basename(args.model_path).replace(".pt", "")}_{accuracy*100:.2f}perc.png'
            output_path = os.path.join(args.results_dir, output_filename)
            
            plot_confusion_matrix(y_true_labels, y_pred_labels, SELECTED_CLASSES, 
                                  f'Confusion Matrix - Fine-tuned CLIP ({os.path.basename(args.model_path)})',
                                  output_path=output_path)
        else:
            print("Cannot generate confusion matrix plot: plotting libraries not available.")

    else:
        print("No valid samples were processed for evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for fine-tuned CLIP model.")

    # --- File Paths ---
    parser.add_argument("--test_csv_path", type=str, required=True, 
                        help="Path to the test CSV file (must contain 'ImagePath' and 'Caption').")
    parser.add_argument("--model_path", type=str, required=True, 
                        help="Path to the fine-tuned model checkpoint file (e.g., 'clip_epoch_75.pt').")
    parser.add_argument("--results_dir", type=str, default="results", 
                        help="Directory to save the confusion matrix plot.")

    # --- Model Configuration ---
    parser.add_argument("--clip_model_name", type=str, default="ViT-B/32", 
                        help="Name of the base CLIP model used (e.g., 'ViT-B/32').")

    args = parser.parse_args()
    main(args)
