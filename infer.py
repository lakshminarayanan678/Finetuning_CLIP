import os
import sys
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

# Disease classes we're working with
SELECTED_CLASSES = ["atelectasis", "mass", "effusion", "infiltration", "nodule"]

# Text descriptions for zero-shot classification
TEXT_PROMPTS = [
    "Posteroanterior view chest X-ray showing atelectasis.",
    "Posteroanterior view chest X-ray showing mass.",
    "Posteroanterior view chest X-ray showing effusion.",
    "Posteroanterior view chest X-ray showing infiltration.",
    "Posteroanterior view chest X-ray showing nodule."
]

# Mapping for extracting ground truth class from caption
caption_to_class_map = {
    "Posteroanterior view chest X-ray showing atelectasis.": "atelectasis",
    "Posteroanterior view chest X-ray showing mass.": "mass",
    "Posteroanterior view chest X-ray showing effusion.": "effusion",
    "Posteroanterior view chest X-ray showing infiltration.": "infiltration",
    "Posteroanterior view chest X-ray showing nodule.": "nodule"
}

def get_ground_truth_class(caption: str) -> str:
    """Extracts the ground truth class from a given caption."""
    for prompt_key, cls_name in caption_to_class_map.items():
        if prompt_key.lower() in caption.lower():
            return cls_name
    return "UNKNOWN" # Default if no match found


def plot_confusion_matrix(y_true: list, y_pred: list, classes: list, title: str, output_path: str = 'confusion_matrix.png'):
    """Generate and save confusion matrix visualization."""
    if not HAS_PLOTTING:
        print("Cannot plot confusion matrix: plotting libraries not available.")
        return

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    plt.close()

def print_single_prediction(image_path: str, predicted_class: str, confidence: float,
                           probabilities: list, model_name: str):
    """Display prediction results for a single image"""
    print(f"\n{'='*60}")
    print(f"{model_name} - {os.path.basename(image_path)}")
    print(f"Predicted: {predicted_class} (Confidence: {confidence*100:.2f}%)")
    print(f"{'='*60}")
    print("Probabilities for all classes:")
    for cls, prob in zip(SELECTED_CLASSES, probabilities):
        bar = "█" * int(prob * 40)
        print(f"  {cls:12s}: {prob*100:5.1f}% {bar}")
    print(f"{'='*60}\n")

# Redefine the Dataset class if not already in scope (or ensure it's from a previous cell)
class ChestXray(Dataset):
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
    def __init__(self, clip_model, embed_dim=512):
        super().__init__()
        self.clip = clip_model
        self.proj = nn.Linear(512, 512)   # trainable head

    def forward(self, images, text_tokens):
        with torch.no_grad(): # Keep CLIP base model frozen during inference
            img_f = self.clip.encode_image(images)
            txt_f = self.clip.encode_text(text_tokens)

        img_f = img_f / img_f.norm(dim=1, keepdim=True)
        txt_f = txt_f / txt_f.norm(dim=1, keepdim=True)
        img_f = img_f.float()
        txt_f = txt_f.float()

        # Apply the projection head
        img_f = self.proj(img_f)

        logits = img_f @ txt_f.t()
        return logits

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load the base CLIP model and its preprocessing function
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)

# Instantiate your fine-tuned model wrapper
ft_model = CLIPFineTune(model).to(device)

# Loadin state dict
try:
    model_path = '/content/drive/MyDrive/COLAB/clip_epoch_75.pt'
    ft_model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"Successfully loaded fine-tuned model from {model_path}")
except FileNotFoundError:
    print(f"Error: Model checkpoint not found at {model_path}. Please ensure the file exists.")
except Exception as e:
    print(f"Error loading model state dictionary: {e}")

ft_model.eval()
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
    text_inputs = tokenizer_fn(TEXT_PROMPTS).to(device)

    # Get logits from the fine-tuned model
    logits = model_ft(image_input, text_inputs)

    # Convert logits to probabilities
    probs = logits.softmax(dim=-1).cpu().numpy()[0]

    # Get the predicted class (highest probability)
    pred_idx = int(np.argmax(probs))
    return SELECTED_CLASSES[pred_idx], float(probs[pred_idx]), probs.tolist()


# Batch Inference and Evaluation
test_csv_path = r"/content/drive/MyDrive/COLAB/test.csv"

if not os.path.exists(test_csv_path):
    print(f"Error: test.csv not found at {test_csv_path}. Please ensure the file exists and is correctly path-mapped.")
else:
    test_df = pd.read_csv(test_csv_path)
    print(f"Loaded {len(test_df)} samples from test.csv")

    y_true_labels = []
    y_pred_labels = []
    all_confidences = []

    print("\nStarting inference on test dataset...")
    for idx, row in test_df.iterrows():
        img_path = row["ImagePath"]
        true_caption = row["Caption"]

        # Get ground truth class from caption
        ground_truth_class = get_ground_truth_class(true_caption)

        if not os.path.exists(img_path):
            print(f"Skipping image {img_path}: File not found.")
            continue

        try:
            predicted_class, confidence, probabilities = predict_image_with_ft_model(
                img_path, ft_model, preprocess, clip.tokenize, device
            )

            if ground_truth_class != "UNKNOWN": # Only include if we found a valid ground truth class
                y_true_labels.append(ground_truth_class)
                y_pred_labels.append(predicted_class)
                all_confidences.append(confidence)

            if (idx + 1) % 50 == 0 or (idx + 1) == len(test_df):
                print(f"Processed {idx + 1}/{len(test_df)} test samples...")

        except Exception as e:
            print(f"Error processing image {img_path}: {e}")

    print("\nInference complete.")

    if y_true_labels and y_pred_labels:
        # Calculate and print metrics
        accuracy = accuracy_score(y_true_labels, y_pred_labels)
        print(f"\n--- Evaluation Results ---")
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_true_labels, y_pred_labels, labels=SELECTED_CLASSES, zero_division=0))

        # Plot confusion matrix
        if HAS_PLOTTING:
            os.makedirs('results', exist_ok=True)
            plot_confusion_matrix(y_true_labels, y_pred_labels, SELECTED_CLASSES, 
                                  f'Confusion Matrix - Fine-tuned CLIP (epoch75) ({accuracy*100:.2f}%)',
                                  output_path='results/confusion_matrix_finetuned_clip_epoch75.png')
            print(f"Confusion matrix saved to results/confusion_matrix_finetuned_clip_epoch75.png")
        else:
            print("Cannot generate confusion matrix plot: plotting libraries not available.")

    else:
        print("No valid samples processed for evaluation.")
