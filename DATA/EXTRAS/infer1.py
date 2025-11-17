
import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from typing import List, Tuple, Dict

# Add parent directory to path to use local CLIP code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Try importing visualization libraries (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False

# Disease classes we're working with
SELECTED_CLASSES = ["Pneumonia", "Effusion", "Emphysema", "Fibrosis", "Hernia"]

# Text descriptions for zero-shot classification
TEXT_PROMPTS = [
    "a chest X-ray showing pneumonia",
    "a chest X-ray showing pleural effusion", 
    "a chest X-ray showing emphysema",
    "a chest X-ray showing pulmonary fibrosis",
    "a chest X-ray showing diaphragmatic hernia"
]


def load_clip_model(device: str):
    """Load Standard CLIP model from local repository"""
    import clip
    model, preprocess = clip.load("ViT-B/32", device=device)
    print(f"Loaded Standard CLIP model on {device}")
    return model, preprocess, clip.tokenize, "openai-clip"



@torch.no_grad()
def predict_single_image(image_path: str, model, preprocess, tokenizer, backend: str, device: str) -> Tuple[str, float, List[float]]:
    """
    Predict disease class for a single X-ray image using zero-shot CLIP.
    Returns predicted class, confidence score, and probability for all classes.
    """
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_input = preprocess(image).unsqueeze(0).to(device)

    # Standard CLIP processing
    text_inputs = tokenizer(TEXT_PROMPTS).to(device)
    image_features = model.encode_image(image_input)
    text_features = model.encode_text(text_inputs)
    
    # Normalize features (important for similarity calculation)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # Calculate similarity and convert to probabilities
    logits = (100.0 * image_features @ text_features.T)
    probs = logits.softmax(dim=-1).cpu().numpy()[0]
  
    # Get the predicted class (highest probability)
    pred_idx = int(np.argmax(probs))
    return SELECTED_CLASSES[pred_idx], float(probs[pred_idx]), probs.tolist()


def print_single_prediction(image_path: str, predicted_class: str, confidence: float, 
                           probabilities: List[float], model_name: str):
    """Display prediction results for a single image"""
    print(f"\n{'='*60}")
    print(f"{model_name} - {os.path.basename(image_path)}")
    print(f"Predicted: {predicted_class} (Confidence: {confidence*100:.2f}%)")
    print(f"{'-'*60}")
    print("Probabilities for all classes:")
    for cls, prob in zip(SELECTED_CLASSES, probabilities):
        bar = "█" * int(prob * 40)
        print(f"  {cls:12s}: {prob*100:5.1f}% {bar}")
    print(f"{'='*60}\n")


def load_ground_truth(labels_file: str) -> Dict[str, str]:
    """Load ground truth labels from file for evaluation"""
    labels = {}
    if not os.path.exists(labels_file):
        return labels
    
    with open(labels_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if line and not line.startswith('#'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 2:
                    labels[parts[0]] = parts[1]
    return labels


def plot_confusion_matrix(y_true: List[str], y_pred: List[str], model_name: str, output_path: str):
    """Generate and save confusion matrix visualization"""
    if not HAS_PLOTTING:
        return
    
    from sklearn.metrics import confusion_matrix
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=SELECTED_CLASSES)
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=SELECTED_CLASSES, yticklabels=SELECTED_CLASSES)
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_predictions_comparison(y_true: List[str], y_pred: List[str], 
                                confidences: List[float], model_name: str, output_path: str):
    """
    Create visualization comparing predictions to ground truth.
    Shows scatter plot with connecting lines and confidence distribution.
    """
    if not HAS_PLOTTING:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convert class names to numbers for plotting
    class_to_int = {cls: i for i, cls in enumerate(SELECTED_CLASSES)}
    y_true_int = [class_to_int[cls] for cls in y_true]
    y_pred_int = [class_to_int[cls] for cls in y_pred]
    x = np.arange(len(y_true))
    
    # Left plot: Predictions vs Ground Truth with connecting lines
    ax1.scatter(x, y_true_int, c='green', marker='o', s=100, alpha=0.6, label='Ground Truth', zorder=3)
    ax1.scatter(x, y_pred_int, c='red', marker='x', s=100, alpha=0.6, label='Predicted', zorder=3)
    
    # Draw lines connecting ground truth to predictions (green=correct, red=wrong)
    for i in range(len(x)):
        color = 'green' if y_true_int[i] == y_pred_int[i] else 'red'
        ax1.plot([x[i], x[i]], [y_true_int[i], y_pred_int[i]], color=color, alpha=0.4, linewidth=1, zorder=1)
    
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Class')
    ax1.set_title(f'Predictions vs Ground Truth - {model_name}')
    ax1.set_yticks(range(len(SELECTED_CLASSES)))
    ax1.set_yticklabels(SELECTED_CLASSES)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Right plot: Confidence distribution for correct vs incorrect predictions
    correct = [conf for i, conf in enumerate(confidences) if y_true[i] == y_pred[i]]
    incorrect = [conf for i, conf in enumerate(confidences) if y_true[i] != y_pred[i]]
    
    bins = np.linspace(0, 1, 20)
    ax2.hist(correct, bins=bins, alpha=0.6, color='green', label=f'Correct ({len(correct)})', edgecolor='black')
    ax2.hist(incorrect, bins=bins, alpha=0.6, color='red', label=f'Incorrect ({len(incorrect)})', edgecolor='black')
    ax2.set_xlabel('Confidence Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Confidence Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def batch_inference(data_dir: str, labels_file: str, model, preprocess, tokenizer, 
                   backend: str, device: str, model_name: str):
    """Run inference on all images in the directory and evaluate results"""
    # Get list of all image files
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"No images found in {data_dir}")
        return
    
    print(f"\nProcessing {len(image_files)} images...")
    ground_truth = load_ground_truth(labels_file)
    
    # Store results
    y_true, y_pred, results = [], [], []
    
    # Process each image
    for i, img_file in enumerate(image_files, 1):
        try:
            predicted_class, confidence, _ = predict_single_image(
                os.path.join(data_dir, img_file), model, preprocess, tokenizer, backend, device
            )
            
            true_label = ground_truth.get(img_file, 'UNKNOWN')
            results.append({'image': img_file, 'true': true_label, 'predicted': predicted_class, 'confidence': confidence})
            
            # Collect for metrics only if we have ground truth
            if true_label in SELECTED_CLASSES:
                y_true.append(true_label)
                y_pred.append(predicted_class)
            
            # Show progress
            if i % 50 == 0 or i == len(image_files):
                print(f"Progress: {i}/{len(image_files)}", end='\r')
        except Exception as e:
            print(f"\nError processing {img_file}: {e}")
    
    print(f"\nCompleted {len(results)} predictions")
    
    # Calculate and display metrics if we have ground truth
    if y_true and y_pred:
        try:
            from metrics_analysis import calculate_metrics, print_metrics
            metrics = calculate_metrics(y_true, y_pred, SELECTED_CLASSES)
            print_metrics(metrics, model_name)
            
            # Generate visualizations
            os.makedirs('results', exist_ok=True)
            
            # Save confusion matrix
            cm_path = f"results/confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
            plot_confusion_matrix(y_true, y_pred, model_name, cm_path)
            
            # Save prediction comparison
            confidences = [r['confidence'] for r in results if r['true'] in SELECTED_CLASSES]
            pred_path = f"results/predictions_analysis_{model_name.lower().replace(' ', '_')}.png"
            plot_predictions_comparison(y_true, y_pred, confidences, model_name, pred_path)
            
            print(f"\nVisualization saved to 'results/' folder")
        except ImportError as e:
            print(f"Error: {e}")


def main():
    """Main function to handle command line arguments and run inference"""
    parser = argparse.ArgumentParser(description='CLIP Chest X-ray Classification')
    parser.add_argument('--image', help='Path to single image for inference')
    parser.add_argument('--batch', action='store_true', help='Run batch inference on all images')
    parser.add_argument('--data', default='finalData', help='Directory containing images')
    parser.add_argument('--labels', default='finalData/image_labels.txt', help='Ground truth labels file')
    parser.add_argument('--model', default='clip', choices=['clip', 'biomed'], help='Model: clip (standard) or biomed (medical)')
    args = parser.parse_args()
    
    # Check if user provided either single image or batch mode
    if not args.image and not args.batch:
        print("Error: Please specify either --image or --batch")
        return
    
    # Setup device (use GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load the selected model
    try:
        if args.model == 'clip':
            model, preprocess, tokenizer, backend = load_clip_model(device)
            model_name = "Standard CLIP"
        else:
            model, preprocess, tokenizer, backend = load_biomedclip_model(device)
            model_name = "BiomedCLIP"
        model.eval()
    except Exception as e:
        print(f"Failed to load model: {e}")
        return
    
    # Run inference based on mode
    if args.image:
        # Single image mode
        if not os.path.exists(args.image):
            print(f"Image not found: {args.image}")
            return
        predicted_class, confidence, probabilities = predict_single_image(
            args.image, model, preprocess, tokenizer, backend, device
        )
        print_single_prediction(args.image, predicted_class, confidence, probabilities, model_name)
    else:
        # Batch mode
        if not os.path.exists(args.data):
            print(f"Directory not found: {args.data}")
            return
        batch_inference(args.data, args.labels, model, preprocess, tokenizer, backend, device, model_name)


if __name__ == '__main__':
    main()