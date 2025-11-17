# 🌟 Fine-Tuning CLIP for Chest X-ray Classification

This repository contains the code and resources for fine-tuning the **CLIP (Contrastive Language-Image Pre-training)** model to perform multi-class classi    fication on Chest X-ray images, specifically identifying five common pulmonary diseases: **atelectasis, mass, effusion, infiltration, and nodule**.

The fine-tuning process focuses solely on adding a trainable projection head to the image encoder while leveraging CLIP's powerful zero-shot image-text understanding capabilities.

---

## 📂 Repository Structure

The project is organized to clearly separate data, core training/inference logic, and execution wrappers.

``` bash
DATA/
│── DATA_ANALYSIS/
│       ├── Results/
│       └── DataAnalysis_5class.py  (Contains dataset description of the 5 classes)
│       └── DataAnalysis.py         (Contains dataset description of all the classes)
│── DATASET/
│       ├── balanced_dataset.csv    (Combined dataset after preprocessing)
│       ├── DATASET_PREPR.py        (Main preprocessing code)
│       ├── DATASETSPLIT.py         (Dataset split code)
│       ├── PATHCHANGE.py           (Changes dataset path acc to local comp)
│       ├── test.csv
│       ├── train.csv
│       └── val.csv
│       └── requirements.txt        (For installing dependencies)
├── FINALDATA/
│       └── images    (Folder containing path to the images)
│── EXERCISES/
│       ├── similarity_calc.ipynb
│       └── softmax.ipynb
├── EXTRAS/
├── MAIN/
│   ├── My_Run/
│   │   ├── results/                (Conf Mat, Weights saved from Google Colab Run)
│   │   ├── infer.py                
│   │   └── train.py
│   └── To_Run_Locally/
│       ├── infer_cli.py
│       └── train_cli.py
├── WORKING_NOTEBOOKS/              (Notebooks used for personal use and training)
│   ├── Finetuning_CLIP.ipynb      
│   └── Openai_clip.ipynb
|   └── Readme.MD                   (References used for model training and inference)          
└── README.md
```

---

## 🚀 Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/lakshminarayanan678/Finetuning_CLIP.git
    cd {your repo path}
    ```

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # macOS/Linux
    # venv\Scripts\activate   # Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```


## ⚙️ Running the Model Locally

All execution can be managed via command-line interface (CLI) scripts for reproducability. The scripts in `MAIN/My_Run/` offer maximum flexibility, while the wrappers in `MAIN/To_Run_Locally/` simplify common execution paths.

### 1. Training the Model (`train.py`)

The core training logic is in `MAIN/My_Run/train.py`.

#### Option A: Using the Simplified CLI Wrapper (train_cli.py)
This wrapper script may contain pre-set paths to simplify execution.
```bash
 python MAIN/To_Run_Locally/train_cli.py \
    --train_csv_path DATA/DATASET/train.csv \
    --val_csv_path DATA/DATASET/val.csv \
    --save_dir MAIN/My_Run/checkpoints \
    --model_name ViT-B/32 \
    --num_epochs \
    --batch_size 32 \
    --learning_rate 1e-3 \
    --checkpoint_path \
    --run_name \ 
    --use_wandb # Add this flag to enable Weights & Biases logging
```
(The default for checkpoint path is none, so u can use it only when u have pretrained weights already) 

💡 Resuming Training
To continue from a saved checkpoint, use the --checkpoint_path and set the correct --start_epoch for accurate logging:

```bash
# Example continuation from epoch 80
--checkpoint_path MAIN/My_Run/checkpoints/clip_epoch_80.pt --start_epoch 81
```

(Consult train_cli.py for any specific default arguments or configuration)

#### Option B: Using the Core Script (Full Control)

Use this option to specify all paths and hyperparameters manually.

```bash
python MAIN/My_Run/train.py

```


### 2. Evaluation (Inference) (infer.py)
The evaluation and metric calculation logic is in MAIN/My_Run/infer.py.


#### Option A: Using the Simplified CLI Wrapper (infer_cli.py)

```bash
python MAIN/To_Run_Locally/infer_cli.py \
    --test_csv_path DATA/DATASET/test.csv \
    --model_path MAIN/My_Run/checkpoints/clip_ft_best.pt \
    --results_dir MAIN/My_Run/results \
    --clip_model_name ViT-B/32
```

(Consult infer_cli.py to confirm the default model checkpoint and test CSV paths it uses.)


#### Option B: Using the Core Script (Full Control)

You must specify the path to the model checkpoint you wish to evaluate.

```bash
python MAIN/My_Run/infer.py
```

The script will print the Classification Report and save a confusion matrix plot to the specified results directory.
