import torch
from PIL import Image
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

img_path = "/content/Finetuning_CLIP/DATA/FINALDATA/images/00000008_002.png"
image = preprocess(Image.open(img_path)).unsqueeze(0)
text = tokenizer(["Posteroanterior view chest X-ray showing mass", "Posteroanterior view chest X-ray showing atelectasis", "Posteroanterior view chest X-ray showing effusion", "Posteroanterior view chest X-ray showing nodule"])

with torch.no_grad(), torch.cuda.amp.autocast():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]


import clip
import torch
import PIL
import matplotlib.pyplot as plt


# OpenAI CLIP model and preprocessing
model, preprocess = clip.load("ViT-B/32", jit=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(device)

from PIL import Image

img = Image.open("/content/drive/MyDrive/DATA/00000010_000.png").convert("RGB")
image_input = preprocess(img).unsqueeze(0).to(device) #what does the unsqueeze do? # PREPROCESS
print("Image size:",img.size)
print("image_input:",image_input.shape)
text_labels= ("Posteroanterior view chest X-ray showing mass.", "Posteroanterior view chest X-ray showing atelectasis.", "Posteroanterior view chest X-ray showing effusion.", "Posteroanterior view chest X-ray showing nodule.", "Posteroanterior view chest X-ray showing infiltration.")
text_inputs = clip.tokenize(text_labels).to(device) # TOKENIZE
print("text_inputs:",text_inputs.shape)

# Calculate image and text features
with torch.no_grad(): #What does torch.no_grad do?
    image_features = model.encode_image(image_input) #ENCODE_IMAGE
    print("image_features:",image_features.shape)
    text_features = model.encode_text(text_inputs) #ENCODE_TEXT
    print("text_features:",text_features.shape)

# Normalize the features
image_features /= image_features.norm(dim=-1, keepdim=True)
print("image_features after norm:",image_features.shape)
text_features /= text_features.norm(dim=-1, keepdim=True)
print("text_features after norm:", text_features.shape)

# Calculate similarity between image and text features
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1) #HOW COMPUTATION HAPPENS HERE?
values, indices = similarity[0].topk(1)

figure, ax = plt.subplots(1,1)
ax.imshow(img)
ax.set_title(f"Predicted Label:{text_labels[indices.item()]}")
plt.show()
