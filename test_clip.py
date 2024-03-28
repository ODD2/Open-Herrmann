import clip
import torch
import numpy as np
from glob import glob
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)
emotions = [
    "amusing",
    "annoying",
    "anxious/tense",
    "beautiful",
    "calm/relaxing/serene",
    "dreamy",
    "energizing/pump-up",
    "erotic/desirous",
    "indignant/defiant",
    "joyful/cheerful",
    "sad/depressing",
    "scary/fearful",
    "triumphant/heroic"
]

files = glob("./frame_cut/*.png")
image_probs = []
with torch.no_grad():
    text = clip.tokenize(
        [
            f"a figure with the {t} emotion" for t in emotions
        ]
    ).to(device)
    text_features = model.encode_text(text)
    for file in files:
        image = preprocess(Image.open(file)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.flatten(0).softmax(dim=-1).cpu().numpy()
        image_probs.append(probs)
    probs = sum(image_probs) / len(image_probs)

print("Label probs:")
indices = np.argsort(probs)[-3:]
for idx in indices:
    print(emotions[idx], round(probs[idx] * 100, 2))
