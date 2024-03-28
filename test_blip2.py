import torch
import requests
from PIL import Image


from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# by default `from_pretrained` loads the weights in float32
# we load in float16 instead to save memory
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b",
    device_map={"": 0},
    torch_dtype=torch.float16
)

# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
# prompt = "Question: how many cats are there? Answer:"

url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png'
image = Image.open(requests.get(url, stream=True).raw)
# prompt = "Question: what country does the image located at ? Answer:"

with torch.no_grad():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = processor(
        images=image,
        # text=prompt,
        return_tensors="pt"
    ).to(
        device="cuda",
        dtype=torch.float16
    )

    generated_ids = model.generate(
        **inputs
    )
    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0].strip()

print(generated_text)
