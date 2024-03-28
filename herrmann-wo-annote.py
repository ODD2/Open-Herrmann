import os
import cv2
import math
import clip
import json
import scipy
import torch
import whisper
import numpy as np
from glob import glob
from PIL import Image
import google.generativeai as genai
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from transformers import AutoProcessor, MusicgenForConditionalGeneration

SCENE_CUT_DIR = "scene_cut"
FRAME_CUT_DIR = "frame_cut"

VIDEO_FILE = "original.mp4"
MIN_CUTS = 15
BLIP2_MODEL = "Salesforce/blip2-opt-2.7b"
WHISPER_MODEL = "small"
MUSICGEN_MODEL = "facebook/musicgen-medium"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def scene_crop_detect():
    scene_list = detect(VIDEO_FILE, AdaptiveDetector())

    # save each scene cut
    split_video_ffmpeg(VIDEO_FILE, scene_list, output_dir=SCENE_CUT_DIR)

    cut_frame_indices = []
    if (len(scene_list) >= MIN_CUTS):
        for (a, b) in scene_list:
            cut_frame_indices.append((a.frame_num + b.frame_num) // 2)
    else:
        # count ratio
        per_scene_frame_num = []
        for (a, b) in scene_list:
            per_scene_frame_num.append(b.frame_num - a.frame_num)
        total = sum(per_scene_frame_num)
        ratio = [
            max(1, round(MIN_CUTS * (frame_num / total)))
            for frame_num in per_scene_frame_num
        ]
        ratio[-1] += max(MIN_CUTS - sum(ratio), 0)

        for r, f, (a, b) in zip(ratio, per_scene_frame_num, scene_list):
            hop = f // (r + 1)
            for i in range(r):
                cut_frame_indices.append(a.frame_num + (1 + i) * hop)

    os.makedirs(FRAME_CUT_DIR, exist_ok=True)
    cap = cv2.VideoCapture(VIDEO_FILE)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    j = 0
    for i in range(round(frames)):
        ret, frame = cap.read()
        if (i in cut_frame_indices):
            cv2.imwrite(os.path.join(FRAME_CUT_DIR, f"{('00'+str(j))[-2:]}.png"), frame)
            j += 1
    cap.release()

    return frames / fps


def image_captioning():
    processor = AutoProcessor.from_pretrained(BLIP2_MODEL)
    # by default `from_pretrained` loads the weights in float32
    # we load in float16 instead to save memory
    model = Blip2ForConditionalGeneration.from_pretrained(
        BLIP2_MODEL,
        device_map={"": 0},
        torch_dtype=torch.float16
    )

    model.to(DEVICE)

    descs = []
    for file in sorted(glob(f"{FRAME_CUT_DIR}/*.png")):
        image = Image.open(file)

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

        descs.append(generated_text)
    return descs


def emotion_detection():
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
    files = sorted(glob(f"{FRAME_CUT_DIR}/*.png"))
    image_probs = []
    with torch.no_grad():
        text = clip.tokenize(
            [
                f"a figure with the {t} emotion" for t in emotions
            ]
        ).to(device)
        for file in files:
            image = preprocess(Image.open(file)).unsqueeze(0).to(device)
            logits_per_image, _ = model(image, text)
            probs = logits_per_image.flatten(0).softmax(dim=-1).cpu().numpy()
            image_probs.append(probs)
        probs = sum(image_probs) / len(image_probs)

    indices = np.argsort(probs)[-3:]
    return [
        f"{emotions[idx]}({round(probs[idx] * 100, 2)}%)"
        for idx in indices[::-1]
    ]


def dialog_extraction():
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(VIDEO_FILE)
    return [seg['text'] for seg in result["segments"]]


def construct_high_level_prompt(captions, emotions, transcripts):
    prompt = (
        "Given the following image captions from a video: " +
        " ".join([f"{i+1}) {c}" for i, c in enumerate(captions)]) + " "
    )
    prompt += (
        "and the following transcriptions: " + " ".join([f"{i+1}) {c}" for i, c in enumerate(transcripts)]) + " "
    )
    prompt += "and given that the sentiments of the video are: " + ", ".join(emotions) + " "
    prompt += (
        "describe the music that would fit such a video. Your output will be fed to a text to music model. " +
        "To help you out, here are some prompts that worked well with the model: " +
        "1) Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach " +
        "2) classic reggae track with an electronic guitar solo " +
        "3) earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves " +
        "4) lofi slow bpm electro chill with organic samples " +
        "5) violins and synths that inspire awe at the finiteness of life and the universe " +
        "6) 80s electronic track with melodic synthesizers, catchy beat and groovy bass. " +
        "Give me only the description of the music without any explanation. Give me a single description."
    )
    return prompt


def fetch_low_level_prompt(hl_prompt):
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

    genai.configure(api_key=GOOGLE_API_KEY)

    model = genai.GenerativeModel('gemini-pro')

    response = model.generate_content(hl_prompt)

    return response.text


def generate_music_with_text_prompt(ll_prompt, duration):

    processor = AutoProcessor.from_pretrained(MUSICGEN_MODEL)
    model = MusicgenForConditionalGeneration.from_pretrained(MUSICGEN_MODEL)

    inputs = processor(
        text=[ll_prompt],
        padding=True,
        return_tensors="pt",
    )

    model = model.to("cuda")
    inputs = inputs.to("cuda")

    audio_values = model.generate(**inputs, max_length=round(256 * math.ceil(duration / 5)))

    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())


if __name__ == "__main__":
    duration = scene_crop_detect()
    captions = image_captioning()
    emotions = emotion_detection()
    transcripts = dialog_extraction()
    hl_prompt = construct_high_level_prompt(captions, emotions, transcripts)
    ll_prompt = fetch_low_level_prompt(hl_prompt)
    generate_music_with_text_prompt(ll_prompt, duration)
    with open("out.json", "w") as f:
        json.dump(
            dict(
                duration=duration,
                captions=captions,
                emotions=emotions,
                transcripts=transcripts,
                hl_prompt=hl_prompt,
                ll_prompt=ll_prompt
            ),
            f,
            sort_keys=True,
            indent=4
        )
