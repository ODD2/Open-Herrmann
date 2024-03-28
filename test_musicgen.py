# %%
import torch
import scipy
import librosa
import torchaudio
from transformers import AutoProcessor, MusicgenForConditionalGeneration

audio, sr = librosa.load("original.wav", mono=True, sr=32000)

processor = AutoProcessor.from_pretrained(
    "facebook/musicgen-medium",
)
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-medium",
)
# %%
inputs = processor(
    text=["a soft music with romantic and calm vibes."],
    audio=[audio[:len(audio) // 2]],
    sampling_rate=32000,
    padding=True,
    return_tensors="pt",
)
# %%
model = model.to("cuda")
inputs = inputs.to("cuda")

audio_values = model.generate(**inputs, max_length=256 * 6)

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

# %%
