# Herrmann-1.0
This project re-implements the Herrmann-1.0 presented in [ICASSP 2024](https://audiomatic-research.github.io/herrmann-1/).

## Installation
```bash
# AudioCraft Preliminaries
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install audiocraft
conda install "ffmpeg<5" -c conda-forge
# MusicGen & BLIP-2
pip install git+https://github.com/huggingface/transformers.git
pip install accelerate
# Herrmann
pip install scenedetect[opencv]
# Whisper
pip install  git+https://github.com/openai/whisper.git
# Gemini
pip install  google-generativeai
```
## Gemini API Key
Please follow the [instruction](https://ai.google.dev/?gad_source=1&gclid=Cj0KCQjwqpSwBhClARIsADlZ_Tn0bUm4aPKdQysdzuc2_SDR04DzhyERWWqNW8qrZWKaU0qx8uiTDCkaAoISEALw_wcB) from Google Gemini to create your own API key, and configure it as the environment variable **GOOGLE_API_KEY**.

## Main Function
Run the following command to process the 'original.mp4' video file to generate the 'musicgen_out.wav' and 'out.json' file.
```bash
export TOKENIZERS_PARALLELISM=false
python -m herrmann-wo-annote
```
### Sample out.json
The following is a sample of the out.json file, including the video duration, the image captions from BLIP-2, the video transcripts from Whisper, emotions from CLIP, the high-level description prompt (hl_prompt), the low-level music prompt (ll_prompt) for MusicGen.
```json
{
    "captions": [
        "the two people are standing on the deck of a boat",
        "a woman in a white dress standing next to a ship",
        "a man is standing on a boat with his hand on his chin",
        "a woman in a white dress standing next to a ship",
        "a young man is standing on a boat with a sunset in the background",
        "a woman with red hair standing in front of a ship",
        "the couple is standing on the deck of a boat at sunset",
        "a man and woman standing on the deck of a boat at sunset",
        "a woman with red hair and a blue shirt",
        "the young man is talking to the woman on the boat",
        "a woman with red hair is looking at a man",
        "a woman with blue eyes is looking at a man",
        "a woman with her eyes closed and a man looking at her",
        "a man and woman are standing on a ship",
        "a man and woman are dancing together on a ship",
        "a man and woman standing on a boat",
        "a man and woman standing next to each other"
    ],
    "duration": 30.166666666666668,
    "emotions": [
        "beautiful(48.54%)",
        "dreamy(23.8%)",
        "energizing/pump-up(7.4%)"
    ],
    "hl_prompt": "Given the following image captions from a video: 1) the two people are standing on the deck of a boat 2) a woman in a white dress standing next to a ship 3) a man is standing on a boat with his hand on his chin 4) a woman in a white dress standing next to a ship 5) a young man is standing on a boat with a sunset in the background 6) a woman with red hair standing in front of a ship 7) the couple is standing on the deck of a boat at sunset 8) a man and woman standing on the deck of a boat at sunset 9) a woman with red hair and a blue shirt 10) the young man is talking to the woman on the boat 11) a woman with red hair is looking at a man 12) a woman with blue eyes is looking at a man 13) a woman with her eyes closed and a man looking at her 14) a man and woman are standing on a ship 15) a man and woman are dancing together on a ship 16) a man and woman standing on a boat 17) a man and woman standing next to each other and the following transcriptions: 1)  I said you might be up to it. 2)  Give me your hand. 3)  Now close your eyes. 4)  Go on. 5)  Now step up. 6)  Now hold on to the railing. and given that the sentiments of the video are: beautiful(48.54%), dreamy(23.8%), energizing/pump-up(7.4%) describe the music that would fit such a video. Your output will be fed to a text to music model. To help you out, here are some prompts that worked well with the model: 1) Pop dance track with catchy melodies, tropical percussion, and upbeat rhythms, perfect for the beach 2) classic reggae track with an electronic guitar solo 3) earthy tones, environmentally conscious, ukulele-infused, harmonic, breezy, easygoing, organic instrumentation, gentle grooves 4) lofi slow bpm electro chill with organic samples 5) violins and synths that inspire awe at the finiteness of life and the universe 6) 80s electronic track with melodic synthesizers, catchy beat and groovy bass. Give me only the description of the music without any explanation. Give me a single description.",
    "ll_prompt": "Dreamy and inspiring electro-pop track with catchy melodies, uplifting synths, and a steady beat, perfect for a summery and romantic atmosphere on the water.",
    "transcripts": [
        " I said you might be up to it.",
        " Give me your hand.",
        " Now close your eyes.",
        " Go on.",
        " Now step up.",
        " Now hold on to the railing."
    ]
}
```



## Experiments
You can test on all the sub-modules with test_*.py files to adjust the parameters and validate the gemini token.

