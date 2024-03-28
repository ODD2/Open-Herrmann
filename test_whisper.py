import whisper

model = whisper.load_model("large")
result = model.transcribe("original.mp4")
for seg in result["segments"]:
    print(f"start={seg['start']} stop={seg['end']} text={seg['text']}")
