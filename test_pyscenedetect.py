import os
import cv2
import math
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
SCENE_CUT_DIR = "scene_cut"
FRAME_CUT_DIR = "frame_cut"

scene_list = detect('original.mp4', AdaptiveDetector())

# save each scene cut
# split_video_ffmpeg('original.mp4', scene_list, output_dir=SCENE_CUT_DIR)


MIN_CUTS = 15
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

print(cut_frame_indices)

os.makedirs(FRAME_CUT_DIR, exist_ok=True)
cap = cv2.VideoCapture("./original.mp4")
j = 0
for i in range(round(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    ret, frame = cap.read()
    if (i in cut_frame_indices):
        cv2.imwrite(os.path.join(FRAME_CUT_DIR, f"{j}.png"), frame)
        j += 1
cap.release()
