import cv2
import os

video_path = 'WhatsApp Video 2024-02-17 at 13.48.31_a4439ec8.mp4'

output_dir = 'output_frames'
os.makedirs(output_dir, exist_ok=True)

vidcap = cv2.VideoCapture(video_path)

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        cv2.imwrite(os.path.join(output_dir, f"frame_{count:04d}.jpg"), image)  
    return hasFrames

sec = 0
frameRate = 0.5 
count = 1
success = getFrame(sec)

while success:
    count += 1
    sec += frameRate
    sec = round(sec, 2)
    success = getFrame(sec)
