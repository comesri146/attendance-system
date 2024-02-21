import cv2
import os

# Path to the folder containing frames
frames_folder = 'processed_output'

# Output video file name
output_video_file = 'output_video.mp4'

# Get the list of frames in the folder
frame_files = [f for f in os.listdir(frames_folder) if f.endswith('.jpg')]

# Extract the frame numbers from file names
frame_numbers = [int(f.split('_')[-1].split('.')[0]) for f in frame_files]

# Sort the frame files based on their numerical order
sorted_frame_files = [f for _, f in sorted(zip(frame_numbers, frame_files))]

# Get the first frame to extract its dimensions
first_frame = cv2.imread(os.path.join(frames_folder, sorted_frame_files[0]))
height, width, _ = first_frame.shape

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_file, fourcc, 25.0, (width, height))

# Write each frame to the video
for frame_file in sorted_frame_files:
    frame_path = os.path.join(frames_folder, frame_file)
    frame = cv2.imread(frame_path)
    out.write(frame)

# Release the VideoWriter object
out.release()

print(f"Video saved as {output_video_file}")
