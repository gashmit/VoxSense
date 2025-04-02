import cv2
from deepface import DeepFace
import csv
import os

# Function to analyze emotions in each frame
def analyze_frame(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    return result[0]['dominant_emotion']  # Get the dominant emotion

# Path to the video and audio
video_path = "test_video.mp4"  # Replace with your video path
audio_segments = [
    (0, 10),  # Start time and end time of the first audio segment (in seconds)
    (10, 20), # Second segment
    # Add more audio segments...
]

# Create a folder to store results
output_folder = "video_results"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Open the video
cap = cv2.VideoCapture(video_path)

# Check if the video is opened
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video

# Open a CSV file to save results
csv_path = os.path.join(output_folder, "emotion_timestamps.csv")
with open(csv_path, mode="w", newline="") as csv_file:
    fieldnames = ["Timestamp (s)", "Dominant Emotion"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Process each audio segment
    for start_time, end_time in audio_segments:
        # Calculate the corresponding video frame for the start and end time
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Read frames within the time range
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Read a frame in the segment range
        ret, frame = cap.read()
        if not ret:
            continue

        # Resize the frame for emotion detection
        frame_resized = cv2.resize(frame, (640, 480))

        # Analyze the frame for emotions
        emotion = analyze_frame(frame_resized)

        # Calculate the timestamp for the middle of the segment (can be adjusted)
        timestamp = (start_time + end_time) / 2

        # Write the result to the CSV file
        writer.writerow({"Timestamp (s)": timestamp, "Dominant Emotion": emotion})

        # Save the frame as an image with the timestamp in the file name
        frame_image_path = os.path.join(output_folder, f"frame_{start_time}-{end_time}.jpg")
        cv2.imwrite(frame_image_path, frame)

        print(f"Processed segment {start_time}-{end_time}s with emotion: {emotion}")

# Release the video capture object
cap.release()

print(f"Processing complete. Results saved in '{output_folder}' folder.")