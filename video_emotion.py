import cv2
from deepface import DeepFace
import matplotlib.pyplot as plt
import csv
import os

# Function to analyze emotions in each frame
def analyze_frame(frame):
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
    return result[0]['dominant_emotion']  # Get the dominant emotion

# Path to the video
video_path = "/Users/ashmitgupta/voxsense.1/Heart Touching Humanity Video.mp4"  # Replace with your video path

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

# Video properties (for timestamp calculation)
fps = cap.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total number of frames
print(f"Video FPS: {fps}, Total frames: {frame_count}")

# Open a CSV file to save results
csv_path = os.path.join(output_folder, "emotion_timestamps.csv")
with open(csv_path, mode="w", newline="") as csv_file:
    fieldnames = ["Timestamp (s)", "Dominant Emotion"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Read the video frame by frame
    frame_number = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process the frame (resize for faster processing)
        frame_resized = cv2.resize(frame, (640, 480))

        # Analyze the frame for emotions
        emotion = analyze_frame(frame_resized)

        # Calculate the timestamp based on frame number and FPS
        timestamp = frame_number / fps

        # Write the timestamp and emotion to the CSV file
        writer.writerow({"Timestamp (s)": timestamp, "Dominant Emotion": emotion})

        # Save the frame as an image with timestamp in the file name
        frame_image_path = os.path.join(output_folder, f"frame_{frame_number:04d}.jpg")
        cv2.imwrite(frame_image_path, frame)

        # Convert the frame to RGB for displaying with matplotlib
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Display the frame with the detected emotion as title
        plt.imshow(frame_rgb)
        plt.title(f"Emotion: {emotion}")
        plt.axis("off")
        plt.show()

        # Increment frame number
        frame_number += 1

# Release the video capture object
cap.release()

print(f"Processing complete. Results saved in '{output_folder}' folder.")