from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Load an image
image_path = "/Users/ashmitgupta/voxsense.1/Happy Child.jpeg"  # Replace with your image path
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Analyze emotion
result = DeepFace.analyze(image_path, actions=['emotion'])

# Display the image with detected emotion
plt.imshow(image_rgb)
plt.axis("off")
plt.title(f"Emotion: {result[0]['dominant_emotion']}")
plt.show()

# Print full emotion analysis
print(result)