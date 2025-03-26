import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import os
from datetime import datetime

# Load label encoder
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Load model
model = load_model("ravdess_emotion_model.h5")

# Load test data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

# Expand dims if needed
if len(X_test.shape) == 3 and model.input_shape[-1] == 1:
    X_test = np.expand_dims(X_test, axis=2)

# Predict
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Define all possible labels
labels = list(range(len(label_encoder.classes_)))

# Define target names
target_names = label_encoder.classes_.astype(str)

print("\nClass Index to Emotion Mapping:")
for idx, name in enumerate(target_names):
    print(f"{idx}: {name}")

# Create output folder if it doesn't exist
output_folder = "evaluation_reports"
os.makedirs(output_folder, exist_ok=True)

# Get current date and time
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Define full path for Excel file
excel_filename = f"classification_report_{timestamp}.xlsx"
excel_path = os.path.join(output_folder, excel_filename)

# Classification Report
report = classification_report(y_true_classes, y_pred_classes, output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_excel(excel_path)
print(report)

# Tips for Improving Performance on Class 2, 5, 6
print("\n🔧 Suggestions to Improve Class 2, 5, 6 Performance:")
print("- Consider augmenting training data for these classes (e.g., noise, pitch shift, time-stretch).")
print("- Use class weights in model training to balance underrepresented classes.")
print("- Check for label noise or overlap — some emotions may be similar.")
print("- Try tuning your model architecture or adding regularization.")
print("- Consider more expressive features beyond MFCCs, like chroma or spectral contrast.")

# Confusion Matrix
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# Save classification report to Excel
df_report.to_excel(excel_path)

print(f"✅ Evaluation metrics saved to Excel: {excel_path}")
