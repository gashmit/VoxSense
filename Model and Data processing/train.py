import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from model import build_cnn_model
from data_preprocessing2 import load_preprocessed_data, prepare_data

if __name__ == "__main__":
    # Load preprocessed merged dataset
    X, y = load_preprocessed_data("merged_X.npy", "merged_y.npy")

    # Prepare training and testing data
    X_train, X_test, y_train, y_test, num_classes = prepare_data(X, y)

    # Save data (optional)
    np.save("X.npy", X)
    np.save("y.npy", y)

    # Expand dims for Conv1D input
    X_train = np.expand_dims(X_train, axis=2)  # Shape becomes (samples, timesteps, features)
    X_test = np.expand_dims(X_test, axis=2)    # Shape becomes (samples, timesteps, features)

    # Build model
    input_shape = X_train.shape[1:]  # (timesteps, features, 1), for example (200, 40, 1)
    model = build_cnn_model(input_shape, num_classes)
    model.summary()

    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs')
    # live_plot = LivePlotting()  # Optional if defined

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=[early_stop, checkpoint, tensorboard]  # Add live_plot if available
    )

    # Save model and test set
    model.save("ravdess_emotion_model.h5")
    np.save("X_test.npy", X_test)
    np.save("y_test.npy", y_test)

    print("✅ Model and test data saved successfully.")

    # Evaluation
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification Report
    print("Classification Report:\n", classification_report(y_true, y_pred_classes))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)

    # Load label encoder if you have one (optional)
    try:
        with open("label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        class_names = label_encoder.classes_
    except:
        class_names = [str(i) for i in range(num_classes)]

    # Plot Confusion Matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()