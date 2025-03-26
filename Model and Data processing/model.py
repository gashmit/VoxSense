from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import Callback

# Live Plotting Class
class LivePlotting(Callback):
    def on_epoch_end(self, epoch, logs=None):
        plt.clf()
        plt.subplot(2, 1, 1)
        plt.plot(self.history.history['accuracy'], label='Training Accuracy')
        plt.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        plt.legend(loc="upper left")
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.legend(loc="upper left")
        plt.title('Training and Validation Loss')

        plt.pause(0.1)

# CNN Model
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(128, kernel_size=3, activation="relu"),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model