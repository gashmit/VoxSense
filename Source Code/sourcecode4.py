import os
import whisper
import numpy as np
import librosa
import tensorflow as tf
from nrclex import NRCLex
import sounddevice as sd
import soundfile as sf
from fpdf import FPDF
import matplotlib.pyplot as plt
import datetime

# Load models
print("Loading models...")
whisper_model = whisper.load_model("base")
audio_emotion_model = tf.keras.models.load_model("ravdess_emotion_model.h5")
print("Model Output Shape:", audio_emotion_model.output_shape)

# Emotion color mapping for PDF highlights
emotion_colors = {
    "happy": (255, 255, 0),       # Yellow
    "angry": (255, 0, 0),         # Red
    "sad": (135, 206, 250),       # Light Blue
    "fearful": (173, 216, 230),   # Sky Blue
    "neutral": (211, 211, 211),   # Light Gray
    "surprise": (255, 182, 193),  # Light Pink
    "disgust": (144, 238, 144),   # Light Green
    "contempt": (255, 222, 173),  # Navajo White
    "mixed": (255, 255, 255),     # White (no highlight)
    "unknown": (255, 255, 255)
}

# Record audio
def record_audio(duration=5, sample_rate=16000):
    print("Recording audio...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32")
    sd.wait()
    print("Recording complete.")
    return audio.flatten(), sample_rate

# Transcribe using Whisper
def transcribe_audio(audio_path):
    print("Transcribing audio...")
    result = whisper_model.transcribe(audio_path)
    return result["text"]

# Detect text emotion
def detect_text_emotion(text):
    emotion = NRCLex(text)
    emotion_scores = emotion.affect_frequencies
    if emotion_scores:
        return max(emotion_scores, key=emotion_scores.get)
    return "neutral"

# Detect audio emotion
def detect_audio_emotion_segment(audio, sample_rate):
    print("Detecting emotion from audio segment...")
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    mfccs = np.expand_dims(mfccs, axis=0)
    prediction = audio_emotion_model.predict(mfccs)
    print("Prediction Shape:", prediction.shape)

    emotion_labels = ["angry", "happy", "neutral", "sad", "fear", "surprise", "disgust", "contempt", "fearful"]

    if prediction.shape[1] == len(emotion_labels):
        emotion = emotion_labels[np.argmax(prediction)]
    else:
        print("Warning: Prediction shape does not match the number of emotion labels.")
        emotion = "unknown"
    return emotion

# Combine both emotions
def combine_emotions(text_emotion, audio_emotion):
    # Normalize text
    text_emotion = text_emotion.lower()
    audio_emotion = audio_emotion.lower()

    # Synonym groups
    emotion_groups = {
        "happy": ["happy", "joy", "excited", "pleased", "cheerful"],
        "angry": ["angry", "mad", "annoyed", "frustrated", "rage"],
        "sad": ["sad", "down", "depressed", "upset", "melancholy"],
        "fearful": ["fear", "afraid", "fearful", "nervous", "anxious"],
        "surprise": ["surprise", "shocked", "amazed"],
        "disgust": ["disgust", "grossed out", "repulsed"],
        "contempt": ["contempt", "disrespect", "scorn"],
        "neutral": ["neutral", "calm", "unemotional"],
    }

    def map_emotion(e):
        for key, synonyms in emotion_groups.items():
            if e in synonyms:
                return key
        return "unknown"

    mapped_text = map_emotion(text_emotion)
    mapped_audio = map_emotion(audio_emotion)

    if mapped_text == mapped_audio:
        return mapped_text
    elif mapped_text == "neutral":
        return mapped_audio
    elif mapped_audio == "neutral":
        return mapped_text
    else:
        return "mixed"
#print(f"Text Emotion: {text_emotion}, Audio Emotion: {audio_emotion}")
# Split text into sentences
def split_sentences(text):
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]

# Create a PDF with highlighted sentences
def create_pdf(transcript, emotion_results, output_folder):
    pdf_filename = os.path.join(output_folder, "transcript.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=16, style="B")
    pdf.cell(200, 10, txt="Emotion Transcript with Audio Analysis", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)

    for sentence, emotion in zip(transcript, emotion_results):
        r, g, b = emotion_colors.get(emotion, (255, 255, 255))
        pdf.set_fill_color(r, g, b)
        text = f"{sentence} [{emotion}]"
        pdf.multi_cell(0, 10, text, fill=True)

    pdf.output(pdf_filename)
    print(f"PDF generated successfully: {pdf_filename}")
    return pdf_filename

# Real-time plotting
def real_time_plotting(emotion_labels):
    plt.ion()
    fig, ax = plt.subplots()
    emotions_count = {emotion: 0 for emotion in emotion_labels}

    def update_plot(emotion):
        if emotion not in emotions_count:
            emotions_count[emotion] = 0
        emotions_count[emotion] += 1
        ax.clear()
        ax.bar(emotions_count.keys(), emotions_count.values(), color="skyblue")
        ax.set_title("Real-Time Emotion Detection")
        ax.set_ylabel("Emotion Count")
        ax.set_xlabel("Emotion")
        plt.draw()
        plt.pause(0.1)

    def save_plot_image(output_folder):
        plot_filename = os.path.join(output_folder, "plot.png")
        fig.savefig(plot_filename)
        print(f"Emotion plot saved as {plot_filename}")

    return update_plot, save_plot_image

# Main function
def main():
    # Create the output directory if it doesn't exist
    session_folder_base = "output"
    if not os.path.exists(session_folder_base):
        os.makedirs(session_folder_base)

    # Generate a timestamp for the session folder name (DDMMYY_HHMM)
    timestamp = datetime.datetime.now().strftime("%d%m%y_%H%M")
    session_folder = os.path.join(session_folder_base, f"session_{timestamp}")
    
    os.makedirs(session_folder)
    print(f"Creating session folder: {session_folder}")

    input_method = input("Do you want to use a pre-recorded file or record in real-time? (file/realtime): ").strip().lower()

    if input_method == "file":
        audio_path = input("Enter the path to the audio file: ").strip()
        audio, sample_rate = librosa.load(audio_path, sr=16000)
    elif input_method == "realtime":
        duration = int(input("Enter recording duration (in seconds): ").strip())
        audio, sample_rate = record_audio(duration=duration)
        audio_path = "temp_recording.wav"
        sf.write(audio_path, audio, sample_rate)
    else:
        print("Invalid input method. Exiting.")
        return

    emotion_labels = list(emotion_colors.keys())
    update_plot, save_plot_image = real_time_plotting(emotion_labels)

    full_text = transcribe_audio(audio_path)
    print(f"\nFull Transcription:\n{full_text}\n")

    audio_emotion = detect_audio_emotion_segment(audio, sample_rate)
    print(f"Audio-based Emotion: {audio_emotion}\n")

    print("Sentence-wise Emotion Detection:")
    sentences = split_sentences(full_text)
    emotion_results = []

    for sentence in sentences:
        text_emotion = detect_text_emotion(sentence)
        print(f"Text Emotion: {text_emotion}")  # Debugging print
        combined = combine_emotions(text_emotion, audio_emotion)
        emotion_results.append(combined)
        print(f"Combined Emotion: {combined}")  # Debugging print
        update_plot(combined)

    create_pdf(sentences, emotion_results, session_folder)
    save_plot_image(session_folder)

    if input_method == "realtime":
        os.remove(audio_path)

if __name__ == "__main__":
    main()