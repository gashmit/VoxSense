import os
import sounddevice as sd
import scipy.io.wavfile as wav
from datetime import datetime
from pathlib import Path

# Define the target sample rate
SAMPLE_RATE = 16000

# Define sentence prompts (~8-9 seconds of speech)
sentences = {
    "happy": [
        "I can't believe it, I finally got the job I've always dreamed of!",
        "Everything is going so well today, it's such a beautiful feeling."
    ],
    "sad": [
        "I tried so hard, but in the end, nothing worked out the way I hoped.",
        "Sometimes it feels like I'm all alone, even in a room full of people."
    ],
    "angry": [
        "You never listen to me, and I am so tired of repeating myself!",
        "This is absolutely ridiculous, I won’t tolerate this anymore!"
    ],
    "fearful": [
        "I heard something outside, and I think someone might be following me.",
        "I don’t know what’s going to happen next, and that really scares me."
    ],
    "surprise": [
        "Wait, are you serious? I never expected this at all!",
        "Oh wow, I didn’t see that coming — what a shock!"
    ],
    "disgust": [
        "That smell is unbearable, how can anyone stand it?",
        "Ugh, that was the most disgusting thing I’ve ever seen."
    ],
    "neutral": [
        "The meeting will start at 3 PM in conference room B.",
        "I went to the store to buy groceries and picked up some bread."
    ]
}

def record_audio(filename):
    print("\n🎤 Press ENTER to start recording.")
    input()
    print("🔴 Recording... Press ENTER to stop.")
    recording = []
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = 1
    stream = sd.InputStream()
    stream.start()
    input()
    stream.stop()
    audio_data, _ = stream.read(stream.read_available)
    audio = audio_data[:, 0]
    wav.write(filename, SAMPLE_RATE, audio)
    print(f"✅ Saved: {filename}")


def main():
    name = input("Enter your name: ").strip()
    gender = input("Enter your gender (male/female): ").strip().lower()
    while gender not in ["male", "female"]:
        gender = input("Please enter 'male' or 'female': ").strip().lower()

    base_path = Path("recordings") / gender / name
    base_path.mkdir(parents=True, exist_ok=True)

    for emotion, prompts in sentences.items():
        for i, sentence in enumerate(prompts):
            print(f"\n--- Emotion: {emotion.upper()} | Sentence {i + 1} ---")
            print(f"👉 Say this: \"{sentence}\"")
            emotion_path = base_path / emotion
            emotion_path.mkdir(parents=True, exist_ok=True)
            filename = emotion_path / f"{emotion}_{i+1}.wav"
            record_audio(filename)

    print("\n🎉 All recordings completed!")
    print(f"Please upload this folder to the shared Google Drive folder:")
    print("📁", base_path.resolve())
    print("🔗 Google Drive link: https://drive.google.com/drive/folders/1idMYuLheOvvRwv5oTeimluXyT79XbW1f?usp=drive_link")

if __name__ == "__main__":
    main()
