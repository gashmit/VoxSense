import os
import librosa
import numpy as np
from tqdm import tqdm
import logging
from sklearn.utils import shuffle

# Unified emotion mapping
emotion_map = {
    'neutral': 0,
    'happy': 1,
    'sad': 2,
    'angry': 3,
    'fear': 4,
    'disgust': 5,
    'surprise': 6
}

# Paths to datasets
DATASETS = {
    "ravdess": "/Users/ashmitgupta/voxsense.1/RAVDESS",
    "tess": "/Users/ashmitgupta/voxsense.1/Tess/TESS Toronto emotional speech set data",
    "crema_d": "//Users/ashmitgupta/voxsense.1/Crema-D",
    "savee": "//Users/ashmitgupta/voxsense.1/Savee",
    "voxceleb": "/Users/ashmitgupta/voxsense.1/Voxceleb-1 Dataset",
    "voxceleb-master": "/Users/ashmitgupta/voxsense.1/voxceleb-master"
}

# Configure logging for error tracking
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

def standardize_emotion(emotion):
    """Standardizes emotion labels across datasets."""
    standardized_emotion_map = {
        'neutral': 'neutral',
        'calm': 'neutral',  # Mapping 'calm' to 'neutral' for consistency
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'fear': 'fear',
        'disgust': 'disgust',
        'surprise': 'surprise'
    }
    return standardized_emotion_map.get(emotion, 'neutral')

def preprocess_audio(audio, sr=16000):
    """Applies basic preprocessing to the audio data (e.g., normalization)."""
    # Normalize audio
    audio = librosa.util.normalize(audio)
    return audio

def extract_emotion_ravdess(filename):
    emotion_code = int(filename.split('-')[2])
    emotion_dict = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fear', 7: 'disgust', 8: 'surprise'
    }
    return emotion_dict.get(emotion_code)

def extract_emotion_tess(folder_name):
    return folder_name.lower().split('_')[-1]

def extract_emotion_crema(filename):
    emotion_code = filename.split('_')[2]
    crema_map = {
        'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fear', 'HAP': 'happy',
        'NEU': 'neutral', 'SAD': 'sad'
    }
    return crema_map.get(emotion_code)

def extract_emotion_savee(filename):
    prefix = filename[:2]
    savee_map = {
        'an': 'angry', 'di': 'disgust', 'fe': 'fear', 'ha': 'happy',
        'ne': 'neutral', 'sa': 'sad', 'su': 'surprise'
    }
    return savee_map.get(prefix)

def extract_features(file_path, sr=16000):
    try:
        audio, sr = librosa.load(file_path, sr=sr)
        audio = preprocess_audio(audio, sr)  # Preprocess audio before feature extraction
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        return mfcc_scaled
    except Exception as e:
        logging.error(f"Error with {file_path}: {e}")
        return None

def load_all_data():
    X, y, dataset_flags = [], [], []

    for dataset, path in DATASETS.items():
        print(f"Processing {dataset}...")
        for root, _, files in os.walk(path):
            for file in tqdm(files):
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)

                    # Get emotion label
                    if dataset == "ravdess":
                        emotion = extract_emotion_ravdess(file)
                    elif dataset == "tess":
                        emotion = extract_emotion_tess(os.path.basename(root))
                    elif dataset == "crema_d":
                        emotion = extract_emotion_crema(file)
                    elif dataset == "savee":
                        emotion = extract_emotion_savee(file)
                    else:
                        continue

                    emotion = standardize_emotion(emotion)  # Standardize the emotion label

                    if emotion not in emotion_map:
                        continue

                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(emotion_map[emotion])
                        dataset_flags.append(dataset)  # Add source dataset name

    # Shuffle the dataset before returning it
    X, y, dataset_flags = shuffle(np.array(X), np.array(y), np.array(dataset_flags), random_state=42)

    return X, y, dataset_flags

if __name__ == "__main__":
    X, y, flags = load_all_data()
    print(f"Final dataset: {X.shape} features, {y.shape} labels, {flags.shape} flags")

np.save("merged_X.npy", X)
np.save("merged_y.npy", y)
np.save("merged_dataset_flags.npy", flags)

print("merged_X.npy, merged_y.npy, and merger_dataset_flags.npy ✅")
