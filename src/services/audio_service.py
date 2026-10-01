import os
import tempfile

import librosa
import numpy as np
import soundfile

from src.utils.errors import AudioProcessingError


def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))

        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))

    return result


def extract_upload_features(audio_file, upload_folder):
    temporary_path = None
    try:
        os.makedirs(upload_folder, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".wav", dir=upload_folder, delete=False) as temporary_file:
            temporary_path = temporary_file.name
        audio_file.save(temporary_path)
        features = extract_feature(temporary_path)
        if features is None or len(features) == 0:
            raise AudioProcessingError("Audio feature extraction failed.")
        return features
    except AudioProcessingError:
        raise
    except Exception as error:
        raise AudioProcessingError("The uploaded file is not a valid WAV audio file.") from error
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)