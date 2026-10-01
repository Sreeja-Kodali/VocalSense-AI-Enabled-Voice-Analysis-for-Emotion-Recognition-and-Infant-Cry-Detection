from io import BytesIO
from types import SimpleNamespace

import numpy as np
import soundfile

from app import create_app


class FixedModel:
    def __init__(self, prediction, probabilities):
        self.prediction = prediction
        self.probabilities = probabilities

    def predict(self, _features):
        return [self.prediction]

    def predict_proba(self, _features):
        return [np.array(self.probabilities)]


def make_models(voice_type=0):
    return SimpleNamespace(
        voice_classifier=FixedModel(voice_type, [1.0, 0.0] if voice_type == 0 else [0.0, 1.0]),
        adult_model=FixedModel("happy", [0.2, 0.8]),
        baby_model=FixedModel(0, [0.9, 0.1]),
    )


def make_wav_upload():
    audio = BytesIO()
    soundfile.write(audio, np.sin(np.linspace(0, 8, 22050)), 22050, format="WAV")
    audio.seek(0)
    return audio


def make_client(monkeypatch, voice_type=0):
    monkeypatch.setattr(
        "src.routes.prediction_routes.extract_upload_features",
        lambda _audio_file, _upload_folder: np.zeros(180),
    )
    application = create_app(models=make_models(voice_type), test_config={"TESTING": True})
    return application.test_client()


def test_health_endpoint(monkeypatch):
    client = make_client(monkeypatch)

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json["status"] == "success"


def test_valid_prediction_request_for_speech(monkeypatch):
    client = make_client(monkeypatch)

    response = client.post(
        "/api/predict",
        data={"audio": (make_wav_upload(), "sample.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json == {
        "status": "success",
        "audio_type": "Human Speech",
        "prediction": "Happy",
        "confidence": 0.8,
    }


def test_valid_prediction_request_for_baby_cry(monkeypatch):
    client = make_client(monkeypatch, voice_type=1)

    response = client.post(
        "/api/predict",
        data={"audio": (make_wav_upload(), "sample.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    assert response.json["audio_type"] == "Baby Cry"
    assert response.json["prediction"] == "Hungry"


def test_missing_file_returns_clear_error(monkeypatch):
    client = make_client(monkeypatch)

    response = client.post("/api/predict", data={}, content_type="multipart/form-data")

    assert response.status_code == 400
    assert response.json["error"] == "No audio file provided."


def test_invalid_file_type_returns_clear_error(monkeypatch):
    client = make_client(monkeypatch)

    response = client.post(
        "/api/predict",
        data={"audio": (BytesIO(b"not audio"), "sample.mp3")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "WAV" in response.json["error"]


def test_corrupted_wav_returns_clear_error():
    application = create_app(models=make_models(), test_config={"TESTING": True})
    client = application.test_client()

    response = client.post(
        "/api/predict",
        data={"audio": (BytesIO(b"not a wav file"), "broken.wav")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert "valid WAV" in response.json["error"]


def test_root_serves_existing_frontend(monkeypatch):
    client = make_client(monkeypatch)

    response = client.get("/")

    assert response.status_code == 200
    assert b"Emotion-Aware Voice Analysis" in response.data