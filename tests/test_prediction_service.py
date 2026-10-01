from types import SimpleNamespace

import numpy as np

from src.services.prediction_service import predict_audio


class FixedModel:
    def __init__(self, prediction, probabilities):
        self.prediction = prediction
        self.probabilities = probabilities

    def predict(self, _features):
        return [self.prediction]

    def predict_proba(self, _features):
        return [np.array(self.probabilities)]


def test_predict_audio_returns_adult_prediction():
    models = SimpleNamespace(
        voice_classifier=FixedModel(0, [1.0, 0.0]),
        adult_model=FixedModel("happy", [0.2, 0.8]),
        baby_model=FixedModel(0, [1.0]),
    )

    result = predict_audio(np.zeros(180), models)

    assert result["audio_type"] == "Human Speech"
    assert result["prediction"] == "Happy"
    assert result["confidence"] == 0.8


def test_predict_audio_returns_baby_prediction():
    models = SimpleNamespace(
        voice_classifier=FixedModel(1, [0.0, 1.0]),
        adult_model=FixedModel("happy", [1.0]),
        baby_model=FixedModel(0, [0.9, 0.1]),
    )

    result = predict_audio(np.zeros(180), models)

    assert result["audio_type"] == "Baby Cry"
    assert result["prediction"] == "Hungry"
    assert result["confidence"] == 0.9