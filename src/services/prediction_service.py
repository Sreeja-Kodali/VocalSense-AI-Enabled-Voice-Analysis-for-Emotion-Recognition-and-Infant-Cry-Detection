import numpy as np

from src.utils.errors import PredictionError

BABY_EMOTIONS = {
    0: "hungry",
    1: "tired",
    2: "pain",
    3: "discomfort",
    4: "burping",
}


def predict_audio(features, models):
    try:
        voice_type = models.voice_classifier.predict([features])[0]

        if voice_type == 0:
            probabilities = models.adult_model.predict_proba([features])[0]
            emotion = models.adult_model.predict([features])[0]
            return {
                "type": "adult",
                "audio_type": "Human Speech",
                "emotion": emotion,
                "prediction": str(emotion).capitalize(),
                "confidence_scores": probabilities.tolist(),
                "confidence": float(np.max(probabilities)),
            }

        probabilities = models.baby_model.predict_proba([features])[0]
        emotion_code = models.baby_model.predict([features])[0]
        emotion = BABY_EMOTIONS.get(emotion_code, "unknown")
        return {
            "type": "baby",
            "audio_type": "Baby Cry",
            "emotion": emotion,
            "prediction": str(emotion).capitalize(),
            "confidence_scores": probabilities.tolist(),
            "confidence": float(np.max(probabilities)),
        }
    except Exception as error:
        raise PredictionError("Prediction could not be completed. Please try another WAV file.") from error