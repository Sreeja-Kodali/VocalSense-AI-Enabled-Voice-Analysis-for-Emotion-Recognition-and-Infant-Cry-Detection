from flask import Blueprint, current_app, jsonify, request

from src.services.audio_service import extract_upload_features
from src.services.prediction_service import predict_audio
from src.utils.logger import get_logger
from src.utils.validation import validate_audio_upload

prediction_routes = Blueprint("prediction_routes", __name__)
logger = get_logger(__name__)


def _run_prediction():
    audio_file = request.files.get("audio")
    validate_audio_upload(
        audio_file,
        current_app.config["ALLOWED_EXTENSIONS"],
        current_app.config["MAX_CONTENT_LENGTH"],
    )
    features = extract_upload_features(audio_file, current_app.config["UPLOAD_FOLDER"])
    return predict_audio(features, current_app.extensions["vocal_sense_models"])


@prediction_routes.get("/api/health")
def health():
    return jsonify({"status": "success", "message": "VocalSense API is healthy"})


@prediction_routes.post("/api/predict")
def api_predict():
    logger.info("Audio prediction request received")
    result = _run_prediction()
    logger.info("Prediction successful: %s / %s", result["audio_type"], result["prediction"])
    return jsonify({
        "status": "success",
        "audio_type": result["audio_type"],
        "prediction": result["prediction"],
        "confidence": result["confidence"],
    })


@prediction_routes.post("/predict")
def legacy_predict():
    logger.info("Legacy audio prediction request received")
    result = _run_prediction()
    logger.info("Prediction successful: %s / %s", result["audio_type"], result["prediction"])
    return jsonify({
        "type": result["type"],
        "emotion": result["emotion"],
        "confidence_scores": result["confidence_scores"],
    })