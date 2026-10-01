from pathlib import Path

from flask import Flask, jsonify, render_template
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from src.config.settings import Settings
from src.models.model_loader import load_models
from src.routes.prediction_routes import prediction_routes
from src.utils.errors import AudioProcessingError, InvalidUploadError, ModelLoadingError, PredictionError
from src.utils.logger import configure_logging, get_logger

logger = get_logger(__name__)


def create_app(models=None, test_config=None):
    configure_logging()
    application = Flask(__name__)
    application.config.from_object(Settings)
    if test_config:
        application.config.update(test_config)

    if models is None:
        models = load_models(application.config["MODEL_PATHS"])
    application.extensions["vocal_sense_models"] = models
    application.register_blueprint(prediction_routes)

    @application.route("/")
    def index():
        return render_template("index.html")

    @application.errorhandler(InvalidUploadError)
    def handle_invalid_upload(error):
        logger.warning("Upload rejected: %s", error)
        return jsonify({"status": "error", "error": str(error)}), 400

    @application.errorhandler(AudioProcessingError)
    def handle_audio_error(error):
        logger.warning("Audio processing failed: %s", error)
        return jsonify({"status": "error", "error": str(error)}), 400

    @application.errorhandler(PredictionError)
    def handle_prediction_error(error):
        logger.exception("Prediction failed")
        return jsonify({"status": "error", "error": str(error)}), 500

    @application.errorhandler(ModelLoadingError)
    def handle_model_error(error):
        logger.error("Model loading failed: %s", error)
        return jsonify({"status": "error", "error": str(error)}), 503

    @application.errorhandler(RequestEntityTooLarge)
    def handle_large_upload(_error):
        return jsonify({"status": "error", "error": "Audio file exceeds the maximum allowed size."}), 413

    @application.errorhandler(Exception)
    def handle_unexpected_error(error):
        if isinstance(error, HTTPException):
            return jsonify({"status": "error", "error": error.description}), error.code
        logger.exception("Unexpected application error")
        return jsonify({"status": "error", "error": "An unexpected server error occurred."}), 500

    logger.info("VocalSense application started")
    return application


app = create_app()


if __name__ == "__main__":
    app.run(debug=False, port=5000)