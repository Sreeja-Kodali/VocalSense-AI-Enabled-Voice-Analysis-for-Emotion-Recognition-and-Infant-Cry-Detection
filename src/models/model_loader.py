import pickle
from dataclasses import dataclass

from src.utils.errors import ModelLoadingError
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PredictionModels:
    voice_classifier: object
    adult_model: object
    baby_model: object


def load_models(model_paths):
    loaded_models = {}
    try:
        for model_name, model_path in model_paths.items():
            logger.info("Loading %s", model_name)
            with open(model_path, "rb") as model_file:
                loaded_models[model_name] = pickle.load(model_file)
    except Exception as error:
        logger.exception("Could not load trained prediction models")
        raise ModelLoadingError("Unable to load the trained prediction models.") from error

    return PredictionModels(**loaded_models)