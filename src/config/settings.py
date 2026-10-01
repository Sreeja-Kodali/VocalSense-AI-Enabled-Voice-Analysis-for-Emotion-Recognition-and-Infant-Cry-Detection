import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings:
    MAX_CONTENT_LENGTH = int(os.environ.get("VOCALSENSE_MAX_UPLOAD_SIZE", 16 * 1024 * 1024))
    ALLOWED_EXTENSIONS = frozenset({".wav"})
    UPLOAD_FOLDER = os.environ.get(
        "VOCALSENSE_UPLOAD_FOLDER",
        str(Path(tempfile.gettempdir()) / "vocalsense-uploads"),
    )
    MODEL_PATHS = {
        "voice_classifier": Path(os.environ.get(
            "VOCALSENSE_VOICE_MODEL", PROJECT_ROOT / "models" / "Pickle_Voice_Classifier.pkl"
        )),
        "adult_model": Path(os.environ.get(
            "VOCALSENSE_ADULT_MODEL", PROJECT_ROOT / "models" / "Pickle_RL_Model.pkl"
        )),
        "baby_model": Path(os.environ.get(
            "VOCALSENSE_BABY_MODEL", PROJECT_ROOT / "models" / "Pickle_Baby_Model.pkl"
        )),
    }