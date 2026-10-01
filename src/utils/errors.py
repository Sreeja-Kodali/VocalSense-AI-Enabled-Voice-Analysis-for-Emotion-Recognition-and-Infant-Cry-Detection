class InvalidUploadError(ValueError):
    """Raised when an uploaded file fails validation."""


class AudioProcessingError(RuntimeError):
    """Raised when an uploaded audio file cannot be decoded or processed."""


class ModelLoadingError(RuntimeError):
    """Raised when one or more trained models cannot be loaded."""


class PredictionError(RuntimeError):
    """Raised when model inference cannot be completed."""