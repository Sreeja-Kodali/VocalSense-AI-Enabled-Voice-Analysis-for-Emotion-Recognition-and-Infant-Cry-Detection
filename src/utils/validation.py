from pathlib import Path

from src.utils.errors import InvalidUploadError


def validate_audio_upload(audio_file, allowed_extensions, max_file_size):
    if audio_file is None:
        raise InvalidUploadError("No audio file provided.")

    filename = audio_file.filename or ""
    if not filename:
        raise InvalidUploadError("No audio file selected.")

    if Path(filename).suffix.lower() not in allowed_extensions:
        raise InvalidUploadError("Only WAV audio files are supported.")

    stream = audio_file.stream
    try:
        stream.seek(0, 2)
        file_size = stream.tell()
        stream.seek(0)
    except (AttributeError, OSError) as error:
        raise InvalidUploadError("The uploaded file could not be read.") from error

    if file_size == 0:
        raise InvalidUploadError("The uploaded file is empty.")
    if file_size > max_file_size:
        raise InvalidUploadError("Audio file exceeds the maximum allowed size.")

    return audio_file