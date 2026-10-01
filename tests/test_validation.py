from io import BytesIO

import pytest
from werkzeug.datastructures import FileStorage

from src.utils.errors import InvalidUploadError
from src.utils.validation import validate_audio_upload


def test_validate_audio_upload_rejects_missing_file():
    with pytest.raises(InvalidUploadError, match="No audio file provided"):
        validate_audio_upload(None, {".wav"}, 1024)


def test_validate_audio_upload_rejects_empty_file():
    upload = FileStorage(stream=BytesIO(b""), filename="sample.wav")

    with pytest.raises(InvalidUploadError, match="empty"):
        validate_audio_upload(upload, {".wav"}, 1024)


def test_validate_audio_upload_rejects_unsupported_extension():
    upload = FileStorage(stream=BytesIO(b"audio"), filename="sample.mp3")

    with pytest.raises(InvalidUploadError, match="WAV"):
        validate_audio_upload(upload, {".wav"}, 1024)


def test_validate_audio_upload_rejects_oversized_file():
    upload = FileStorage(stream=BytesIO(b"audio"), filename="sample.wav")

    with pytest.raises(InvalidUploadError, match="maximum"):
        validate_audio_upload(upload, {".wav"}, 4)