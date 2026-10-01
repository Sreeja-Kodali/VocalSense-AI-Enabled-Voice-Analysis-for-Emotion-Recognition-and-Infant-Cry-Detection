# VocalSense: AI-Enabled Voice Analysis for Emotion Recognition and Infant Cry Detection

## Project Overview

VocalSense is a Flask application that analyzes WAV audio and uses three existing trained machine-learning models to distinguish human speech from infant cries, then predict an emotion or cry category. This ASE enhancement organizes the existing inference code into small modules without retraining or changing the model algorithms.

## Problem Statement

Speech emotion and infant-cry classification involve separate trained models and audio-processing steps. Keeping upload handling, audio features, model loading, and prediction in one script makes it harder to test and maintain the application.

## Objectives

- Preserve the current audio feature extraction and trained-model prediction behavior.
- Separate routes, validation, audio processing, inference, configuration, and logging.
- Expose simple health and prediction API endpoints.
- Test request handling and inference without requiring dataset-based retraining.

## Technology Stack

- Python and Flask
- librosa, SoundFile, and NumPy for audio features
- scikit-learn for the serialized model estimators
- pytest for automated tests

The scikit-learn dependency is pinned to the model serialization version, 1.6.1.

## Architecture

The root `app.py` creates the Flask application and loads the trained models once from `models/`. Route handlers validate requests, the audio service extracts the same 180 MFCC/chroma/mel features as before, and the prediction service applies the existing voice classifier followed by the appropriate adult or baby model. Dataset files are used by the training script only and are not read during web inference.

## Modules

- `src/routes/prediction_routes.py`: health, API prediction, and compatibility prediction routes.
- `src/services/audio_service.py`: temporary upload handling and existing audio feature extraction.
- `src/services/prediction_service.py`: existing model selection and prediction sequence.
- `src/models/model_loader.py`: loads the three serialized models and reports startup failures.
- `src/utils/validation.py`: checks upload presence, extension, and size.
- `src/utils/errors.py`: application error types and friendly HTTP responses.
- `src/utils/logger.py`: application logging setup.
- `src/config/settings.py`: model paths, upload folder, extensions, and upload limit.
- `training/main.py`: existing dataset training script; it is not run by the web application. The root `main.py` remains as a compatibility launcher.
- `templates/index.html`: the existing frontend, served by Flask without changing its UI.
- `models/`: the three existing trained model files.


## Setup and Run

Use Python 3.10 or newer. From the project root:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000/`. The existing frontend continues to post to `/predict`.

Model paths can be overridden with `VOCALSENSE_VOICE_MODEL`, `VOCALSENSE_ADULT_MODEL`, and `VOCALSENSE_BABY_MODEL`. `VOCALSENSE_MAX_UPLOAD_SIZE` sets the maximum upload size in bytes, and `VOCALSENSE_UPLOAD_FOLDER` sets the temporary-upload directory. The defaults point to the trained models in `models/`, a 16 MiB limit, and a temporary OS directory.

## Testing

Run the tests from the project root:

```powershell
pytest
```

Tests use small fake estimators and in-memory WAV files. They do not load or retrain models and do not require the dataset.

## Limitations

- The application accepts WAV uploads only.
- Prediction quality depends on the existing trained models and their training data.
- Pickle model files must be loaded only from trusted sources and should use the compatible scikit-learn version.
- The app is intended as a student project and local demonstration, not a clinical or production diagnostic system.

## Future Enhancements

- Add more automated tests around real audio decoding and model loading.
- Improve model evaluation on separate, representative datasets.
- Add configuration for production hosting and secure upload controls if deployment becomes a requirement.