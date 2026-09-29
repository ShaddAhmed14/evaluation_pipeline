# Evaluation Pipeline

Evaluate gesture detection models on labeled clips and full videos. The pipeline runs each enabled model in Docker, reads its frame predictions, and writes CSV metrics.

## Requirements

- Python 3.14 or newer and [uv](https://docs.astral.sh/uv/)
- Docker with a running daemon and access to the configured model image (`zephasus/envisionhgdetector_docker:latest`)

## Setup

From this directory, install the locked dependencies and create a local environment file:

```sh
uv sync
cp .env.template .env
```

On PowerShell, use `Copy-Item .env.template .env` instead of `cp` if preferred. Set the paths in `.env` before running the pipeline:

| Variable | Purpose |
| --- | --- |
| `INPUT_FOLDER_CLIPS` | Clip data, grouped into subfolders named for their true labels; each contains `.npz` files. |
| `INPUT_FOLDER_VIDEOS` | Full video `.npz` files containing an `fps` value. |
| `CLIPS_INFO_FOLDER` | JSON ground truth files named `<video>_clips_info.json`, with clip `start`, `end`, and `label` fields. |
| `PREDICTION_FOLDER_CLIP_BASED` | Model prediction CSV output for clips. |
| `PREDICTION_FOLDER_VIDEOS_BASED` | Model prediction CSV output for full videos. |
| `SAVE_GROUND_TRUTH_FOLDER` | Generated frame ground truth CSV files. |
| `EVALUATION_FOLDER` | Evaluation metrics and video FPS cache. |

Use paths that Docker can mount. For the optional video annotation script, also set `VIDEO_FOLDER_VIDEOS` (source `.mp4` files) and `LABELLED_VIDEOS_FOLDER` (output folder). Its clip source uses `INPUT_FOLDER_CLIPS` and expects matching `.mp4` files there.

## Run

```sh
uv run src/main.py
```

The command runs clip evaluation first, then full video evaluation. It creates output folders as needed and writes logs to `Logs/`.

## Features and output

- **Clip evaluation:** Uses the most common frame prediction per clip. Reports per-label accuracy, precision, recall, and F1, plus balanced accuracy and macro and weighted F1. Writes one row per model to `<EVALUATION_FOLDER>/clip_based/evaluation_metrics.csv` and one row per model and corpus to `evaluation_metrics_by_corpus.csv` in the same folder.
- **Video evaluation:** Converts clip timestamps to frame labels and compares them with predictions at matching frame indices for frame precision, recall, and F1. It also compares predicted segments with clip intervals for overlap, hit/miss rates, false positive rate, fragmentation, and boundary errors. Writes per-model frame and segment metrics to `frame_based_evaluation_results.csv` and `segment_based_evaluation_results.csv`, plus per-model and per-corpus metrics to `frame_based_evaluation_results_by_corpus.csv` and `segment_based_evaluation_results_by_corpus.csv` in `<EVALUATION_FOLDER>/video_based/`. Each video contributes equally to these averages.
- **Ground truth export:** Writes `<video>_groundtruth.csv` files to `SAVE_GROUND_TRUTH_FOLDER` and caches FPS values in `<EVALUATION_FOLDER>/video_based/fps_info.json`.
- **Model selection:** Edit `config.yaml` to enable or disable models and set `default_label` and `segment_label`. The included models are EnvisionHGDetector LightGBM and CNN.

Model predictions are stored under each prediction folder in a subfolder named after the model, with files named `<input-stem>_predictions.csv`. Prediction CSVs need `prediction` values; video predictions also need `frame_index` values.

## Optional annotated videos

`label.py` is intended to sample source `.mp4` files and overlay predictions (and full video ground truth, when available):

```sh
uv run src/label.py
```

