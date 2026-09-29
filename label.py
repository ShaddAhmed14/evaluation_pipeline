import cv2
import os
import random
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from .env file

def main():
    N = 1 # # of samples per label / corpus
    clips_input_dir = Path(os.environ("INPUT_FOLDER_CLIPS"))
    video_input_dir = Path(os.environ("VIDEO_FOLDER_VIDEOS"))
    clips_prediction_dir = Path(os.environ("PREDICTION_FOLDER_CLIP_BASED"))
    video_prediction_dir = Path(os.environ("PREDICTION_FOLDER_VIDEOS_BASED"))
    video_ground_truth_dir = Path(os.environ("SAVE_GROUND_TRUTH_FOLDER"))
    output_dir = Path(os.environ("LABELLED_VIDEOS_FOLDER"))
    clips_output_dir = output_dir / "clips"
    video_output_dir = output_dir / "videos"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    clips_output_dir.mkdir(parents=True, exist_ok=True)
    video_output_dir.mkdir(parents=True, exist_ok=True)

    label_clips(clips_input_dir, clips_prediction_dir, clips_output_dir, N)
    label_videos(video_input_dir, video_prediction_dir, video_output_dir, N, video_ground_truth_dir)

def label_clips(clips_input_dir: Path, clips_prediction_dir: Path, output_dir: Path, N: int):
    selected_files = sample_clip_paths(clips_input_dir, N)
    predictions = return_prediction_mapping(clips_prediction_dir, selected_files)
    add_predictions_to_videos(predictions, output_dir)

def label_videos(video_input_dir: Path, video_prediction_dir: Path, output_dir: Path, N: int, video_ground_truth_dir: Path):
    selected_files = sample_video_paths(video_input_dir, N)
    predictions = return_prediction_mapping(video_prediction_dir, selected_files, remove_suffix=False)
    add_predictions_to_videos(predictions, output_dir, video_ground_truth_dir)

def sample_video_paths(video_input_dir: Path, N: int) -> dict:
    corpus_file_mapping = {}
    for video in video_input_dir.glob('*.mp4'):
        corpus_name = video.stem.split('_')[0]
        corpus_file_mapping.setdefault(corpus_name, []).append(video)

    selected_files = []
    for corpus, list_of_videos in corpus_file_mapping.items():
        selected_files.extend(random.sample(list_of_videos, min(N, len(list_of_videos))))

    return selected_files

def sample_clip_paths(input_dir: Path, N: int) -> list[Path]:
    # find random input videos for each label
    selected_files = []
    for subdirectory in input_dir.iterdir():
        if subdirectory.is_dir():
            file_paths = [path for path in subdirectory.glob('*.mp4')]
            selected_file_paths = random.sample(file_paths, min(N, len(file_paths)))
            selected_files.extend(selected_file_paths)
    return selected_files

def return_prediction_mapping(prediction_dir: Path, selected_files: list[Path], remove_suffix: bool=True) -> dict:
    # find matching prediction files for selected input videos
    predictions = {}
    for model_dir in prediction_dir.iterdir():
        if model_dir.is_dir():
            predictions[model_dir.name] = []
            for file in selected_files:
                if remove_suffix:
                    file_name = file.stem.rsplit('_', 1)[0]  # Remove the last part (e.g., "MediaPipePoseWorldLandmarker")
                else:
                    file_name = file.stem
                prediction_file_name = f"{file_name}_predictions.csv"
                prediction_file_path = model_dir / prediction_file_name
                if prediction_file_path.exists():
                    predictions[model_dir.name].append([ file, prediction_file_path])
                else:
                    print(f"Prediction file not found: {prediction_file_path}")
    return predictions

def add_predictions_to_videos(predictions: dict, output_dir: Path, video_ground_truth_dir: Path = None):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    margin = 10
    # add (ground truth and) prediction text to videos and save to output folder
    for model_name, file_pairs in predictions.items():
        model_output_dir = output_dir / model_name
        model_output_dir.mkdir(parents=True, exist_ok=True)

        for input_file, prediction_file in file_pairs:
            df = pd.read_csv(prediction_file)
            gt_df = None # default
            video_name = input_file.stem
            # load ground truth if exists
            if video_ground_truth_dir:
                ground_truth_file_path = video_ground_truth_dir / f"{video_name}_groundtruth.csv"
                if ground_truth_file_path.exists():
                    gt_df = pd.read_csv(ground_truth_file_path)
                else:
                    print(f"Ground truth file not found: {ground_truth_file_path}")
            
            with cv2.VideoCapture(str(input_file)) as cap:
                print(f"Processing video: {video_name} with model: {model_name}")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                output_file_path = model_output_dir / f"{video_name}_labelled.mp4"
                out = cv2.VideoWriter(str(output_file_path), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

                frame_index = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Get prediction for the current frame
                    if frame_index < len(df):
                        prediction = df.iloc[frame_index]['prediction']
                        text = f"{model_name}\n{prediction}"
                    else:
                        text = "Missing"

                    (text_width, text_height), _ = cv2.getTextSize(
                        text, font, font_scale, thickness
                    )
                    x = frame.shape[1] - text_width - margin
                    y = margin + text_height + (text_height + 5)

                    cv2.putText(
                        frame,
                        text,
                        (x, y),
                        font,
                        font_scale,
                        (255, 255, 255),
                        thickness,
                    )

                    if gt_df is not None and frame_index < len(gt_df):
                        ground_truth = gt_df.iloc[frame_index]['ground_truth']
                        ground_truth_text = f"Ground Truth\n{ground_truth}"
                        cv2.putText(
                            frame,
                            ground_truth_text,
                            (5, text_height + 5), # top left corner
                            font,
                            font_scale,
                            (255, 255, 255),
                            thickness,
                        )

                    # Write the frame to the output video
                    out.write(frame)
                    frame_index += 1

                out.release()

if __name__ == "__main__":
    main()