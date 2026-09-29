import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple

class VideoBasedEvaluator:
    def __init__(self, video_files_mapping: dict, models: list, prediction_folder: Path, evaluation_folder: Path, clips_info_folder: Path, save_ground_truth_folder: Path, segment_labels: list):
        self.prediction_folder = prediction_folder
        self.evaluation_folder = evaluation_folder
        self.clips_info_folder = clips_info_folder
        self.save_ground_truth_folder = save_ground_truth_folder
        self.segment_labels = segment_labels
        self.video_files_mapping = video_files_mapping
        self.models = models
        self.iou_threshold = 0.5

        self.fps_json_path = self.evaluation_folder / "fps_info.json"
        self.fps_info = self.save_metadata_videos()


    def find_segments_predicted(self, model_name: str) -> dict:
        segments_predicted = {}
        for video_name in self.video_files_mapping.keys():
            prediction_file_path = self.prediction_folder / str(model_name) / f"{video_name}_predictions.csv"

            if not prediction_file_path.exists():
                logging.warning(f"Output file {prediction_file_path} not found for model {model_name}. Skipping.")
                continue

            try:
                df = pd.read_csv(prediction_file_path)
            except (OSError, pd.errors.ParserError) as exc:
                logging.warning("Cannot read predictions for %s/%s: %s", model_name, video_name, exc)
                continue
            if not {'frame_index', 'prediction'}.issubset(df.columns):
                logging.warning("Missing prediction columns for %s/%s. Skipping.", model_name, video_name)
                continue
            frame_indices = pd.to_numeric(df['frame_index'], errors='coerce')
            if frame_indices.isna().any() or (frame_indices < 0).any() or (frame_indices % 1 != 0).any():
                logging.warning("Invalid frame indices for %s/%s. Skipping.", model_name, video_name)
                continue
            df['frame_index'] = frame_indices.astype('int64')
            fps = self.fps_info.get(video_name)
            if fps is None or not np.isfinite(fps) or fps <= 0:
                logging.warning(f"FPS not found for video {video_name}. Skipping.")
                continue

            segments_predicted[video_name] = []
            current_segment_start = None

            # NOTE: works for 2 labels only, if more than 2 labels, need to modify this logic
            for index, row in df.iterrows():
                frame_number = row['frame_index']
                prediction_label = row['prediction']

                if prediction_label in self.segment_labels:
                    if current_segment_start is None: # new segment starts
                        current_segment_start = frame_number
                    # else continue the segment
                else: # segment ends
                    if current_segment_start is not None:
                        start_time = current_segment_start / fps
                        end_time = frame_number / fps
                        segments_predicted[video_name].append((start_time, end_time))
                        current_segment_start = None

            # Handle case where the last segment goes till the end of the video
            if current_segment_start is not None:
                start_time = current_segment_start / fps
                end_time = (df['frame_index'].iloc[-1] + 1) / fps  # Assuming last frame is inclusive
                segments_predicted[video_name].append((start_time, end_time))

        return segments_predicted

    def find_matching_clip_info(self, video_name: str) -> list | None: # this will serve as our ground truth for evaluation
        '''
        Note: using clips info instead of original ground truth since different datasets need to be handled differently. Clips info already stores the start and end time of the gesture clip for each video. So we can use that to get the ground truth timestamps.
        '''
        clip_info_file = self.clips_info_folder / f"{video_name}_clips_info.json"
        try:
            with open(clip_info_file, 'r') as f:
                clip_info = json.load(f)
            return [
                (clip['start'], clip['end'])
                for clip in clip_info
                if clip.get('label') in self.segment_labels
            ]
        except (OSError, ValueError, TypeError, KeyError) as exc:
            logging.warning("Cannot load clip info for %s: %s. Skipping.", video_name, exc)
            return None

    def save_metadata_videos(self) -> dict:
        if self.fps_json_path.exists() and self.fps_json_path.stat().st_size > 0: # if the file exists and is not empty, load it
            try:
                with open(self.fps_json_path, 'r') as f:
                    fps_info = json.load(f)
            except (OSError, ValueError) as exc:
                raise Exception("Cannot read FPS cache %s: %s. Rebuilding.", self.fps_json_path, exc)
        else:
            fps_info = {}

        for video_name, file in self.video_files_mapping.items():
            if video_name in fps_info:
                continue  # Skip if already exists
            try:
                with np.load(file) as data:
                    fps = float(data['fps'])
                if not np.isfinite(fps) or fps <= 0:
                    raise ValueError(f"Invalid FPS {fps}")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                logging.warning("Cannot load FPS for %s from %s: %s. Skipping.", video_name, file, exc)
                continue
            fps_info[video_name] = fps

        # Save the fps_info to a JSON file
        try:
            with open(self.fps_json_path, 'w') as f:
                json.dump(fps_info, f)
        except OSError as exc:
            logging.warning("Cannot save FPS cache %s: %s", self.fps_json_path, exc)
        return fps_info

    def save_ground_truth_frames(self, video_name: str, segments: List[Tuple[float, float]], frame_count: int) -> pd.DataFrame | None:
        """Label frame timestamps in [start, end) as Gesture."""
        fps = self.fps_info.get(video_name)
        if fps is None or not np.isfinite(fps) or fps <= 0:
            logging.warning("Missing or invalid FPS for %s. Skipping.", video_name)
            return None
        frame_indices = np.arange(frame_count)
        gesture = np.zeros(frame_count, dtype=bool)
        for start, end in segments:
            # OR this into existing value
            gesture |= (frame_indices / fps >= start) & (frame_indices / fps < end)
        ground_truth = pd.DataFrame({
            'frame_index': frame_indices,
            'ground_truth': np.where(gesture, self.segment_labels[0], 'NoGesture'),
        })
        try:
            self.save_ground_truth_folder.mkdir(parents=True, exist_ok=True)
            ground_truth.to_csv(self.save_ground_truth_folder / f'{video_name}_groundtruth.csv', index=False)
        except OSError as exc:
            logging.warning("Cannot save ground truth for %s: %s. Skipping.", video_name, exc)
            return None
        return ground_truth

    def frame_based_eval(self, ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict | None:
        if not {'frame_index', 'prediction'}.issubset(predictions.columns):
            logging.warning("Prediction CSV is missing frame_index or prediction. Skipping.")
            return None
        if predictions['frame_index'].duplicated().any():
            logging.warning("Prediction CSV contains duplicate frame indices. Skipping.")
            return None
        if not predictions['frame_index'].isin(ground_truth['frame_index']).all():
            logging.warning("Prediction CSV contains frame indices outside the video. Skipping.")
            return None
        if len(predictions) != len(ground_truth):
            logging.warning("Prediction CSV does not contain one row for every video frame. Skipping.")
            return None

        aligned = ground_truth.merge(predictions[['frame_index', 'prediction']], on='frame_index', validate='one_to_one')
        reference = aligned['ground_truth'].isin(self.segment_labels)
        comparison = aligned['prediction'].isin(self.segment_labels)
        tp = (reference & comparison).sum()
        fp = (~reference & comparison).sum()
        fn = (reference & ~comparison).sum()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {'precision': round(float(precision), 3), 'recall': round(float(recall), 3), 'f1': round(float(f1), 3)}
 
    def segment_based_eval(self, reference_segments: List[Tuple[float, float]], comparison_segments: List[Tuple[float, float]], iou_threshold: float) -> dict:
        """
        Greedy best-IoU matching between reference_segments and comparison_segments segments.
        Each reference_segments segment is matched to its highest-IoU comparison_segments
        segment (if any); each comparison_segments segment can only be used once.
        """
        if not reference_segments:
            return {
                "mean_iou": None, "hit_rate": None, "miss_rate": None,
                "false_positive_rate": (1.0 if comparison_segments else 0.0),
                "fragmentation": None,
                "mean_start_error": None, "mean_end_error": None,
            }

        # All (reference_segments_idx, comparison_segments_idx, iou) candidate pairs with iou > 0
        candidates = []
        for i, r in enumerate(reference_segments):
            for j, c in enumerate(comparison_segments):
                iou = self.compute_iou(r, c)
                if iou > 0:
                    candidates.append((iou, i, j))
        candidates.sort(reverse=True)  # highest IoU first

        matched_reference_segments, matched_comparison_segments = set(), set()
        reference_segments_to_comparison_segments = {}  # reference_segments_idx -> list of matched comparison_segments_idx (for fragmentation)
        ious = []

        for iou, i, j in candidates:
            if j in matched_comparison_segments:
                continue  # comparison_segments segment already used
            matched_comparison_segments.add(j)
            reference_segments_to_comparison_segments.setdefault(i, []).append(j)
            if i not in matched_reference_segments:
                matched_reference_segments.add(i)
                ious.append(iou)  # best iou per reference_segments segment (candidates sorted desc)

        hits = sum(1 for i in reference_segments_to_comparison_segments if self.compute_iou(reference_segments[i], comparison_segments[reference_segments_to_comparison_segments[i][0]]) >= iou_threshold)
        misses = len(reference_segments) - hits

        fp_count = len(comparison_segments) - len(matched_comparison_segments)

        mean_iou = round(sum(ious) / len(ious), 3) if ious else 0.0
        hit_rate = round(hits / len(reference_segments), 3)
        miss_rate = round(misses / len(reference_segments), 3)
        fp_rate = round(fp_count / len(comparison_segments), 3) if comparison_segments else 0.0
        fragmentation = round(
            sum(len(v) for v in reference_segments_to_comparison_segments.values()) / len(reference_segments_to_comparison_segments), 3
        ) if reference_segments_to_comparison_segments else 0.0

        # Boundary errors on the best-matched pair per reference_segmentserence segment
        start_errors, end_errors = [], []
        for i, comparison_segments_idxs in reference_segments_to_comparison_segments.items():
            best_j = comparison_segments_idxs[0]
            start_errors.append(abs(reference_segments[i][0] - comparison_segments[best_j][0]))
            end_errors.append(abs(reference_segments[i][1] - comparison_segments[best_j][1]))

        mean_start_error = round(sum(start_errors) / len(start_errors), 1) if start_errors else None
        mean_end_error = round(sum(end_errors) / len(end_errors), 1) if end_errors else None

        return {
            "mean_iou": mean_iou,
            "hit_rate": hit_rate,
            "miss_rate": miss_rate,
            "false_positive_rate": fp_rate,
            "fragmentation": fragmentation,
            "mean_start_error": mean_start_error,
            "mean_end_error": mean_end_error,
        }

    def compute_iou(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        overlap = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
        if overlap == 0:
            return 0.0
        union = max(a[1], b[1]) - min(a[0], b[0])
        return overlap / union

    def evaluate(self) -> None:
        frame_based_results = {}
        segment_based_results = {}
        matching_clip_info = {}
        ground_truth_frames = {}
        for model_name in self.models:
            segments_predicted = self.find_segments_predicted(model_name)

            for video_name in segments_predicted.keys():
                prediction_file = self.prediction_folder / str(model_name) / f'{video_name}_predictions.csv'
                try:
                    predictions = pd.read_csv(prediction_file)
                except (OSError, pd.errors.ParserError) as exc:
                    logging.warning("Cannot read predictions for %s/%s: %s. Skipping.", model_name, video_name, exc)
                    continue
                if not {'frame_index', 'prediction'}.issubset(predictions.columns):
                    logging.warning("Prediction columns missing for %s/%s. Skipping.", model_name, video_name)
                    continue
                frame_indices = pd.to_numeric(predictions['frame_index'], errors='coerce')
                if frame_indices.isna().any() or not np.array_equal(np.sort(frame_indices.to_numpy()), np.arange(len(predictions))):
                    logging.warning("Prediction frames are not contiguous from zero for %s/%s. Skipping.", model_name, video_name)
                    continue
                predictions['frame_index'] = frame_indices.astype('int64')

                if video_name not in matching_clip_info:
                    matching_clip_info[video_name] = self.find_matching_clip_info(video_name)
                reference_segments = matching_clip_info[video_name]
                if reference_segments is None:
                    continue
                if video_name not in ground_truth_frames:
                    ground_truth_frames[video_name] = self.save_ground_truth_frames(video_name, reference_segments, len(predictions))
                if ground_truth_frames[video_name] is None:
                    logging.warning("Ground truth unavailable for %s/%s. Skipping.", model_name, video_name)
                    continue

                comparison_segments = segments_predicted[video_name]
                frame_metrics = self.frame_based_eval(ground_truth_frames[video_name], predictions)
                if frame_metrics is None:
                    logging.warning("Frame metrics unavailable for %s/%s. Skipping.", model_name, video_name)
                    continue
                segment_metrics = self.segment_based_eval(reference_segments, comparison_segments, self.iou_threshold)

                frame_based_results.setdefault(model_name, {})[video_name] = frame_metrics
                segment_based_results.setdefault(model_name, {})[video_name] = segment_metrics

        def average_video_metrics(results: dict) -> pd.DataFrame:
            rows = [
                {"model_name": model_name, "video_name": video_name, **metrics}
                for model_name, videos in results.items()
                for video_name, metrics in videos.items()
            ]
            if not rows:
                return pd.DataFrame(columns=["model_name"])

            return (
                pd.DataFrame(rows)
                .drop(columns="video_name")
                .groupby("model_name", sort=False, as_index=False)
                .mean(numeric_only=True)
                .round(3)
            )

        frame_based_df = average_video_metrics(frame_based_results)
        segment_based_df = average_video_metrics(segment_based_results)

        for name, result in (
            ("frame_based_evaluation_results.csv", frame_based_df),
            ("segment_based_evaluation_results.csv", segment_based_df),
        ):
            try:
                result.to_csv(self.evaluation_folder / name, index=False)
            except OSError as exc:
                logging.warning("Cannot save %s: %s", name, exc)
