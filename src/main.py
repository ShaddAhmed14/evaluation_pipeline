import os
import yaml
import docker
import logging
import importlib
from dotenv import load_dotenv
from datetime import datetime
from pathlib import Path
from evaluation.clip_based import ClipBasedEvaluator
from evaluation.video_based import VideoBasedEvaluator

load_dotenv()  # Load environment variables from .env file

def main():
    current_session = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_folder = Path("Logs")
    log_folder.mkdir(parents=True, exist_ok=True)
    logs_file_path = log_folder / f"{current_session}.log"
    logging.basicConfig(level=logging.INFO, filename=logs_file_path, format='%(levelname)s - %(filename)s:%(funcName)s%(lineno)d - %(message)s')

    docker_client = docker.from_env()

    INPUT_FOLDER_CLIPS = Path(os.getenv("INPUT_FOLDER_CLIPS"))
    INPUT_FOLDER_VIDEOS = Path(os.getenv("INPUT_FOLDER_VIDEOS"))
    CLIPS_INFO_FOLDER = Path(os.getenv("CLIPS_INFO_FOLDER"))
    
    PREDICTION_FOLDER_CLIP_BASED = Path(os.getenv("PREDICTION_FOLDER_CLIP_BASED"))
    PREDICTION_FOLDER_VIDEOS_BASED = Path(os.getenv("PREDICTION_FOLDER_VIDEOS_BASED"))
    SAVE_GROUND_TRUTH_FOLDER = Path(os.getenv("SAVE_GROUND_TRUTH_FOLDER"))
    EVALUATION_FOLDER = Path(os.getenv("EVALUATION_FOLDER"))
    EVALUATION_FOLDER_CLIPS = EVALUATION_FOLDER / "clip_based"
    EVALUATION_FOLDER_VIDEOS = EVALUATION_FOLDER / "video_based"

    PREDICTION_FOLDER_CLIP_BASED.mkdir(parents=True, exist_ok=True)
    PREDICTION_FOLDER_VIDEOS_BASED.mkdir(parents=True, exist_ok=True)
    EVALUATION_FOLDER.mkdir(parents=True, exist_ok=True)
    EVALUATION_FOLDER_CLIPS.mkdir(parents=True, exist_ok=True)
    EVALUATION_FOLDER_VIDEOS.mkdir(parents=True, exist_ok=True)
    SAVE_GROUND_TRUTH_FOLDER.mkdir(parents=True, exist_ok=True)

    config = load_config()
    default_label = config.get("default_label")
    segment_label = config.get("segment_label") # video based

    models = load_models(config.get("models"), docker_client)
    model_names = [model.name for model in models]

    run_clips_pipeline(INPUT_FOLDER_CLIPS, PREDICTION_FOLDER_CLIP_BASED, models, default_label, model_names, EVALUATION_FOLDER_CLIPS)
    run_videos_pipeline(INPUT_FOLDER_VIDEOS, PREDICTION_FOLDER_VIDEOS_BASED, segment_label, model_names, models, EVALUATION_FOLDER_VIDEOS, CLIPS_INFO_FOLDER, SAVE_GROUND_TRUTH_FOLDER, default_label)

def run_videos_pipeline(INPUT_FOLDER: Path, PREDICTION_FOLDER: Path, segment_label: str, model_names: list, models: list, EVALUATION_FOLDER: Path, CLIPS_INFO_FOLDER: Path, SAVE_GROUND_TRUTH_FOLDER: Path, default_label: str):
    print('Running models on VIDEOS INPUT data...')
    run_model(models, [INPUT_FOLDER], PREDICTION_FOLDER)

    video_files_mapping = {}
    files = list(INPUT_FOLDER.glob('*npz'))
    for file in files:
        video_name = file.stem.replace('_predictions', '')
        video_files_mapping[video_name] = file

    print('Evaluating model performance on VIDEOS...')
    video_based_evaluator = VideoBasedEvaluator(
        models=model_names,
        video_files_mapping=video_files_mapping,
        prediction_folder=PREDICTION_FOLDER,
        evaluation_folder=EVALUATION_FOLDER,
        clips_info_folder=CLIPS_INFO_FOLDER,
        save_ground_truth_folder=SAVE_GROUND_TRUTH_FOLDER,
        segment_label=segment_label,
        default_label=default_label
    )
    video_based_evaluator.evaluate()

def run_clips_pipeline(INPUT_FOLDER_CLIPS: Path, PREDICTION_FOLDER_CLIP_BASED: Path, models: list, default_label: str, model_names: list, EVALUATION_FOLDER_CLIPS: Path):
    metalabel_files_mapping = return_metalabel_mapping(INPUT_FOLDER_CLIPS)
    for metalabel, files in metalabel_files_mapping.items():
        print(f"Metalabel: {metalabel}, Number of files: {len(files)}")

    print('Running models on CLIPS INPUT data...')
    metalabel_folders = [INPUT_FOLDER_CLIPS / metalabel for metalabel in list(metalabel_files_mapping.keys())]
    run_model(models, metalabel_folders, PREDICTION_FOLDER_CLIP_BASED)

    print('Evaluating model performance on CLIPS...')
    clip_based_evaluator = ClipBasedEvaluator(metalabel_files_mapping, default_label, model_names, INPUT_FOLDER_CLIPS, PREDICTION_FOLDER_CLIP_BASED, EVALUATION_FOLDER_CLIPS)
    clip_based_evaluator.evaluate()

def return_metalabel_mapping(input_folder: Path) -> dict:
    metalabels = [dir_name.stem for dir_name in input_folder.iterdir() if dir_name.is_dir()]
    metalabel_files_mapping = {}
    for metalabel in metalabels:
        metalabel_folder = input_folder / metalabel
        files = list(metalabel_folder.glob("*.npz"))
        metalabel_files_mapping[metalabel] = files
    return metalabel_files_mapping

def run_model(models: list, metalabel_folders: list, OUTPUT_FOLDER: Path):
    for model in models:
        for folder in metalabel_folders:
            model.run_container(folder, OUTPUT_FOLDER)

def parse_code_file(code_file: str) -> tuple[str, str]:
    if not code_file or '.' not in code_file:
        raise ValueError(f"Invalid code_file format: {code_file}. Expected format: 'module.path.ClassName'")
    
    parts = code_file.split('.')
    class_name = parts[-1]
    module_path = '.'.join(parts[:-1])
    
    return module_path, class_name

def load_config() -> dict:
    config_file_path = Path(__file__).resolve().with_name("config.yaml")

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Configuration file is empty or not found.")

    return config

def load_models(model_specifications: dict, docker_client) -> list:
    models = []
    for model_specification in model_specifications:
        try:
            if not model_specification.get("enabled", False):
                logging.info(f"Model {model_specification.get('name')} is disabled in the configuration. Skipping.")
                continue
            model_name = model_specification.get("name")
            module_path, class_name = parse_code_file(model_specification.get("module"))
            model_module = importlib.import_module(module_path)
            model_class = getattr(model_module, class_name)
            model = model_class(docker_client, model_name)
            models.append(model)
        except (ImportError, AttributeError, TypeError) as e:
            print(f"Error instantiating model {model_specification.get('name')}: {e}")
            raise e

    return models

if __name__ == "__main__":
    main()
