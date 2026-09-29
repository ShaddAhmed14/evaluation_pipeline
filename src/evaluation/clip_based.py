import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    precision_recall_fscore_support,
    balanced_accuracy_score,
    f1_score,
    multilabel_confusion_matrix,
)

# Metrics currently focus on 2 categories

class ClipBasedEvaluator:
    def __init__(self, metalabel_files_mapping: dict, default_label: str, models: list, input_folder: Path, output_folder: Path, evaluation_folder: Path):
        self.metalabel_files_mapping = metalabel_files_mapping
        self.labels = sorted(list(metalabel_files_mapping.keys()))
        self.models = models
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.evaluation_folder = evaluation_folder
        self.default_label = default_label

    def return_mode(self, file_path: Path, model_name: str) -> str:
        if not file_path.exists():
            logging.info(f"Output file {file_path} not found for model {model_name}. Assuming prediction is {self.default_label}.")
            mode = self.default_label
            return mode
        
        df = pd.read_csv(file_path)
        mode = df['prediction'].mode()[0]  # Get the most frequent value
        return mode

    def per_model_metrics(self, df: pd.DataFrame):
        y_true = df['true_label']
        y_pred = df['prediction']

        precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=self.labels, zero_division=0)
        mcm = multilabel_confusion_matrix(y_true, y_pred, labels=self.labels)
        tn, fp, fn, tp = mcm[:, 0, 0], mcm[:, 0, 1], mcm[:, 1, 0], mcm[:, 1, 1]
        overall_accuracy = (tp + tn) / len(y_true)  # Overall accuracy across all labels

        output = {}
        for i, label in enumerate(self.labels):
            output[f'{label}_accuracy'] = overall_accuracy[i]
            output[f'{label}_f1_score'] = f1[i]
            output[f'{label}_precision'] = precision[i]
            output[f'{label}_recall'] = recall[i]

        output['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        output['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        output['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)

        return pd.Series(output)

    def evaluate(self) -> None:
        combined_predictions = []
        # returns per model, per label, per file, the prediction mode label
        for metalabel, input_files in self.metalabel_files_mapping.items():
            for input_file in input_files:
                for model_name in self.models:
                    output_file_path = self.output_folder / model_name  / f"{input_file.stem}_predictions.csv"
                    combined_predictions.append({
                        'file_name': input_file.stem,
                        'corpus': input_file.stem.split('_', 1)[0],
                        'true_label': metalabel,
                        'model_name': model_name,
                        'prediction': self.return_mode(output_file_path, model_name)
                    })
        combined_predictions_df = pd.DataFrame(combined_predictions)

        # calculate per model metrics
        result = (
            combined_predictions_df.groupby('model_name', sort=False)
            .apply(self.per_model_metrics)
            .reset_index()
        )

        result.to_csv(self.evaluation_folder / "evaluation_metrics.csv", index=False)

        corpus_result = (
            combined_predictions_df.groupby(['model_name', 'corpus'], sort=False)
            .apply(self.per_model_metrics)
            .reset_index()
        )
        corpus_result.to_csv(self.evaluation_folder / "evaluation_metrics_by_corpus.csv", index=False)
