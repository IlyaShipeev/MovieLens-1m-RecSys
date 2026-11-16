"""Выбор полного pipeline по январской validation."""

import json

import pandas as pd


def select_best_pipeline(root):
    comparison = pd.read_csv(root / "reports/tuning/pipeline_comparison.csv")
    comparison = comparison[comparison["k"] == 10]
    best_name = comparison.loc[comparison["ndcg"].idxmax(), "model"]

    # У каждой модели свой конфиг: нельзя менять CatBoost отдельно от retrieval.
    if best_name == "CatBoost_original":
        config_path = root / "configs/experiment.json"
        model_path = root / "artifacts/ranking/catboost_base.cbm"
    elif best_name == "CatBoost_tuned":
        config_path = root / "configs/tuned_validation.json"
        model_path = root / "artifacts/tuning/catboost_tuned.cbm"
    else:
        raise ValueError(f"Неизвестная модель в таблице validation: {best_name}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    return config, model_path
