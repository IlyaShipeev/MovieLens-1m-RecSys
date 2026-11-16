"""Small staged parameter search on January validation only."""

from copy import deepcopy
from itertools import product
from pathlib import Path
import json
import random
from time import perf_counter

import pandas as pd
from catboost import CatBoostRanker, Pool

from .data import load_movielens
from .evaluation import evaluate_recommendations
from .ranking import build_snapshot, split_snapshot
from .retrieval import fit_retrieval, recommend_als, recommend_item_knn, recommend_popular


def search_spaces(seed=42):
    """Return 12 ALS, 4 KNN and 12 CatBoost settings, including current settings."""
    als = []
    for factors, regularization, alpha in product([32, 64, 128], [0.05, 0.1], [1.0, 10.0]):
        als.append({
            "factors": factors, "regularization": regularization,
            "alpha": alpha, "iterations": 15,
        })

    knn = [20, 50, 100, 200]

    # Текущий вариант включён явно. Остальные 11 выбираются случайно без повторов.
    baseline = {"depth": 6, "learning_rate": 0.05, "l2_leaf_reg": 3, "random_strength": 1.0}
    possibilities = []
    for depth, rate, l2, strength in product([4, 6, 8], [0.03, 0.05, 0.1], [1, 3, 10], [0.5, 1.0, 2.0]):
        parameters = {"depth": depth, "learning_rate": rate, "l2_leaf_reg": l2, "random_strength": strength}
        if parameters != baseline:
            possibilities.append(parameters)
    catboost = [baseline] + random.Random(seed).sample(possibilities, 11)
    return als, knn, catboost


def merge_candidates(als, knn, popular, config):
    """Same ordered union as the notebooks: ALS150 + KNN100 + Popular fill."""
    candidates = []
    for movie_id in als[:config["retrieval"]["als_candidates"]]:
        if movie_id not in candidates:
            candidates.append(movie_id)
    for movie_id in knn[:config["retrieval"]["item_knn_candidates"]]:
        if movie_id not in candidates:
            candidates.append(movie_id)
    for movie_id in popular:
        if len(candidates) >= config["retrieval"]["max_candidates"]:
            break
        if movie_id not in candidates:
            candidates.append(movie_id)
    return candidates


def score_ranker(model, frame, features, ground_truth, evaluation_users, catalog, name):
    """Rank candidates and evaluate against ALL validation positives."""
    scored = frame[["user_id", "movie_id"]].copy()
    scored["score"] = model.predict(frame[features])
    scored = scored.sort_values(["user_id", "score", "movie_id"], ascending=[True, False, True])
    recommendations = scored.groupby("user_id")["movie_id"].apply(list).to_dict()
    _, metrics = evaluate_recommendations(
        recommendations, ground_truth, evaluation_users, catalog, name, k=10,
    )
    return metrics


def run_search(project_dir, thread_count=4):
    """Run 28 trials and save a separate tuned model; never change production config."""
    root = Path(project_dir)
    config = json.loads((root / "configs/experiment.json").read_text(encoding="utf-8"))
    tuned_config = deepcopy(config)
    report_dir = root / "reports/tuning"
    artifact_dir = root / "artifacts/tuning"
    report_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    ratings, movies, users = load_movielens(root / config["data_dir"])
    start = pd.Timestamp(config["split"]["validation_start"])
    end = pd.Timestamp(config["split"]["test_start"])
    # Февраль и более поздние оценки полностью исключаем из этого эксперимента.
    ratings = ratings[ratings["datetime"] < end].copy()
    history, target = split_snapshot(ratings, start, end)
    ground_truth = (
        target[target["rating"] >= config["positive_threshold"]]
        .groupby("user_id")["movie_id"].apply(set).to_dict()
    )
    evaluation_users = sorted(set(ground_truth) & set(history["user_id"]))
    catalog = set(history["movie_id"])
    seed = config["seed"]
    als_grid, knn_grid, catboost_grid = search_spaces(seed)

    # 1. ALS: выбираем по Recall@150 - именно столько ALS отдаёт в наш Union.
    als_results = []
    best_als_recall = -1
    best_retrieval = None
    for trial, parameters in enumerate(als_grid, start=1):
        retrieval = fit_retrieval(
            history, config["positive_threshold"], seed,
            n_neighbors=max(knn_grid), als_parameters=parameters,
        )
        recommendations = {}
        for user_id in evaluation_users:
            recommendations[user_id] = recommend_als(user_id, 250, retrieval)
        _, at_150 = evaluate_recommendations(recommendations, ground_truth, evaluation_users, catalog, "ALS", k=150)
        _, at_250 = evaluate_recommendations(recommendations, ground_truth, evaluation_users, catalog, "ALS", k=250)
        als_results.append({
            "trial": trial, **parameters, "recall_at_150": at_150["recall"],
            "recall_at_250": at_250["recall"], "ndcg_at_250": at_250["ndcg"],
        })
        if at_150["recall"] > best_als_recall:
            best_als_recall = at_150["recall"]
            tuned_config["retrieval"]["als_parameters"] = dict(parameters)
            best_retrieval = retrieval
        print(f"ALS {trial}/12: Recall@150={at_150['recall']:.6f}, Recall@250={at_250['recall']:.6f}", flush=True)
    pd.DataFrame(als_results).to_csv(report_dir / "als_trials.csv", index=False)

    # 2. ItemKNN: матрица cosine одна и та же, меняем число соседей фильма.
    # Выбираем по Recall@250 ВСЕГО Union, чтобы KNN дополнял выбранный ALS.
    full_neighbors = best_retrieval["item_neighbors"]
    als_recommendations = {}
    popular_recommendations = {}
    for user_id in evaluation_users:
        als_recommendations[user_id] = recommend_als(user_id, 250, best_retrieval)
        popular_recommendations[user_id] = recommend_popular(user_id, 250, best_retrieval)
    knn_results = []
    best_union_recall = -1
    for trial, neighbors in enumerate(knn_grid, start=1):
        best_retrieval["item_neighbors"] = {idx: values[:neighbors] for idx, values in full_neighbors.items()}
        knn_recommendations = {}
        union_recommendations = {}
        for user_id in evaluation_users:
            knn_recommendations[user_id] = recommend_item_knn(user_id, 250, best_retrieval)
            union_recommendations[user_id] = merge_candidates(
                als_recommendations[user_id], knn_recommendations[user_id],
                popular_recommendations[user_id], tuned_config,
            )
        _, at_100 = evaluate_recommendations(knn_recommendations, ground_truth, evaluation_users, catalog, "ItemKNN", k=100)
        _, at_250 = evaluate_recommendations(knn_recommendations, ground_truth, evaluation_users, catalog, "ItemKNN", k=250)
        _, union_metrics = evaluate_recommendations(union_recommendations, ground_truth, evaluation_users, catalog, "Union", k=250)
        knn_results.append({
            "trial": trial, "neighbors": neighbors, "recall_at_100": at_100["recall"],
            "recall_at_250": at_250["recall"], "union_recall_at_250": union_metrics["recall"],
        })
        if union_metrics["recall"] > best_union_recall:
            best_union_recall = union_metrics["recall"]
            tuned_config["retrieval"]["item_knn_neighbors"] = neighbors
        print(f"KNN {trial}/4: neighbors={neighbors}, Union Recall@250={union_metrics['recall']:.6f}", flush=True)
    pd.DataFrame(knn_results).to_csv(report_dir / "knn_trials.csv", index=False)
    del best_retrieval, full_neighbors, retrieval

    # 3. Пересоздаём все снимки с выбранными ALS/KNN, а не со старой моделью.
    snapshots = []
    dates = pd.date_range(config["split"]["ranker_train_start"], start, freq="MS")
    for month_start, month_end in zip(dates[:-1], dates[1:]):
        print(f"Rebuilding train snapshot {month_start.date()}...", flush=True)
        snapshots.append(build_snapshot(ratings, users, month_start, month_end, tuned_config, seed))
    train_frame = pd.concat(snapshots, ignore_index=True)
    validation_frame = build_snapshot(ratings, users, start, end, tuned_config, seed)
    assert set(validation_frame["user_id"]) == set(evaluation_users)
    assert len(snapshots) == 4
    stats = train_frame.groupby("group_id")["target"].agg(["min", "max"])
    useful_groups = stats[stats["min"] < stats["max"]].index
    train_fit = train_frame[train_frame["group_id"].isin(useful_groups)].sort_values(["group_id", "movie_id"])
    validation_frame = validation_frame.sort_values(["group_id", "movie_id"])
    features = config["ranking"]["features"]
    assert "target" not in features and len(features) == 11
    train_pool = Pool(train_fit[features], label=train_fit["target"], group_id=train_fit["group_id"])
    validation_pool = Pool(validation_frame[features], label=validation_frame["target"], group_id=validation_frame["group_id"])

    # 4. CatBoost: 12 наборов параметров; early stopping на январе.
    # Победителя сравниваем по end-to-end NDCG@10, включая промахи retrieval.
    catboost_results = []
    best_ndcg = -1
    for trial, parameters in enumerate(catboost_grid, start=1):
        model = CatBoostRanker(
            **parameters, iterations=300, loss_function="YetiRank", eval_metric="NDCG:top=10",
            random_seed=seed, thread_count=thread_count, verbose=False, allow_writing_files=False,
        )
        model.fit(train_pool, eval_set=validation_pool, early_stopping_rounds=40, use_best_model=True)
        metrics = score_ranker(model, validation_frame, features, ground_truth, evaluation_users, catalog, "CatBoost_tuned")
        catboost_results.append({
            "trial": trial, **parameters, "tree_count": model.tree_count_,
            **metrics,
        })
        if metrics["ndcg"] > best_ndcg:
            best_ndcg = metrics["ndcg"]
            model.save_model(str(artifact_dir / "catboost_tuned.cbm"))
            tuned_config["ranking"]["parameters"] = {
                **parameters, "iterations": model.tree_count_, "loss_function": "YetiRank",
                "eval_metric": "NDCG:top=10", "thread_count": thread_count, "allow_writing_files": False,
            }
            tuned_config["ranking"]["selected_model"] = "CatBoost_tuned"
            tuned_config["ranking"]["selection"] = {
                "dataset": "January 2001 validation", "primary_metric": "end_to_end_ndcg_at_10",
                "selected_best_iteration_zero_based": model.tree_count_ - 1,
                "hyperparameter_search_performed": True,
            }
            tuned_config["ranking"]["previous_february_test_already_seen"] = True
            (root / "configs/tuned_validation.json").write_text(
                json.dumps(tuned_config, indent=2, ensure_ascii=False), encoding="utf-8",
            )
        print(f"CatBoost {trial}/12: NDCG@10={metrics['ndcg']:.6f}, trees={model.tree_count_}", flush=True)
    pd.DataFrame(catboost_results).to_csv(report_dir / "catboost_trials.csv", index=False)

    # Контроль: прежняя модель на прежних кандидатах. Это не новый search trial.
    original_frame = build_snapshot(ratings, users, start, end, config, seed)
    original_model = CatBoostRanker()
    original_model.load_model(str(root / "artifacts/ranking/catboost_base.cbm"))
    original_metrics = score_ranker(original_model, original_frame, features, ground_truth, evaluation_users, catalog, "CatBoost_original")
    best_row = max(catboost_results, key=lambda row: row["ndcg"])
    tuned_metrics = {key: best_row[key] for key in original_metrics}
    pd.DataFrame([original_metrics, tuned_metrics]).to_csv(report_dir / "pipeline_comparison.csv", index=False)
    total_seconds = perf_counter() - started
    print(f"Finished 28 search trials in {total_seconds / 60:.1f} min. Original NDCG@10={original_metrics['ndcg']:.6f}; tuned={best_ndcg:.6f}", flush=True)
    return pd.DataFrame([original_metrics, tuned_metrics])
