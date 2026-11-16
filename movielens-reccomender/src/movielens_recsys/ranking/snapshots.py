"""Build one temporal dataset for CatBoost ranking."""

import numpy as np
import pandas as pd

from movielens_recsys.retrieval import (
    fit_retrieval,
    recommend_als,
    recommend_item_knn,
    recommend_popular,
)


def split_snapshot(ratings, start, end):
    """Split ratings into history before start and answers inside [start, end)."""
    history = ratings[ratings["datetime"] < start].copy()
    target_window = ratings[
        (ratings["datetime"] >= start)
        & (ratings["datetime"] < end)
    ].copy()

    assert history["datetime"].max() < start
    return history, target_window


def build_snapshot_candidates(history, requested_users, snapshot_time, config, seed, retrieval=None):
    """Build ALS150 + KNN100 candidates and fill remaining places with Popular."""
    # Приложение может передать уже обученные модели и использовать их повторно.
    if retrieval is None:
        retrieval = fit_retrieval(
            history,
            positive_threshold=config["positive_threshold"],
            seed=seed,
            n_neighbors=config["retrieval"]["item_knn_neighbors"],
            als_parameters=config["retrieval"]["als_parameters"],
        )

    rows = []

    for user_id in requested_users:
        als_movies, als_scores = recommend_als(
            user_id,
            n_candidates=config["retrieval"]["max_candidates"],
            retrieval=retrieval,
            return_scores=True,
        )
        knn_movies, knn_scores = recommend_item_knn(
            user_id,
            n_candidates=config["retrieval"]["item_knn_candidates"],
            retrieval=retrieval,
            return_scores=True,
        )
        popular_movies = recommend_popular(
            user_id,
            n_candidates=config["retrieval"]["max_candidates"],
            retrieval=retrieval,
        )

        candidates = []

        # Сначала берём 150 кандидатов ALS.
        for movie_id in als_movies[:config["retrieval"]["als_candidates"]]:
            if movie_id not in candidates:
                candidates.append(movie_id)

        # Затем добавляем 100 кандидатов KNN без повторов.
        for movie_id in knn_movies[:config["retrieval"]["item_knn_candidates"]]:
            if movie_id not in candidates:
                candidates.append(movie_id)

        # Если после удаления повторов мест меньше 250, заполняем их Popular.
        for movie_id in popular_movies:
            if len(candidates) >= config["retrieval"]["max_candidates"]:
                break
            if movie_id not in candidates:
                candidates.append(movie_id)

        als_ranks = {}
        for rank, movie_id in enumerate(als_movies, start=1):
            if movie_id in als_scores:
                als_ranks[movie_id] = rank

        knn_ranks = {}
        for rank, movie_id in enumerate(knn_movies, start=1):
            if movie_id in knn_scores:
                knn_ranks[movie_id] = rank

        popular_ranks = {}
        for rank, movie_id in enumerate(popular_movies, start=1):
            popular_ranks[movie_id] = rank

        # Одна строка таблицы соответствует одному фильму-кандидату.
        for movie_id in candidates:
            from_als = (
                movie_id in als_movies[:config["retrieval"]["als_candidates"]]
                and movie_id in als_scores
            )
            from_knn = (
                movie_id in knn_movies[:config["retrieval"]["item_knn_candidates"]]
                and movie_id in knn_scores
            )
            rows.append({
                "snapshot_time": snapshot_time,
                "user_id": user_id,
                "movie_id": movie_id,
                "als_score": als_scores.get(movie_id, np.nan),
                "knn_score": knn_scores.get(movie_id, np.nan),
                "als_rank": als_ranks.get(movie_id, 10000),
                "knn_rank": knn_ranks.get(movie_id, 10000),
                "popular_rank": popular_ranks.get(movie_id, 10000),
                "from_als": int(from_als),
                "from_knn": int(from_knn),
                "from_popular": int(not from_als and not from_knn),
                "als_score_missing": int(movie_id not in als_scores),
                "knn_score_missing": int(movie_id not in knn_scores),
            })

    return pd.DataFrame(rows)


def add_history_features(candidates, history, users):
    """Add features using past ratings only; no future labels are needed."""
    user_stats = (
        history
        .groupby("user_id")
        .agg(
            user_rating_count=("movie_id", "count"),
            user_mean_rating=("rating", "mean"),
        )
        .reset_index()
    )
    item_stats = (
        history
        .groupby("movie_id")
        .agg(
            item_rating_count=("user_id", "count"),
            item_mean_rating=("rating", "mean"),
        )
        .reset_index()
    )

    frame = candidates.merge(user_stats, on="user_id", how="left")
    frame = frame.merge(item_stats, on="movie_id", how="left")

    # MovieLens не содержит историю изменения профиля пользователя.
    profile_columns = ["gender", "age", "occupation"]
    user_profiles = users[["user_id", *profile_columns]].copy()
    assert user_profiles["user_id"].is_unique
    frame = frame.merge(
        user_profiles,
        on="user_id",
        how="left",
        validate="many_to_one",
    )
    for column in profile_columns:
        frame[column] = frame[column].fillna("unknown").astype(str)
    assert len(frame) == len(candidates)
    return frame


def add_labels_and_features(
    candidates, history, target_window, snapshot_time, users, positive_threshold,
):
    """Add future labels for offline training/evaluation after computing features."""
    frame = add_history_features(candidates, history, users)
    positive_target = target_window[target_window["rating"] >= positive_threshold]
    ground_truth = positive_target.groupby("user_id")["movie_id"].apply(set).to_dict()

    targets = []
    for row in frame.itertuples():
        target = int(
            row.movie_id in ground_truth.get(row.user_id, set())
        )
        targets.append(target)
    frame["target"] = targets

    # Один пользователь в разные месяцы образует разные группы CatBoost.
    frame["group_id"] = (
        snapshot_time.strftime("%Y-%m-%d")
        + "_user_"
        + frame["user_id"].astype(str)
    )

    assert not frame.duplicated(["group_id", "movie_id"]).any()
    seen = history.groupby("user_id")["movie_id"].apply(set).to_dict()
    for row in frame.itertuples():
        assert row.movie_id not in seen.get(row.user_id, set())
    assert frame["target"].isin([0, 1]).all()
    return frame


def build_snapshot(ratings, users, start, end, config, seed):
    """Build candidates, features and labels for one calendar window."""
    history, target_window = split_snapshot(ratings, start, end)

    ground_truth = (
        target_window[target_window["rating"] >= config["positive_threshold"]]
        .groupby("user_id")["movie_id"]
        .apply(set)
        .to_dict()
    )
    eligible_users = sorted(set(history["user_id"]) & set(ground_truth))

    candidates = build_snapshot_candidates(
        history=history,
        requested_users=eligible_users,
        snapshot_time=start,
        config=config,
        seed=seed,
    )
    return add_labels_and_features(
        candidates=candidates,
        history=history,
        target_window=target_window,
        snapshot_time=start,
        users=users,
        positive_threshold=config["positive_threshold"],
    )
