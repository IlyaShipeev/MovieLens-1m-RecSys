"""The notebook algorithms with explicit data instead of global variables.

``retrieval`` is an ordinary dictionary containing one cutoff's models and data.
By default recommenders return a list of original movie IDs (for notebook 03).
With ``return_scores=True`` they always return ``(movie_ids, scores_by_movie_id)``.
ALS/KNN scores describe only real model candidates. Popular fallback has no
ALS/KNN score: the caller can represent missing values with NaN.
"""

import numpy as np
from scipy.sparse import csr_matrix


def fit_retrieval(history, positive_threshold=4, seed=42, n_neighbors=50, als_parameters=None):
    """Train fresh models on the supplied history, without reading future data."""
    from implicit.als import AlternatingLeastSquares
    from sklearn.metrics.pairwise import cosine_similarity

    if history.empty:
        raise ValueError("Retrieval needs a nonempty history")
    if n_neighbors <= 0:
        raise ValueError("n_neighbors must be positive")

    # 1. Положительная история
    positive_history = history[history["rating"] >= positive_threshold]
    # 2. Seen по всем оценкам 1–5
    seen_movies = history.groupby("user_id")["movie_id"].apply(set).to_dict()

    user_ids = np.sort(history["user_id"].unique())
    movie_ids = np.sort(history["movie_id"].unique())
    user_mapper = dict(zip(user_ids, range(len(user_ids))))
    movie_mapper = dict(zip(movie_ids, range(len(movie_ids))))
    movie_inv_mapper = dict(zip(range(len(movie_ids)), movie_ids))

    positive_pairs = positive_history[["user_id", "movie_id"]].drop_duplicates()
    values = np.ones(len(positive_pairs), dtype=np.float32)
    row_indices = positive_pairs["user_id"].map(user_mapper).to_numpy(dtype=int)
    column_indices = positive_pairs["movie_id"].map(movie_mapper).to_numpy(dtype=int)
    X = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(user_mapper), len(movie_mapper)),
        dtype=np.float32,
    )

    popularity = (
        positive_history.groupby("movie_id").size()
        .reset_index(name="popularity")
        .sort_values(["popularity", "movie_id"], ascending=[False, True])
    )

    parameters = {"factors": 32, "regularization": 0.1, "iterations": 15, "alpha": 1.0}
    if als_parameters is not None:
        parameters.update(als_parameters)
    als_model = AlternatingLeastSquares(
        **parameters,
        use_gpu=False, random_state=seed, calculate_training_loss=True,
    )
    als_model.fit(X, show_progress=False)

    item_similarity = cosine_similarity(X.T.tocsr(), dense_output=True)
    np.fill_diagonal(item_similarity, 0)
    item_neighbors = {}
    for item_idx in range(item_similarity.shape[0]):
        similarities = item_similarity[item_idx]
        sorted_indices = np.argsort(-similarities, kind="stable")
        neighbors = []
        for neighbor_idx in sorted_indices:
            if similarities[neighbor_idx] <= 0:
                break
            neighbors.append(neighbor_idx)
            if len(neighbors) == n_neighbors:
                break
        item_neighbors[item_idx] = neighbors

    return {
        "X": X,
        "user_mapper": user_mapper,
        "movie_mapper": movie_mapper,
        "movie_inv_mapper": movie_inv_mapper,
        "seen_movies": seen_movies,
        "popular_movie_ids": popularity["movie_id"].tolist(),
        "popularity_scores": popularity.set_index("movie_id")["popularity"].to_dict(),
        "als_model": als_model,
        "item_similarity": item_similarity,
        "item_neighbors": item_neighbors,
    }


def recommend_popular(user_id, n_candidates, retrieval, return_scores=False):
    """Return unseen popular movie IDs; optional scores are positive counts."""
    seen = retrieval["seen_movies"].get(user_id, set())
    recommendations = []
    for movie_id in retrieval["popular_movie_ids"]:
        if len(recommendations) >= max(0, n_candidates):
            break
        if movie_id not in seen:
            recommendations.append(int(movie_id))

    if return_scores:
        scores_by_movie_id = {}
        for movie_id in recommendations:
            scores_by_movie_id[movie_id] = float(retrieval["popularity_scores"][movie_id])
        return recommendations, scores_by_movie_id
    return recommendations


def recommend_als(user_id, n_candidates, retrieval, return_scores=False):
    """Return unseen ALS candidates, then Popular fallback if needed."""
    X = retrieval["X"]
    user_mapper = retrieval["user_mapper"]
    movie_mapper = retrieval["movie_mapper"]
    movie_inv_mapper = retrieval["movie_inv_mapper"]
    seen_movies = retrieval["seen_movies"]
    als_model = retrieval["als_model"]

    # Проверка границ: пустой запрос и неизвестный пользователь.
    if n_candidates <= 0:
        return ([], {}) if return_scores else []
    if user_id not in user_mapper or X[user_mapper[user_id]].nnz == 0:
        recommendations = recommend_popular(user_id, n_candidates, retrieval)
        # Popular-счётчики нельзя выдавать за ALS-score.
        return (recommendations, {}) if return_scores else recommendations

    user_idx = user_mapper[user_id]
    user_vector = als_model.user_factors[user_idx]
    scores = (als_model.item_factors @ user_vector).copy()
    # Без положительных событий у фильма нет полезного ALS-сигнала.
    positive_item_mask = np.asarray(X.getnnz(axis=0)).ravel() > 0
    scores[~positive_item_mask] = -np.inf
    seen = seen_movies.get(user_id, set())
    seen_indices = [
        movie_mapper[movie_id]
        for movie_id in seen
        if movie_id in movie_mapper
    ]
    scores[seen_indices] = -np.inf
    top_indices = np.argsort(-scores, kind="stable")[:n_candidates]
    recommendations = []
    scores_by_movie_id = {}
    for item_idx in top_indices:
        if not np.isfinite(scores[item_idx]):
            continue
        movie_id = int(movie_inv_mapper[item_idx])
        recommendations.append(movie_id)
        scores_by_movie_id[movie_id] = float(scores[item_idx])

    # Если доступных ALS-кандидатов мало, дополняем Popular без дублей и seen.
    for movie_id in recommend_popular(user_id, n_candidates, retrieval):
        if len(recommendations) >= n_candidates:
            break
        if movie_id not in recommendations:
            recommendations.append(movie_id)

    if return_scores:
        return recommendations, scores_by_movie_id
    return recommendations


def recommend_item_knn(user_id, n_candidates, retrieval, return_scores=False):
    """Return item-neighbor candidates with scores keyed by original movie ID."""
    X = retrieval["X"]
    user_mapper = retrieval["user_mapper"]
    movie_mapper = retrieval["movie_mapper"]
    movie_inv_mapper = retrieval["movie_inv_mapper"]
    seen_movies = retrieval["seen_movies"]
    item_neighbors = retrieval["item_neighbors"]
    item_similarity = retrieval["item_similarity"]

    # Проверка границ: пустой запрос и неизвестный пользователь.
    if n_candidates <= 0:
        return ([], {}) if return_scores else []
    if user_id not in user_mapper:
        recommendations = recommend_popular(user_id, n_candidates, retrieval)
        return (recommendations, {}) if return_scores else recommendations
    user_idx = user_mapper[user_id]

    # Внутренние индексы положительных фильмов пользователя
    positive_item_indices = X[user_idx].indices
    if len(positive_item_indices) == 0:
        recommendations = recommend_popular(user_id, n_candidates, retrieval)
        return (recommendations, {}) if return_scores else recommendations

    candidate_scores = {}
    # Для каждого понравившегося фильма
    for positive_item_idx in positive_item_indices:
        # Берём его всех 50 соседей
        neighbors = item_neighbors[positive_item_idx]
        for neighbor_idx in neighbors:  # сходство исходного фильма с соседним фильмом
            similarity = item_similarity[positive_item_idx, neighbor_idx]
            # Один кандидат может прийти от нескольких фильмов
            if neighbor_idx not in candidate_scores:
                candidate_scores[neighbor_idx] = 0
            candidate_scores[neighbor_idx] += float(similarity)

    # Исключаем всю просмотренную историю
    seen = seen_movies.get(user_id, set())
    seen_indices = [
        movie_mapper[movie_id]
        for movie_id in seen
        if movie_id in movie_mapper
    ]
    for seen_idx in seen_indices:
        candidate_scores.pop(seen_idx, None)  # просмотренные из кандидатов удалить

    # score DESC, movie_id ASC при одинаковом score
    sorted_candidates = sorted(
        candidate_scores.items(),
        key=lambda pair: (
            -pair[1],  # сортировка по убыванию, т.к. по дефолту по возрастанию
            movie_inv_mapper[pair[0]],  # по возрст исх id
        ),
    )
    recommendations = []
    scores_by_movie_id = {}
    for item_idx, score in sorted_candidates[:n_candidates]:
        movie_id = int(movie_inv_mapper[item_idx])
        recommendations.append(movie_id)
        # Новый словарь: movie_id -> score.
        scores_by_movie_id[movie_id] = float(score)

    # Если KNN нашёл мало кандидатов - дополняем Popular
    used = set(recommendations)
    for movie_id in recommend_popular(user_id, n_candidates, retrieval):
        if len(recommendations) >= n_candidates:
            break
        if movie_id not in used:
            recommendations.append(movie_id)
            used.add(movie_id)

    if return_scores:
        return recommendations, scores_by_movie_id
    return recommendations
