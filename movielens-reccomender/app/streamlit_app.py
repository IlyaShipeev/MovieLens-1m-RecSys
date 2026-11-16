"""Run with: python -m streamlit run app/streamlit_app.py"""

from pathlib import Path
import json
import sys

import pandas as pd
import streamlit as st
from catboost import CatBoostRanker

# Позволяет запускать приложение без установки проекта через pip install -e.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from movielens_recsys.data import load_movielens
from movielens_recsys.ranking import select_best_pipeline
from movielens_recsys.ranking.snapshots import (
    add_history_features,
    build_snapshot_candidates,
)
from movielens_recsys.retrieval import fit_retrieval, recommend_popular


@st.cache_resource(show_spinner="Готовим модели рекомендаций. Первый запуск займёт немного времени…")
def load_resources(config_text, model_path, model_modified):
    # Обучаем retrieval один раз, затем используем его для всех пользователей.
    config = json.loads(config_text)
    ratings, movies, users = load_movielens(ROOT / config["data_dir"])
    cutoff = pd.Timestamp(config["split"]["test_start"])
    history = ratings[ratings["datetime"] < cutoff].copy()
    # Будущие оценки нужны только экрану сравнения, не моделям.
    test_end = pd.Timestamp(config["split"]["test_end"])
    observed_ratings = ratings[
        (ratings["datetime"] >= cutoff) & (ratings["datetime"] < test_end)
    ].copy()
    retrieval = fit_retrieval(
        history,
        positive_threshold=config["positive_threshold"],
        seed=config["seed"],
        n_neighbors=config["retrieval"]["item_knn_neighbors"],
        als_parameters=config["retrieval"]["als_parameters"],
    )
    ranker = CatBoostRanker()
    ranker.load_model(model_path)
    assert ranker.feature_names_ == config["ranking"]["features"]
    return config, history, movies, users, retrieval, ranker, observed_ratings


def recommend_for_user(user_id, resources):
    config, history, movies, users, retrieval, ranker = resources
    cutoff = pd.Timestamp(config["split"]["test_start"])

    # Здесь нет target и будущих оценок. Нужны только кандидаты и история.
    candidates = build_snapshot_candidates(
        history, [user_id], cutoff, config, config["seed"], retrieval=retrieval,
    )
    frame = add_history_features(candidates, history, users)
    frame["score"] = ranker.predict(frame[config["ranking"]["features"]])
    ranked = frame.sort_values(["score", "movie_id"], ascending=[False, True])

    popular_ids = recommend_popular(user_id, config["retrieval"]["max_candidates"], retrieval)
    popular = pd.DataFrame({"movie_id": popular_ids})
    return {
        "CatBoost": ranked.merge(movies, on="movie_id", how="left", sort=False),
        "Union": frame.merge(movies, on="movie_id", how="left", sort=False),
        "MostPopular": popular.merge(movies, on="movie_id", how="left", sort=False),
    }


st.set_page_config(page_title="MovieLens · Что посмотреть", page_icon="🎬", layout="wide")
st.title("🎬 Что посмотреть дальше?")
st.caption("Персональные рекомендации на MovieLens 1M · ALS + ItemKNN → CatBoost")

config, model_path = select_best_pipeline(ROOT)
if not model_path.exists():
    st.error("Сначала сохраните модели в ноутбуке 04 и выполните подбор параметров из ноутбука 05.")
    st.stop()

config_text = json.dumps(config)
resources = load_resources(config_text, str(model_path), model_path.stat().st_mtime_ns)
config, history, movies, users, retrieval, ranker, observed_ratings = resources

with st.sidebar:
    st.header("Ваш кинопрофиль")
    cohort = st.selectbox("Каких пользователей показать", [
        "С оценками 4–5 в феврале",
        "С любыми оценками в феврале",
        "Всех с историей до февраля",
    ], key="user_cohort")
    history_users = set(history["user_id"])
    if cohort == "С оценками 4–5 в феврале":
        february_users = set(observed_ratings.loc[
            observed_ratings["rating"] >= config["positive_threshold"], "user_id",
        ])
        user_ids = sorted(history_users & february_users)
    elif cohort == "С любыми оценками в феврале":
        user_ids = sorted(history_users & set(observed_ratings["user_id"]))
    else:
        user_ids = sorted(history_users)
    if not user_ids:
        st.info("В выбранной группе нет пользователей с историей.")
        st.stop()
    st.caption(f"Доступно пользователей: {len(user_ids)}")
    default_index = user_ids.index(19) if 19 in user_ids else 0
    user_id = st.selectbox("Пользователь MovieLens", user_ids, index=default_index, key="user_id")
    method = st.selectbox("Способ рекомендаций", ["CatBoost", "Union", "MostPopular"], key="method")
    top_k = st.selectbox("Сколько фильмов показать (K)", [5, 10, 20, 50, 100], index=1, key="top_k")
    st.caption("CatBoost упорядочивает кандидатов по персональному score. Union сохраняет порядок ALS → KNN → Popular.")
    st.caption("Демо использует только оценки до 1 февраля 2001 года. Ранее оценённые фильмы исключены.")

user_history = history[history["user_id"] == user_id]
left, middle, right = st.columns(3)
left.metric("Оценено фильмов", len(user_history))
middle.metric("Понравилось", int((user_history["rating"] >= config["positive_threshold"]).sum()))
right.metric("Средняя оценка", f"{user_history['rating'].mean():.1f} / 5")

recommendations_tab, history_tab, comparison_tab, results_tab = st.tabs([
    "Рекомендации", "История пользователя", "Что оценил в феврале", "Результаты проекта",
])

with recommendations_tab:
    # Оценки февраля не передаём функции рекомендаций.
    tables = recommend_for_user(user_id, resources[:6])
    selected = tables[method].head(top_k).copy()
    actual = observed_ratings[observed_ratings["user_id"] == user_id].copy()
    actual_ratings = actual.set_index("movie_id")["rating"]
    positive_ids = set(actual.loc[actual["rating"] >= config["positive_threshold"], "movie_id"])
    st.subheader(f"{method} · {len(selected)} фильмов для пользователя {user_id}")
    selected.insert(0, "Место", range(1, len(selected) + 1))
    selected["Оценка в феврале"] = selected["movie_id"].map(actual_ratings).astype("Int64")
    selected["Попадание (4–5)"] = selected["movie_id"].isin(positive_ids)
    shown = selected[["Место", "title", "genres", "Оценка в феврале", "Попадание (4–5)"]].rename(
        columns={"title": "Фильм", "genres": "Жанры"},
    )
    st.dataframe(shown, hide_index=True, width="stretch")
    st.caption("Пустая оценка означает, что в феврале оценки нет. Это не доказательство, что фильм не понравился. Попадание - реальная оценка 4–5.")
    st.download_button("Скачать рекомендации CSV", shown.to_csv(index=False).encode("utf-8-sig"), "recommendations.csv", "text/csv")
    with st.expander("Из каких кандидатов выбирал CatBoost?"):
        st.caption("До 150 фильмов ALS и 100 ItemKNN, без повторов. Свободные места до 250 заполняет MostPopular.")
        details = tables["CatBoost"][["title", "score", "als_rank", "knn_rank", "from_popular"]].head(top_k)
        st.dataframe(details, hide_index=True, width="stretch")
        st.caption("Score задаёт порядок фильмов; это не вероятность. Rank 10000 означает отсутствие позиции в источнике.")

with history_tab:
    st.subheader("Что пользователь уже оценил")
    liked_only = st.checkbox("Только оценки 4–5", value=True)
    shown_history = user_history.copy()
    if liked_only:
        shown_history = shown_history[shown_history["rating"] >= config["positive_threshold"]]
    shown_history = shown_history.sort_values(["datetime", "movie_id"], ascending=[False, True])
    shown_history = shown_history.merge(movies, on="movie_id", how="left")
    st.dataframe(
        shown_history[["title", "genres", "rating", "datetime"]].rename(
            columns={"title": "Фильм", "genres": "Жанры", "rating": "Оценка", "datetime": "Дата оценки"},
        ), hide_index=True, width="stretch",
    )

with comparison_tab:
    st.subheader("Рекомендации и реальные оценки за февраль")
    st.caption("Период: 1 февраля - 1 марта 2001 года, не включая 1 марта. Эти оценки показываются после выдачи и не участвуют в рекомендациях.")
    if actual.empty:
        st.info("У этого пользователя нет оценок за февраль. Рекомендации доступны, но сравнить их с февральскими ответами нельзя.")
    else:
        top_ids = set(selected["movie_id"])
        candidate_ids = set(tables["Union"]["movie_id"])
        hits = len(top_ids & positive_ids)
        first, second, third = st.columns(3)
        first.metric("Оценок за февраль", len(actual))
        second.metric("Понравилось в феврале", len(positive_ids))
        third.metric(f"Попаданий в top-{top_k}", hits)

        if positive_ids:
            st.write(f"Recall@{top_k} этого пользователя: {hits} / {len(positive_ids)} = {hits / len(positive_ids):.1%}")
        else:
            st.info("Положительных оценок за февраль нет, поэтому Recall для этого пользователя не определён.")

        actual = actual.sort_values(["rating", "movie_id"], ascending=[False, True])
        actual["В выбранном top-K"] = actual["movie_id"].isin(top_ids)
        actual["Среди 250 кандидатов Union"] = actual["movie_id"].isin(candidate_ids)
        actual = actual.merge(movies, on="movie_id", how="left")
        st.dataframe(
            actual[["title", "genres", "rating", "datetime", "В выбранном top-K", "Среди 250 кандидатов Union"]].rename(
                columns={"title": "Фильм", "genres": "Жанры", "rating": "Оценка", "datetime": "Дата оценки"},
            ), hide_index=True, width="stretch",
        )
        st.caption("Если понравившегося фильма нет среди кандидатов Union, retrieval его пропустил. Если он среди кандидатов, но вне top-K, он не попал в верхнюю часть выбранной выдачи. MostPopular строит собственный список.")

with results_tab:
    st.subheader("Финальная проверка за февраль 2001")
    test_path = ROOT / "reports/ranking/february_test_metrics.csv"
    if test_path.exists():
        metrics = pd.read_csv(test_path)
        st.caption(f"Общие метрики для top-{top_k} на февральских пользователях с историей и положительными оценками.")
        st.dataframe(metrics[metrics["k"] == top_k], hide_index=True, width="stretch")
    else:
        st.info("Результаты появятся после выполнения ноутбука 06.")
    st.caption("CatBoost обучен на снимках сентября–декабря, лучший вариант выбран по январю. Февраль используется для оценки. Он уже был просмотрен до дополнительного подбора параметров, поэтому повторный запуск не является новой независимой проверкой.")
