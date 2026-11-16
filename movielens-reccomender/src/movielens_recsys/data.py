"""Read original ML-1M files without modifying them."""
from pathlib import Path

import pandas as pd


def load_movielens(data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = Path(data_dir)
    ratings = pd.read_csv(
        root / "ratings.dat", sep="::", engine="python", header=None,
        names=["user_id", "movie_id", "rating", "timestamp"],
        dtype={"user_id": "int32", "movie_id": "int32", "rating": "int8", "timestamp": "int64"},
    )
    ratings["datetime"] = pd.to_datetime(ratings["timestamp"], unit="s", utc=True)
    movies = pd.read_csv(
        root / "movies.dat", sep="::", engine="python", header=None, encoding="latin-1",
        names=["movie_id", "title", "genres"],
    )
    users = pd.read_csv(
        root / "users.dat", sep="::", engine="python", header=None,
        names=["user_id", "gender", "age", "occupation", "zip_code"],
        dtype={"zip_code": "string"},
    )
    return ratings, movies, users
