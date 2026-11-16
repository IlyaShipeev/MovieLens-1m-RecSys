"""The same binary ranking metrics used in notebook 03."""
import math

import pandas as pd


def evaluate_recommendations(
    recommendations, ground_truth, eligible_users, catalog, model_name, k=10,
):
    """Macro metrics over the complete cohort, with full ground-truth IDCG."""
    assert k > 0 and len(eligible_users) > 0 and len(catalog) > 0
    rows = []
    recommended_items = set()
    for user_id in eligible_users:
        recs = recommendations[user_id][:k]
        relevant = ground_truth[user_id]
        assert relevant
        assert len(recs) == len(set(recs))
        assert set(recs).issubset(catalog)
        recommended_items.update(recs)
        hits = sum(movie_id in relevant for movie_id in recs)
        dcg = sum(
            1 / math.log2(rank + 1)
            for rank, movie_id in enumerate(recs, start=1)
            if movie_id in relevant
        )
        idcg = sum(
            1 / math.log2(rank + 1)
            for rank in range(1, min(len(relevant), k) + 1)
        )
        rows.append({
            "user_id": user_id,
            "recommendation_count": len(recs),
            "precision": hits / k,
            "recall": hits / len(relevant),
            "hitrate": float(hits > 0),
            "ndcg": dcg / idcg,
        })
    by_user = pd.DataFrame(rows)
    result = {
        "model": model_name,
        "k": k,
        "precision": by_user["precision"].mean(),
        "recall": by_user["recall"].mean(),
        "hitrate": by_user["hitrate"].mean(),
        "ndcg": by_user["ndcg"].mean(),
        "catalog_coverage": len(recommended_items) / len(catalog),
        "mean_list_length": by_user["recommendation_count"].mean(),
        "users": len(eligible_users),
    }
    return by_user, result
