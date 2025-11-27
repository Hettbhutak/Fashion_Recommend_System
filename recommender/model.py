from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def build_pivot_table(ratings_df: pd.DataFrame) -> pd.DataFrame:
	# Pivot to user-item matrix with ratings
	pivot = ratings_df.pivot_table(
		index="User_ID",
		columns="Item_ID",
		values="Rating",
		aggfunc="mean",
	)
	return pivot


def compute_user_similarity(pivot: pd.DataFrame) -> pd.DataFrame:
	pivot_filled = pivot.fillna(0)
	sim = cosine_similarity(pivot_filled)
	return pd.DataFrame(sim, index=pivot.index, columns=pivot.index)


def compute_item_similarity(pivot: pd.DataFrame) -> pd.DataFrame:
	pivot_filled = pivot.fillna(0)
	sim = cosine_similarity(pivot_filled.T)
	return pd.DataFrame(sim, index=pivot.columns, columns=pivot.columns)


def recommend_for_user(
	user_id: int,
	pivot: pd.DataFrame,
	user_similarity_df: pd.DataFrame,
	n_recommendations: int = 5,
	min_rating_threshold: float = 4.0,
) -> List[int]:
	if user_id not in pivot.index:
		return []

	# Sort similar users (exclude the user itself)
	similar_users = user_similarity_df.loc[user_id].sort_values(ascending=False).iloc[1:]

	recommendations: list[int] = []
	for similar_user in similar_users.index:
		user_ratings = pivot.loc[similar_user]
		high_rated_items = user_ratings[user_ratings > min_rating_threshold].index.tolist()
		recommendations.extend(high_rated_items)

	# Remove items already rated by the target user
	already_rated = pivot.loc[user_id].dropna().index.tolist()
	unique_recs = [i for i in list(dict.fromkeys(recommendations)) if i not in already_rated]
	return unique_recs[:n_recommendations]


def recommend_similar_items(
	item_id: int,
	item_similarity_df: pd.DataFrame,
	n_recommendations: int = 5,
) -> List[int]:
	if item_id not in item_similarity_df.index:
		return []
	series = item_similarity_df.loc[item_id].sort_values(ascending=False)
	series = series.iloc[1:]  # exclude self
	return series.index.to_list()[:n_recommendations]


