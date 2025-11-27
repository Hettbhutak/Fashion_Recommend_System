import random
from typing import List, Tuple

import pandas as pd


def generate_synthetic_fashion_data(
	num_users: int = 100,
	num_items: int = 50,
	categories: List[str] | None = None,
	price_range: Tuple[int, int] = (20, 100),
	seed: int | None = 42,
) -> pd.DataFrame:
	if seed is not None:
		random.seed(seed)

	if categories is None:
		categories = [
			"Shirt",
			"Pants",
			"Shoes",
			"Jacket",
			"Hat",
			"Gloves",
			"Sweater",
			"Dress",
			"Scarf",
			"Boots",
		]

	# Each user buys 10 items on average
	n_rows = num_users * 10
	user_ids = [random.randint(1, num_users) for _ in range(n_rows)]
	item_ids = [random.randint(1, num_items) for _ in range(n_rows)]
	ratings = [random.randint(1, 5) for _ in range(n_rows)]

	# Item categories and prices by item id
	item_categories = [random.choice(categories) for _ in range(num_items)]
	prices = {i: round(random.uniform(price_range[0], price_range[1]), 2) for i in range(1, num_items + 1)}

	data = {
		"User_ID": user_ids,
		"Item_ID": item_ids,
		"Rating": ratings,
		"Category": [item_categories[item_id - 1] for item_id in item_ids],
		"Price": [prices[item_id] for item_id in item_ids],
	}
	return pd.DataFrame(data)


def load_csv(path_or_buffer) -> pd.DataFrame:
	return pd.read_csv(path_or_buffer)


def save_csv(df: pd.DataFrame, path: str) -> None:
	df.to_csv(path, index=False)


