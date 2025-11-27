import io
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from recommender.data import generate_synthetic_fashion_data, load_csv
from recommender.model import (
	build_pivot_table,
	compute_item_similarity,
	compute_user_similarity,
	recommend_for_user,
	recommend_similar_items,
)


st.set_page_config(page_title="Fashion Recommender", layout="wide")
st.title("Fashion Recommendation Demo")

# Basic card styles
st.markdown(
	"""
	<style>
	.card {
		border: 1px solid #e6e6e6;
		border-radius: 10px;
		padding: 10px;
		box-shadow: 0 2px 8px rgba(0,0,0,0.06);
		background: #ffffff;
		text-align: center; /* center all text inside card */
	}
	.card-title {
		font-weight: 600;
		margin-top: 8px;
		margin-bottom: 4px;
		text-align: center;
	}
	.card-sub {
		color: #5c5c5c;
		font-size: 0.9rem;
		margin-bottom: 0;
		text-align: center;
	}
	.price-badge {
		display: inline-block;
		background: #eef7ff;
		color: #0366d6;
		border-radius: 999px;
		padding: 2px 10px;
		font-size: 0.85rem;
		margin-top: 6px;
		margin-left: auto;
		margin-right: auto; /* center the badge */
	}
	</style>
	""",
	unsafe_allow_html=True,
)


# Simple demo image mapping per category (royalty-free Unsplash sources)
CATEGORY_IMAGE_URLS = {
	"Shirt": [
		"https://images.unsplash.com/photo-1520975922284-7b683fe724d3?w=800",
		"https://images.unsplash.com/photo-1602810318383-9e6f9fdb57fd?w=800",
	],
	"Pants": [
		"https://images.unsplash.com/photo-1541099649105-f69ad21f3246?w=800",
		"https://images.unsplash.com/photo-1542272604-787c3835535d?w=800",
	],
	"Shoes": [
		"https://images.unsplash.com/photo-1528701800489-20be3c2ea41f?w=800",
		"https://images.unsplash.com/photo-1519741497674-611481863552?w=800",
	],
	"Jacket": [
		"https://images.unsplash.com/photo-1520975594081-6f2aeb4b90b1?w=800",
		"https://images.unsplash.com/photo-1539533018447-63fcce2678e3?w=800",
	],
	"Hat": [
		"https://images.unsplash.com/photo-1521572267360-ee0c2909d518?w=800",
		"https://images.unsplash.com/photo-1520974735194-6d78e08c6c5d?w=800",
	],
	"Gloves": [
		"https://images.unsplash.com/photo-1516642499105-492ff3ac521e?w=800",
		"https://images.unsplash.com/photo-1546869956-37a7f2d2bc45?w=800",
	],
	"Sweater": [
		"https://images.unsplash.com/photo-1512436991641-6745cdb1723f?w=800",
		"https://images.unsplash.com/photo-1548372293-0c8f6d5f3d7d?w=800",
	],
	"Dress": [
		"https://images.unsplash.com/photo-1445205170230-053b83016050?w=800",
		"https://images.unsplash.com/photo-1490481651871-ab68de25d43d?w=800",
	],
	"Scarf": [
		"https://images.unsplash.com/photo-1519689680058-324335c77eba?w=800",
		"https://images.unsplash.com/photo-1516826957135-700dedea698c?w=800",
	],
	"Boots": [
		"https://images.unsplash.com/photo-1525966222134-fcfa99b8ae77?w=800",
		"https://images.unsplash.com/photo-1542314831-068cd1dbfeeb?w=800",
	],
}


def _item_image_url(item_id: int | None, category: str | None) -> str:
	fallback = "https://images.unsplash.com/photo-1490111718993-d98654ce6cf7?w=800"
	if not category:
		return fallback
	urls = CATEGORY_IMAGE_URLS.get(category)
	if not urls:
		return fallback
	if item_id is None:
		return urls[0]
	# Deterministic pick based on item_id
	return urls[item_id % len(urls)]


def render_item_gallery(items_df: pd.DataFrame, title: str, cols_per_row: int = 4, image_height: int = 180) -> None:
	if items_df.empty:
		st.info("No items to display.")
		return
	st.markdown(f"**{title}**")
	# Ensure uniqueness per Item_ID
	unique_items = items_df.drop_duplicates("Item_ID")
	rows = (len(unique_items) + cols_per_row - 1) // cols_per_row
	iterator = unique_items.to_dict("records")
	idx = 0
	for _ in range(rows):
		cols = st.columns(cols_per_row)
		for col in cols:
			if idx >= len(iterator):
				break
			item = iterator[idx]
			idx += 1
			with col:
				st.markdown('<div class="card">', unsafe_allow_html=True)
				st.image(
					_item_image_url(item.get("Item_ID"), item.get("Category")),
					use_container_width=True,
					caption=None,
				)
				st.markdown(
					f"<div class='card-title'>Item {item.get('Item_ID')}</div>",
					unsafe_allow_html=True,
				)
				st.markdown(
					f"<p class='card-sub'>{item.get('Category', 'N/A')}</p>",
					unsafe_allow_html=True,
				)
				price = item.get("Price", "—")
				st.markdown(f"<span class='price-badge'>$ {price}</span>", unsafe_allow_html=True)
				st.markdown('</div>', unsafe_allow_html=True)


@st.cache_data
def _generate_data_cached(num_users: int, num_items: int, seed: Optional[int]) -> pd.DataFrame:
	return generate_synthetic_fashion_data(num_users=num_users, num_items=num_items, seed=seed)


with st.sidebar:
	st.header("Data Source")
	mode = st.radio("Choose data source", ["Generate sample", "Upload CSV"], index=0)

	if mode == "Generate sample":
		num_users = st.slider("Users", min_value=20, max_value=1000, value=100, step=10)
		num_items = st.slider("Items", min_value=20, max_value=1000, value=50, step=10)
		seed = st.number_input("Random seed", min_value=0, value=42)
		data_df = _generate_data_cached(num_users, num_items, int(seed))
		csv_buf = io.StringIO()
		data_df.to_csv(csv_buf, index=False)
		st.download_button("Download generated CSV", csv_buf.getvalue(), file_name="fashion_data.csv", mime="text/csv")
	else:
		upload = st.file_uploader("Upload CSV with columns: User_ID, Item_ID, Rating, Category, Price", type=["csv"]) 
		if upload is not None:
			data_df = load_csv(upload)
		else:
			st.stop()

	st.header("Layout")
	cols_per_row = st.slider("Cards per row", 2, 6, 4)
	img_height = st.slider("Image height (px)", 120, 320, 180)


st.subheader("Preview")
st.dataframe(data_df.head(20), use_container_width=True)


# Build matrices
pivot = build_pivot_table(data_df)
user_sim_df = compute_user_similarity(pivot)
item_sim_df = compute_item_similarity(pivot)


tab1, tab2, tab3 = st.tabs(["User-based", "Item-based", "Visualize"])

with tab1:
	st.markdown("**User-based Recommendations**")
	user_ids = sorted(pivot.index.to_list())
	if not user_ids:
		st.info("No users available.")
	else:
		user_id = st.selectbox("Select User_ID", user_ids)
		n_recs = st.slider("Number of recommendations", 1, 20, 5)
		threshold = st.slider("Min rating threshold (neighbors)", 1.0, 5.0, 4.0, 0.5)
		recs = recommend_for_user(user_id, pivot, user_sim_df, n_recommendations=n_recs, min_rating_threshold=threshold)
		if recs:
			st.success(f"Recommended Item_IDs for user {user_id}: {recs}")
			# Show gallery with images
			meta_cols = [c for c in ["Item_ID", "Category", "Price"] if c in data_df.columns]
			if meta_cols:
				render_item_gallery(
					data_df[data_df["Item_ID"].isin(recs)][meta_cols],
					title="Recommended Items",
					cols_per_row=cols_per_row,
					image_height=img_height,
				)
				show_table = st.checkbox("Show table view", value=False, key="user_tab_table")
				if show_table:
					st.dataframe(
						data_df[data_df["Item_ID"].isin(recs)][meta_cols].drop_duplicates("Item_ID"),
						use_container_width=True,
					)
		else:
			st.warning("No recommendations found (user may have rated most items or dataset is sparse).")

with tab2:
	st.markdown("**Item-based Similar Items**")
	item_ids = sorted(pivot.columns.to_list())
	if not item_ids:
		st.info("No items available.")
	else:
		item_id = st.selectbox("Select Item_ID", item_ids)
		n_recs = st.slider("Number of similar items", 1, 20, 5, key="similar_count")
		recs = recommend_similar_items(item_id, item_sim_df, n_recommendations=n_recs)
		if recs:
			st.success(f"Items similar to {item_id}: {recs}")
			meta_cols = [c for c in ["Item_ID", "Category", "Price"] if c in data_df.columns]
			if meta_cols:
				render_item_gallery(
					data_df[data_df["Item_ID"].isin([item_id] + recs)][meta_cols],
					title="Similar Items",
					cols_per_row=cols_per_row,
					image_height=img_height,
				)
				show_table = st.checkbox("Show table view", value=False, key="item_tab_table")
				if show_table:
					st.dataframe(
						data_df[data_df["Item_ID"].isin([item_id] + recs)][meta_cols].drop_duplicates("Item_ID"),
						use_container_width=True,
					)
		else:
			st.warning("No similar items found.")

with tab3:
	st.markdown("**Similarity Heatmaps**")
	col1, col2 = st.columns(2)
	with col1:
		st.caption("User-User Similarity")
		fig_u, ax_u = plt.subplots(figsize=(6, 4))
		sns.heatmap(user_sim_df, cmap="viridis", ax=ax_u)
		ax_u.set_xlabel("User_ID")
		ax_u.set_ylabel("User_ID")
		st.pyplot(fig_u, clear_figure=True)
	with col2:
		st.caption("Item-Item Similarity")
		fig_i, ax_i = plt.subplots(figsize=(6, 4))
		sns.heatmap(item_sim_df, cmap="magma", ax=ax_i)
		ax_i.set_xlabel("Item_ID")
		ax_i.set_ylabel("Item_ID")
		st.pyplot(fig_i, clear_figure=True)


