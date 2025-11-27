# Fashion Recommender (Streamlit)

## Setup

1. Create a virtual environment (optional but recommended)

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Run the app

```bash
streamlit run app.py
```

The app lets you generate a synthetic dataset or upload your own CSV with the columns: `User_ID, Item_ID, Rating, Category, Price`.

## Features
- User-based recommendations using cosine similarity
- Item-based similar item lookup
- Similarity heatmaps (user-user and item-item)
- Download generated CSV

## Notes
- Pivot is built as `User_ID` x `Item_ID` with mean `Rating`.
- Similarity is computed on the NA-filled (0) matrix.

