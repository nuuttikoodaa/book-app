"""
Matrix Factorization model using TruncatedSVD from scikit-learn.
No scikit-surprise dependency — uses only sklearn + numpy + pandas.
Requires manual training via POST /train?model=mf.
Model is persisted to disk and loaded at startup.
"""
import os
import pickle
import numpy as np
import pandas as pd
from database import get_conn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mf_model.pkl")

_model = None       # dict with SVD artifacts
_user_index = {}    # username -> row index in reconstructed matrix
_book_index = {}    # book_id -> col index
_reconstructed = None  # numpy array (n_users x n_books)


def load():
    global _model, _user_index, _book_index, _reconstructed
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        _model = data
        _user_index = data["user_index"]
        _book_index = data["book_index"]
        _reconstructed = data["reconstructed"]


def train() -> str:
    from sklearn.decomposition import TruncatedSVD

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT username, book_id, SUM(weight) AS total_weight
            FROM interactions
            GROUP BY username, book_id
        """).fetchall()

    if len(rows) < 5:
        return "Not enough interactions to train (need at least 5)."

    df = pd.DataFrame([dict(r) for r in rows])
    df["total_weight"] = df["total_weight"].clip(-2, 4).astype(float)

    user_item = df.pivot_table(
        index="username", columns="book_id",
        values="total_weight", fill_value=0,
    )

    n_components = min(50, user_item.shape[0] - 1, user_item.shape[1] - 1)
    if n_components < 1:
        return "Not enough unique users or books to factorize."

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item.values)
    reconstructed = user_factors @ svd.components_

    user_index = {u: i for i, u in enumerate(user_item.index)}
    book_index = {b: i for i, b in enumerate(user_item.columns)}

    global _model, _user_index, _book_index, _reconstructed
    _user_index = user_index
    _book_index = book_index
    _reconstructed = reconstructed
    _model = {
        "user_index": user_index,
        "book_index": book_index,
        "reconstructed": reconstructed,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(_model, f)

    return (
        f"Matrix factorization model trained on {len(df)} interaction records "
        f"({len(user_index)} users × {len(book_index)} books, {n_components} components)."
    )


def recommend(username: str, n: int = 30) -> list[dict]:
    if _reconstructed is None or username not in _user_index:
        return _fallback(n)

    with get_conn() as conn:
        interacted = {r["book_id"] for r in conn.execute(
            "SELECT DISTINCT book_id FROM interactions WHERE username = ?", (username,)
        ).fetchall()}
        hidden_ids = {r["book_id"] for r in conn.execute(
            "SELECT DISTINCT book_id FROM interactions WHERE username = ? AND weight < 0", (username,)
        ).fetchall()}

    user_row = _reconstructed[_user_index[username]]
    scored = sorted(
        [(bid, float(user_row[idx]))
         for bid, idx in _book_index.items()
         if bid not in interacted and bid not in hidden_ids],
        key=lambda x: x[1],
        reverse=True,
    )[:n]

    if not scored:
        return _fallback(n)

    top_ids = [bid for bid, _ in scored]
    with get_conn() as conn:
        placeholders = ",".join("?" * len(top_ids))
        rows = conn.execute(
            f"SELECT * FROM books WHERE book_id IN ({placeholders})", top_ids
        ).fetchall()

    book_map = {r["book_id"]: dict(r) for r in rows}
    result = []
    for bid, score in scored:
        if bid in book_map:
            book_map[bid]["score"] = score
            result.append(book_map[bid])
    return result


def explain(username: str, book_id: str) -> str:
    if _reconstructed is None:
        return "Matrix factorization model not yet trained. Use the Train Model button to train it."
    if username not in _user_index or book_id not in _book_index:
        return "Not enough data to explain this recommendation for your account."
    score = float(_reconstructed[_user_index[username]][_book_index[book_id]])
    return (
        f"Matrix factorization (SVD) predicted a score of {score:.2f} for you on this book, "
        "based on patterns across all users with similar taste."
    )


def _fallback(n: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM books LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows]
