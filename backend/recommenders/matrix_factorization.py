"""
Matrix Factorization model using SVD via scikit-surprise.
Requires manual training via POST /train?model=mf.
Model is persisted to disk and loaded at startup.
"""
import os
import pickle
from database import get_conn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "mf_model.pkl")
_model = None
_book_id_list = []  # ordered list of all known book_ids

try:
    from surprise import SVD, Dataset, Reader
    from surprise import accuracy
    import pandas as pd
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False


def load():
    global _model, _book_id_list
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            data = pickle.load(f)
            _model = data["model"]
            _book_id_list = data["book_id_list"]


def train() -> str:
    if not SURPRISE_AVAILABLE:
        return "scikit-surprise not installed. Run: pip install scikit-surprise"

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT username, book_id, SUM(weight) AS total_weight
            FROM interactions
            GROUP BY username, book_id
        """).fetchall()

    if len(rows) < 5:
        return "Not enough interactions to train (need at least 5)."

    import pandas as pd
    df = pd.DataFrame([dict(r) for r in rows])
    # Clamp to a rating-like scale: weight can be -1..+4 (e.g. add+info twice)
    df["rating"] = df["total_weight"].clip(-2, 4).astype(float)

    reader = Reader(rating_scale=(-2, 4))
    dataset = Dataset.load_from_df(df[["username", "book_id", "rating"]], reader)
    trainset = dataset.build_full_trainset()

    algo = SVD(n_factors=50, n_epochs=30, random_state=42)
    algo.fit(trainset)

    book_id_list = list({r["book_id"] for r in rows})

    global _model, _book_id_list
    _model = algo
    _book_id_list = book_id_list

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": algo, "book_id_list": book_id_list}, f)

    return f"Matrix factorization model trained on {len(df)} interaction records."


def recommend(username: str, n: int = 30) -> list[dict]:
    if _model is None:
        return _fallback(n)

    with get_conn() as conn:
        interacted = {r["book_id"] for r in conn.execute(
            "SELECT DISTINCT book_id FROM interactions WHERE username = ?", (username,)
        ).fetchall()}
        hidden_ids = {r["book_id"] for r in conn.execute(
            "SELECT DISTINCT book_id FROM interactions WHERE username = ? AND weight < 0", (username,)
        ).fetchall()}

    candidates = [bid for bid in _book_id_list if bid not in interacted and bid not in hidden_ids]
    scored = [(bid, _model.predict(username, bid).est) for bid in candidates]
    scored.sort(key=lambda x: x[1], reverse=True)
    top_ids = [bid for bid, _ in scored[:n]]

    with get_conn() as conn:
        placeholders = ",".join("?" * len(top_ids))
        rows = conn.execute(
            f"SELECT * FROM books WHERE book_id IN ({placeholders})", top_ids
        ).fetchall()

    book_map = {r["book_id"]: dict(r) for r in rows}
    result = []
    for bid, score in scored[:n]:
        if bid in book_map:
            book_map[bid]["score"] = score
            result.append(book_map[bid])
    return result


def explain(username: str, book_id: str) -> str:
    if _model is None:
        return "Matrix factorization model not yet trained. Use POST /train?model=mf to train it."
    pred = _model.predict(username, book_id)
    return (
        f"Matrix factorization (SVD) predicted a score of {pred.est:.2f} for you on this book, "
        "based on patterns across all users with similar taste."
    )


def _fallback(n: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM books LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows]
