"""
Previously Liked model: TF-IDF cosine similarity between user's positively
interacted books and all catalog books (genre, author, description).
"""
from database import get_conn
import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def _get_all_books(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM books").fetchall()
    return [dict(r) for r in rows]


def _book_text(book: dict) -> str:
    parts = [
        book.get("title") or "",
        book.get("author") or "",
        book.get("genres") or "",
        book.get("description") or "",
        str(book.get("year") or ""),
    ]
    return " ".join(p for p in parts if p)


def recommend(username: str, n: int = 30) -> list[dict]:
    if not SKLEARN_AVAILABLE:
        return _fallback(n)

    with get_conn() as conn:
        liked_rows = conn.execute("""
            SELECT DISTINCT book_id FROM interactions
            WHERE username = ? AND weight > 0
        """, (username,)).fetchall()
        liked_ids = {r["book_id"] for r in liked_rows}
        hidden_ids = {r["book_id"] for r in conn.execute("""
            SELECT DISTINCT book_id FROM interactions
            WHERE username = ? AND weight < 0
        """, (username,)).fetchall()}
        all_books = _get_all_books(conn)

    if not liked_ids or not all_books:
        return _fallback(n)

    texts = [_book_text(b) for b in all_books]
    vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf = vectorizer.fit_transform(texts)

    liked_indices = [i for i, b in enumerate(all_books) if b["book_id"] in liked_ids]
    liked_vectors = tfidf[liked_indices]
    avg_liked = np.asarray(liked_vectors.mean(axis=0))

    sims = cosine_similarity(avg_liked, tfidf)[0]

    scored = sorted(
        [(all_books[i], float(sims[i])) for i in range(len(all_books))
         if all_books[i]["book_id"] not in liked_ids
         and all_books[i]["book_id"] not in hidden_ids],
        key=lambda x: x[1],
        reverse=True,
    )
    result = []
    for book, score in scored[:n]:
        book["score"] = score
        result.append(book)
    return result


def explain(username: str, book_id: str) -> str:
    if not SKLEARN_AVAILABLE:
        return "Recommended based on your reading history."

    with get_conn() as conn:
        liked_rows = conn.execute("""
            SELECT b.title FROM interactions i
            JOIN books b ON b.book_id = i.book_id
            WHERE i.username = ? AND i.weight > 0
            ORDER BY i.timestamp DESC LIMIT 3
        """, (username,)).fetchall()
        target = conn.execute("SELECT * FROM books WHERE book_id = ?", (book_id,)).fetchone()

    liked_titles = [r["title"] for r in liked_rows]
    if liked_titles and target:
        return (
            f"Recommended because it shares genre, author, or themes with books you liked: "
            + ", ".join(f'"{t}"' for t in liked_titles) + "."
        )
    return "Recommended based on your reading history."


def _fallback(n: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM books LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows]
