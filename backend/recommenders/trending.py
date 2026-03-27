"""
Trending model: ranks books by total positive interaction weight across all users.
No personalization — good for cold-start.
"""
from database import get_conn


def recommend(username: str, n: int = 30) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT b.*, COALESCE(SUM(i.weight), 0) AS score
            FROM books b
            LEFT JOIN interactions i ON b.book_id = i.book_id AND i.weight > 0
            GROUP BY b.book_id
            ORDER BY score DESC, b.title
            LIMIT ?
        """, (n,)).fetchall()
    return [_row_to_dict(r) for r in rows]


def explain(book_id: str) -> str:
    with get_conn() as conn:
        row = conn.execute("""
            SELECT COALESCE(SUM(weight), 0) AS score,
                   COUNT(*) AS interactions
            FROM interactions
            WHERE book_id = ? AND weight > 0
        """, (book_id,)).fetchone()
    if row and row["interactions"] > 0:
        return f"This book has a trending score of {row['score']} from {row['interactions']} positive interactions across all users."
    return "This book is trending in the catalog."


def _row_to_dict(row) -> dict:
    return dict(row)
