"""
Genre Match and Keyword Match baseline recommenders.
Both are stateless — computed at request time, no training needed.
"""
from database import get_conn
import re

STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "is", "it", "its", "was", "are", "be",
    "as", "this", "that", "his", "her", "he", "she", "they", "we", "you",
    "i", "not", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "about", "into", "than", "then",
    "so", "if", "when", "who", "which", "also", "after", "before", "more",
}


def _parse_genres(genres_str: str) -> set[str]:
    if not genres_str:
        return set()
    return {g.strip().lower() for g in genres_str.split(",") if g.strip()}


def _tokenise(text: str) -> set[str]:
    tokens = re.split(r"[\s\-_/,\.;:!?\"\'()]+", (text or "").lower())
    return {t for t in tokens if len(t) > 2 and t not in STOP_WORDS}


def _get_liked_books(conn, username: str) -> list[dict]:
    rows = conn.execute("""
        SELECT DISTINCT b.* FROM interactions i
        JOIN books b ON b.book_id = i.book_id
        WHERE i.username = ? AND i.weight > 0
    """, (username,)).fetchall()
    return [dict(r) for r in rows]


def _get_excluded_ids(conn, username: str) -> set[str]:
    rows = conn.execute("""
        SELECT DISTINCT book_id FROM interactions WHERE username = ?
    """, (username,)).fetchall()
    return {r["book_id"] for r in rows}


def _get_all_books(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM books").fetchall()]


# ---------------------------------------------------------------------------
# Genre Match
# ---------------------------------------------------------------------------

def recommend_genre(username: str, n: int = 30) -> list[dict]:
    with get_conn() as conn:
        liked = _get_liked_books(conn, username)
        excluded = _get_excluded_ids(conn, username)
        all_books = _get_all_books(conn)

    if not liked:
        return all_books[:n]

    liked_genres = set()
    for b in liked:
        liked_genres |= _parse_genres(b.get("genres", ""))

    if not liked_genres:
        return [b for b in all_books if b["book_id"] not in excluded][:n]

    scored = []
    for book in all_books:
        if book["book_id"] in excluded:
            continue
        overlap = _parse_genres(book.get("genres", "")) & liked_genres
        if overlap:
            book["score"] = len(overlap)
            scored.append((book, len(overlap)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [b for b, _ in scored[:n]]


def explain_genre(username: str, book_id: str) -> str:
    with get_conn() as conn:
        liked = _get_liked_books(conn, username)
        target = conn.execute(
            "SELECT * FROM books WHERE book_id = ?", (book_id,)
        ).fetchone()

    if not target:
        return "No explanation available."

    liked_genres = set()
    for b in liked:
        liked_genres |= _parse_genres(b.get("genres", ""))

    matched = _parse_genres(dict(target).get("genres", "")) & liked_genres
    if matched:
        return (
            "Recommended because it matches genre(s) from your reading history: "
            + ", ".join(sorted(matched)) + "."
        )
    return "Recommended based on genre similarity."


# ---------------------------------------------------------------------------
# Keyword Match
# ---------------------------------------------------------------------------

def _book_keywords(book: dict) -> set[str]:
    text = " ".join(filter(None, [
        book.get("title", ""),
        book.get("author", ""),
        book.get("genres", ""),
        book.get("description", ""),
    ]))
    return _tokenise(text)


def recommend_keyword(username: str, n: int = 30) -> list[dict]:
    with get_conn() as conn:
        liked = _get_liked_books(conn, username)
        excluded = _get_excluded_ids(conn, username)
        all_books = _get_all_books(conn)

    if not liked:
        return all_books[:n]

    liked_keywords = set()
    liked_genres = set()
    for b in liked:
        liked_keywords |= _book_keywords(b)
        liked_genres |= _parse_genres(b.get("genres", ""))

    scored = []
    for book in all_books:
        if book["book_id"] in excluded:
            continue
        genre_overlap = len(_parse_genres(book.get("genres", "")) & liked_genres)
        keyword_overlap = len(_book_keywords(book) & liked_keywords)
        # Genre weighted 2x over keyword
        score = genre_overlap * 2 + keyword_overlap
        if score > 0:
            book["score"] = score
            scored.append((book, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [b for b, _ in scored[:n]]


def explain_keyword(username: str, book_id: str) -> str:
    with get_conn() as conn:
        liked = _get_liked_books(conn, username)
        target = conn.execute(
            "SELECT * FROM books WHERE book_id = ?", (book_id,)
        ).fetchone()

    if not target:
        return "No explanation available."

    liked_keywords = set()
    liked_genres = set()
    for b in liked:
        liked_keywords |= _book_keywords(b)
        liked_genres |= _parse_genres(b.get("genres", ""))

    target_dict = dict(target)
    matched_genres = _parse_genres(target_dict.get("genres", "")) & liked_genres
    matched_keywords = _book_keywords(target_dict) & liked_keywords

    parts = []
    if matched_genres:
        parts.append("genres: " + ", ".join(sorted(matched_genres)))
    if matched_keywords:
        top_kw = sorted(matched_keywords)[:8]
        parts.append("keywords: " + ", ".join(top_kw))

    if parts:
        return "Recommended based on matching " + "; and ".join(parts) + "."
    return "Recommended based on keyword similarity to your reading history."
