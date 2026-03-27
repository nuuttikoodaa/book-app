"""
Seed script: inserts debug books and interaction history for user1.
Run once from the backend directory:
    python seed.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from database import init_db, get_conn

BOOKS = [
    {
        "book_id": "OL895706W",
        "title": "Foundation",
        "author": "Isaac Asimov",
        "year": 1951,
        "genres": "Science Fiction, Classic, Space Opera",
        "description": "The first novel in Asimov's classic science-fiction masterpiece about the fall of a Galactic Empire.",
        "cover_url": "https://covers.openlibrary.org/b/id/8228691-M.jpg",
    },
    {
        "book_id": "OL275558W",
        "title": "Dracula",
        "author": "Bram Stoker",
        "year": 1897,
        "genres": "Gothic Fiction, Horror, Classic",
        "description": "The classic vampire novel following Count Dracula's attempt to move from Transylvania to England.",
        "cover_url": "https://covers.openlibrary.org/b/id/8231856-M.jpg",
    },
    {
        "book_id": "OL102749W",
        "title": "Dune",
        "author": "Frank Herbert",
        "year": 1965,
        "genres": "Science Fiction, Epic, Adventure",
        "description": "Set in the distant future, Dune tells the story of young Paul Atreides on the desert planet Arrakis.",
        "cover_url": "https://covers.openlibrary.org/b/id/8786071-M.jpg",
    },
    {
        "book_id": "OL102975W",
        "title": "The Mysterious Affair at Styles",
        "author": "Agatha Christie",
        "year": 1920,
        "genres": "Mystery, Detective Fiction, Classic",
        "description": "Hercule Poirot's first case — a poisoning murder at a country house estate.",
        "cover_url": "https://covers.openlibrary.org/b/id/8237265-M.jpg",
    },
]

# Interactions for user1: all added to reading list, plus info on two of them
INTERACTIONS = [
    ("user1", "OL895706W",  "reading_list", 2),
    ("user1", "OL275558W",  "reading_list", 2),
    ("user1", "OL102749W",  "reading_list", 2),
    ("user1", "OL102975W",  "reading_list", 2),
    ("user1", "OL895706W",  "info",         0.1),
    ("user1", "OL102749W",  "info",         0.1),
]

def seed():
    init_db()
    with get_conn() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO books
                (book_id, title, author, year, genres, description, cover_url)
            VALUES
                (:book_id, :title, :author, :year, :genres, :description, :cover_url)
        """, BOOKS)

        conn.execute("INSERT OR IGNORE INTO users (username) VALUES ('user1')")

        existing = conn.execute(
            "SELECT COUNT(*) FROM interactions WHERE username = 'user1'"
        ).fetchone()[0]
        if existing == 0:
            conn.executemany("""
                INSERT INTO interactions (username, book_id, action, weight)
                VALUES (?, ?, ?, ?)
            """, INTERACTIONS)

    print(f"Seeded {len(BOOKS)} books and {len(INTERACTIONS)} interactions for user1.")

if __name__ == "__main__":
    seed()
