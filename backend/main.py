import sys
import os
import logging
sys.path.insert(0, os.path.dirname(__file__))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import httpx

from database import init_db, get_conn
from recommenders import trending, previously_liked, matrix_factorization, sequential, genre_match
import seed as seed_module

app = FastAPI(title="Book Recommender")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OPEN_LIBRARY_SEARCH = "https://openlibrary.org/search.json"
COVER_URL = "https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
PAGE_SIZE = 30


@app.on_event("startup")
def startup():
    init_db()
    if os.environ.get("POPULATE_DB") == "1":
        seed_module.seed()
        logging.info("POPULATE_DB=1: seed data loaded.")
    matrix_factorization.load()
    sequential.load()


# ---------------------------------------------------------------------------
# Books
# ---------------------------------------------------------------------------

def _ol_to_book(doc: dict) -> dict:
    cover_id = (doc.get("cover_i") or (doc.get("cover_edition_key") and None))
    cover_url = COVER_URL.format(cover_id=cover_id) if cover_id else ""
    first_publish = doc.get("first_publish_year")
    author_list = doc.get("author_name") or []
    subjects = doc.get("subject") or []
    genre_tags = ", ".join(subjects[:5])
    book_id = doc.get("key", "").replace("/works/", "")
    return {
        "book_id": book_id,
        "title": doc.get("title", "Unknown"),
        "author": ", ".join(author_list[:2]),
        "year": first_publish,
        "genres": genre_tags,
        "description": doc.get("first_sentence", {}).get("value", "") if isinstance(doc.get("first_sentence"), dict) else "",
        "cover_url": cover_url,
    }


def _cache_books(books: list[dict]):
    with get_conn() as conn:
        conn.executemany("""
            INSERT OR IGNORE INTO books (book_id, title, author, year, genres, description, cover_url)
            VALUES (:book_id, :title, :author, :year, :genres, :description, :cover_url)
        """, books)


@app.get("/books")
async def search_books(query: str = "fiction", page: int = 1):
    offset = (page - 1) * PAGE_SIZE
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(OPEN_LIBRARY_SEARCH, params={
            "q": query,
            "limit": PAGE_SIZE,
            "offset": offset,
            "fields": "key,title,author_name,first_publish_year,cover_i,subject,first_sentence",
        })
    if resp.status_code != 200:
        raise HTTPException(502, "Open Library API error")

    data = resp.json()
    books = [_ol_to_book(doc) for doc in data.get("docs", []) if doc.get("key")]
    _cache_books(books)
    return {"books": books, "total": data.get("numFound", 0), "page": page}


# ---------------------------------------------------------------------------
# Interactions
# ---------------------------------------------------------------------------

ACTION_WEIGHTS = {"reading_list": 2, "info": 0.1, "hide": -1}


class InteractionIn(BaseModel):
    username: str
    book_id: str
    action: str  # reading_list | info | hide


@app.post("/interactions")
def log_interaction(body: InteractionIn):
    if body.action not in ACTION_WEIGHTS:
        raise HTTPException(400, f"action must be one of {list(ACTION_WEIGHTS)}")
    weight = ACTION_WEIGHTS[body.action]
    with get_conn() as conn:
        conn.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (body.username,))
        conn.execute(
            "INSERT INTO interactions (username, book_id, action, weight) VALUES (?, ?, ?, ?)",
            (body.username, body.book_id, body.action, weight),
        )
    logging.info("interaction | user=%-20s book=%-30s action=%-15s weight=%+d",
                 body.username, body.book_id, body.action, weight)
    return {"status": "ok", "weight": weight}


@app.get("/reading-list")
def reading_list(username: str):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT b.* FROM interactions i
            JOIN books b ON b.book_id = i.book_id
            WHERE i.username = ? AND i.action = 'reading_list'
            GROUP BY b.book_id
            ORDER BY MAX(i.timestamp) DESC
        """, (username,)).fetchall()
    return {"books": [dict(r) for r in rows]}


# ---------------------------------------------------------------------------
# Recommendations
# ---------------------------------------------------------------------------

MODELS = {
    "trending": {
        "label": "Trending",
        "description": "Books popular across all users right now.",
    },
    "previously_liked": {
        "label": "Previously Liked",
        "description": "Books similar to ones you've added or explored.",
    },
    "matrix_factorization": {
        "label": "Matrix Factorization (SVD)",
        "description": "Collaborative filtering — users like you also enjoyed these.",
    },
    "sequential": {
        "label": "Sequential (GRU)",
        "description": "Deep learning model that learns from the order of your interactions.",
    },
    "genre_match": {
        "label": "Genre Match",
        "description": "Books that share genres with your reading history.",
    },
    "keyword_match": {
        "label": "Keyword Match",
        "description": "Books matched by genre (2x) and keywords from title, author, and description.",
    },
}


@app.get("/models")
def list_models():
    return {"models": [{"id": k, **v} for k, v in MODELS.items()]}


@app.get("/recommendations")
def get_recommendations(username: str, model: str = "trending", n: int = 30):
    if model == "trending":
        books = trending.recommend(username, n)
    elif model == "previously_liked":
        books = previously_liked.recommend(username, n)
    elif model == "matrix_factorization":
        books = matrix_factorization.recommend(username, n)
    elif model == "sequential":
        books = sequential.recommend(username, n)
    elif model == "genre_match":
        books = genre_match.recommend_genre(username, n)
    elif model == "keyword_match":
        books = genre_match.recommend_keyword(username, n)
    else:
        raise HTTPException(400, "Unknown model")
    return {"books": books, "model": model}


@app.get("/explain")
def explain(username: str, book_id: str, model: str = "trending"):
    if model == "trending":
        text = trending.explain(book_id)
    elif model == "previously_liked":
        text = previously_liked.explain(username, book_id)
    elif model == "matrix_factorization":
        text = matrix_factorization.explain(username, book_id)
    elif model == "sequential":
        text = sequential.explain(username, book_id)
    elif model == "genre_match":
        text = genre_match.explain_genre(username, book_id)
    elif model == "keyword_match":
        text = genre_match.explain_keyword(username, book_id)
    else:
        raise HTTPException(400, "Unknown model")
    return {"explanation": text}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@app.post("/train")
def train(model: str = Query(..., description="mf or gru")):
    if model == "mf":
        result = matrix_factorization.train()
    elif model == "gru":
        result = sequential.train()
    else:
        raise HTTPException(400, "model must be 'mf' or 'gru'")
    return {"result": result}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")

@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))
