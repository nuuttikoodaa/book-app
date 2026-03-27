# Project Status

## Overview
Book recommendation PoC — single-page web app with a FastAPI backend, SQLite storage, and four recommendation models. Users browse books from Open Library, interact with them, and receive personalised recommendations.

## What's Built

### Backend (`backend/`)
- **FastAPI** app serving REST API + frontend static file
- **SQLite** schema: `users`, `books`, `interactions` tables
- **Open Library API** integration for book search and cover images
- **Interaction logging** with structured stdout logs (user, book, action, weight)
- **`POPULATE_DB=1` env var** triggers seed data on startup

#### API Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/books` | Search Open Library, cache results to SQLite |
| POST | `/interactions` | Log user action with weight |
| GET | `/reading-list` | Return user's saved books |
| GET | `/recommendations` | Ranked books from chosen model |
| GET | `/explain` | Why a book was recommended |
| GET | `/models` | List available models |
| POST | `/train` | Manually trigger model training |
| GET | `/` | Serve frontend |

#### Interaction Weights
| Action | Weight |
|---|---|
| Add to reading list | +2 |
| Info (why this?) | +0.1 |
| Hide | -1 |

### Recommendation Models (`backend/recommenders/`)
| Model | ID | Status | Notes |
|---|---|---|---|
| Trending | `trending` | Ready | SQL aggregate of positive weights across all users |
| Previously Liked | `previously_liked` | Ready | TF-IDF cosine similarity on genre/author/description |
| Matrix Factorization | `matrix_factorization` | Ready (needs training) | SVD via scikit-surprise; `POST /train?model=mf` |
| Sequential GRU | `sequential` | Ready (needs training) | PyTorch GRU on interaction sequences; `POST /train?model=gru` |

All models fall back to a plain catalog listing if not yet trained.

### Frontend (`frontend/index.html`)
- Single HTML file, vanilla JS + CSS Grid, dark theme
- Username input (persisted to `localStorage`), default: `user1`
- Model selector dropdown (live re-fetches recommendations on change)
- Book grid — ~30 books, paginated, with cover images
- Per-card actions: Add to Reading List / Hide / Why this?
- Collapsible reading list panel
- "Why this?" modal with per-model explanations
- **Hide reviewed** toggle — hides added/hidden books from grid, persisted

### Infrastructure
- **Dockerfile** — `python:3.13-slim`
- **docker-compose.yml** — single service, named volume `app-data` for DB + model persistence
- **Makefile** — two commands:
  - `make pull_and_start` — pull, build, seed, start
  - `make clear_database` — stop containers, delete volume

### Seed Data (`backend/seed.py`)
Pre-loaded books and `reading_list` + `info` interactions for `user1`:
- Foundation — Isaac Asimov
- Dracula — Bram Stoker
- Dune — Frank Herbert
- The Mysterious Affair at Styles — Agatha Christie

Idempotent — safe to run multiple times.

## Dependencies
```
fastapi==0.135.2
uvicorn==0.42.0
httpx==0.28.1
scikit-learn==1.8.0
pandas==3.0.1
numpy==2.4.3
# optional: scikit-surprise==1.1.4, torch==2.11.0
```

## How to Run
```bash
make pull_and_start      # build, seed, and start
# → http://localhost:8000
```

## Pending / Next Steps
- [ ] Train and validate Matrix Factorization model (needs ~10+ interactions across users)
- [ ] Train and validate Sequential GRU model (needs ~20+ interactions per user)
- [ ] Expand seed data with more users and interactions for meaningful ML training
- [ ] "Why this?" explanations for baseline models could be richer
- [ ] Consider adding a `/reset-interactions` endpoint for easier testing
