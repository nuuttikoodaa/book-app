FROM python:3.13-slim

WORKDIR /app

# Install build deps needed by some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY backend/ ./backend/
# Copy frontend so FastAPI can serve it
COPY frontend/ ./frontend/

# SQLite DB and trained model files will live in /app/data (mounted volume)
ENV DB_PATH=/app/data/books.db

WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
