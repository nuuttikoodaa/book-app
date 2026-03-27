"""
Sequential GRU model: treats user interaction history as a token sequence.
Predicts the next book a user will positively interact with.
Requires manual training via POST /train?model=gru.
"""
import os
import pickle
from database import get_conn

MODEL_PATH = os.path.join(os.path.dirname(__file__), "gru_model.pkl")
VOCAB_PATH = os.path.join(os.path.dirname(__file__), "gru_vocab.pkl")

_model = None
_vocab = {}       # book_id -> int index
_inv_vocab = {}   # int index -> book_id
_device = None

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class GRURecommender(torch.nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size + 1)

    def forward(self, x):
        e = self.embed(x)
        out, _ = self.gru(e)
        return self.fc(out[:, -1, :])


def load():
    global _model, _vocab, _inv_vocab, _device
    if not TORCH_AVAILABLE:
        return
    if os.path.exists(MODEL_PATH) and os.path.exists(VOCAB_PATH):
        import torch
        with open(VOCAB_PATH, "rb") as f:
            vocab_data = pickle.load(f)
        _vocab = vocab_data["vocab"]
        _inv_vocab = vocab_data["inv_vocab"]
        _device = torch.device("cpu")
        model = GRURecommender(vocab_size=len(_vocab))
        model.load_state_dict(torch.load(MODEL_PATH, map_location=_device))
        model.eval()
        _model = model


def _build_sequences(min_len: int = 3):
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT username, book_id FROM interactions
            WHERE weight > 0
            ORDER BY username, timestamp
        """).fetchall()

    from collections import defaultdict
    user_seqs = defaultdict(list)
    for r in rows:
        user_seqs[r["username"]].append(r["book_id"])

    return {u: seq for u, seq in user_seqs.items() if len(seq) >= min_len}


def train() -> str:
    if not TORCH_AVAILABLE:
        return "PyTorch not installed. Run: pip install torch"

    import torch
    import torch.nn as nn

    sequences = _build_sequences(min_len=3)
    if not sequences:
        return "Not enough interaction sequences to train (need users with 3+ positive interactions)."

    # Build vocabulary
    all_ids = sorted({bid for seq in sequences.values() for bid in seq})
    vocab = {bid: i + 1 for i, bid in enumerate(all_ids)}  # 0 = padding
    inv_vocab = {i: bid for bid, i in vocab.items()}

    # Build (input_seq, target) pairs with a sliding window
    SEQ_LEN = 5
    X, Y = [], []
    for seq in sequences.values():
        indices = [vocab[bid] for bid in seq]
        for i in range(1, len(indices)):
            inp = indices[max(0, i - SEQ_LEN):i]
            inp = [0] * (SEQ_LEN - len(inp)) + inp  # left-pad
            X.append(inp)
            Y.append(indices[i])

    if len(X) < 5:
        return "Not enough training pairs. Add more interactions first."

    device = torch.device("cpu")
    X_t = torch.tensor(X, dtype=torch.long, device=device)
    Y_t = torch.tensor(Y, dtype=torch.long, device=device)

    model = GRURecommender(vocab_size=len(vocab))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    EPOCHS = 30
    BATCH = 32
    model.train()
    for epoch in range(EPOCHS):
        perm = torch.randperm(len(X_t))
        total_loss = 0.0
        for start in range(0, len(X_t), BATCH):
            idx = perm[start:start + BATCH]
            xb, yb = X_t[idx], Y_t[idx]
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()

    global _model, _vocab, _inv_vocab, _device
    _model = model
    _vocab = vocab
    _inv_vocab = inv_vocab
    _device = device

    torch.save(model.state_dict(), MODEL_PATH)
    with open(VOCAB_PATH, "wb") as f:
        pickle.dump({"vocab": vocab, "inv_vocab": inv_vocab}, f)

    return f"GRU model trained on {len(X)} sequence pairs from {len(sequences)} users."


def recommend(username: str, n: int = 30) -> list[dict]:
    if _model is None or not TORCH_AVAILABLE:
        return _fallback(n)

    import torch

    with get_conn() as conn:
        rows = conn.execute("""
            SELECT book_id FROM interactions
            WHERE username = ? AND weight > 0
            ORDER BY timestamp
        """, (username,)).fetchall()
        hidden_ids = {r["book_id"] for r in conn.execute(
            "SELECT DISTINCT book_id FROM interactions WHERE username = ? AND weight < 0", (username,)
        ).fetchall()}

    history = [r["book_id"] for r in rows if r["book_id"] in _vocab]
    if not history:
        return _fallback(n)

    SEQ_LEN = 5
    indices = [_vocab[bid] for bid in history[-SEQ_LEN:]]
    indices = [0] * (SEQ_LEN - len(indices)) + indices
    inp = torch.tensor([indices], dtype=torch.long, device=_device)

    with torch.no_grad():
        logits = _model(inp)[0]

    scores = logits.cpu().numpy()
    interacted_indices = {_vocab[bid] for bid in history if bid in _vocab}
    hidden_indices = {_vocab.get(bid, -1) for bid in hidden_ids}

    ranked = sorted(
        [(idx, float(scores[idx])) for idx in range(1, len(scores))
         if idx not in interacted_indices and idx not in hidden_indices],
        key=lambda x: x[1],
        reverse=True,
    )[:n]

    top_ids = [_inv_vocab[idx] for idx, _ in ranked if idx in _inv_vocab]
    if not top_ids:
        return _fallback(n)

    with get_conn() as conn:
        placeholders = ",".join("?" * len(top_ids))
        rows = conn.execute(
            f"SELECT * FROM books WHERE book_id IN ({placeholders})", top_ids
        ).fetchall()

    book_map = {r["book_id"]: dict(r) for r in rows}
    result = []
    for idx, score in ranked:
        bid = _inv_vocab.get(idx)
        if bid and bid in book_map:
            book_map[bid]["score"] = score
            result.append(book_map[bid])
    return result


def explain(username: str, book_id: str) -> str:
    if _model is None:
        return "Sequential GRU model not yet trained. Use POST /train?model=gru to train it."
    with get_conn() as conn:
        rows = conn.execute("""
            SELECT b.title FROM interactions i
            JOIN books b ON b.book_id = i.book_id
            WHERE i.username = ? AND i.weight > 0
            ORDER BY i.timestamp DESC LIMIT 3
        """, (username,)).fetchall()
    recent = [r["title"] for r in rows]
    if recent:
        return (
            "The GRU sequential model predicted this book based on your recent reading sequence: "
            + ", ".join(f'"{t}"' for t in recent) + "."
        )
    return "Recommended by the sequential model based on your interaction history."


def _fallback(n: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM books LIMIT ?", (n,)).fetchall()
    return [dict(r) for r in rows]
