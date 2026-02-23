"""
FastAPI server for the multi-task BERT model.

Serves two predictions per request:
  - Sentiment  (Negative / Neutral / Positive)
  - Star Rating (1–5)

Usage:
    uvicorn inference.model_serve:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoTokenizer

# ── Ensure project root is importable ─────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MultiTaskBERT

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL_NAME = "indobenchmark/indobert-base-p1"
MODEL_PATH = PROJECT_ROOT / "outputs" / "best_model.pt"
MAX_LEN = 128
NUM_SENTIMENT = 3
NUM_STAR = 5
DROPOUT = 0.3
STAR_EMBED_DIM = 16
FUSION_DIM = 256

SENTIMENT_LABELS = ["Negative", "Neutral", "Positive"]
STAR_LABELS = ["1", "2", "3", "4", "5"]

# ── Global state ──────────────────────────────────────────────────────────────

model: MultiTaskBERT | None = None
tokenizer: AutoTokenizer | None = None
device: torch.device | None = None


# ── Lifespan (load model on startup) ─────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model and tokenizer once at startup, release on shutdown."""
    global model, tokenizer, device

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔧 Device: {device}")

    # Load tokenizer
    print(f"📦 Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load model
    print(f"📦 Loading model weights: {MODEL_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Train the model first with: python -m src.main --mode train"
        )

    model = MultiTaskBERT(
        model_name=MODEL_NAME,
        num_sentiment_classes=NUM_SENTIMENT,
        num_star_classes=NUM_STAR,
        dropout=DROPOUT,
        star_embed_dim=STAR_EMBED_DIM,
        fusion_dim=FUSION_DIM,
    )
    state_dict = torch.load(MODEL_PATH, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ Model loaded and ready!")

    yield  # ── App is running ──

    # Cleanup
    del model, tokenizer
    torch.cuda.empty_cache()
    print("🛑 Model unloaded.")


# ── FastAPI App ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Star-Correction Multi-Task API",
    description=(
        "Multi-task BERT model for sentiment classification (3 classes) "
        "and corrected star rating prediction (1–5)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        examples=["Tempat wisata yang sangat bagus dan menyenangkan!"],
    )
    star: int = Field(
        ...,
        ge=1,
        le=5,
        examples=[5],
        description="Original star rating from user (1–5)",
    )


class PredictResponse(BaseModel):
    text: str
    original_star: int
    sentiment: str
    sentiment_confidence: float
    sentiment_probabilities: dict[str, float]
    predicted_star: int
    star_confidence: float
    star_probabilities: dict[str, float]


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        examples=[
            [
                "Tempat wisata yang sangat bagus!",
                "Pelayanannya buruk sekali, tidak akan kembali lagi.",
            ]
        ],
    )
    stars: list[int] = Field(
        ...,
        min_length=1,
        max_length=64,
        examples=[[5, 1]],
        description="Original star ratings (1–5), must match length of texts",
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ── Inference Helper ──────────────────────────────────────────────────────────


@torch.no_grad()
def predict_single(text: str, star: int) -> PredictResponse:
    """Run inference on a single text string with original star rating."""
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    star_input = torch.tensor([star - 1], dtype=torch.long).to(device)  # 1-5 → 0-4

    sentiment_logits, star_logits = model(input_ids, attention_mask, star_input)

    # Sentiment
    sent_probs = F.softmax(sentiment_logits, dim=1).squeeze(0).cpu().tolist()
    sent_idx = int(torch.argmax(sentiment_logits, dim=1).item())
    sent_label = SENTIMENT_LABELS[sent_idx]

    # Star
    star_probs = F.softmax(star_logits, dim=1).squeeze(0).cpu().tolist()
    star_idx = int(torch.argmax(star_logits, dim=1).item())
    star_label = star_idx + 1  # 0-indexed → 1-5

    return PredictResponse(
        text=text,
        original_star=star,
        sentiment=sent_label,
        sentiment_confidence=round(sent_probs[sent_idx], 4),
        sentiment_probabilities={
            label: round(prob, 4)
            for label, prob in zip(SENTIMENT_LABELS, sent_probs)
        },
        predicted_star=star_label,
        star_confidence=round(star_probs[star_idx], 4),
        star_probabilities={
            label: round(prob, 4)
            for label, prob in zip(STAR_LABELS, star_probs)
        },
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the model is loaded and ready to serve."""
    return HealthResponse(
        status="ok" if model is not None else "not_ready",
        model_loaded=model is not None,
        device=str(device) if device else "unknown",
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict(request: PredictRequest):
    """Predict sentiment and star rating for a single text."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    return predict_single(request.text, request.star)


@app.post(
    "/predict/batch",
    response_model=list[PredictResponse],
    tags=["Prediction"],
)
async def predict_batch(request: BatchPredictRequest):
    """Predict sentiment and star rating for multiple texts (max 64)."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    if len(request.texts) != len(request.stars):
        raise HTTPException(
            status_code=422,
            detail="texts and stars must have the same length.",
        )

    return [predict_single(t, s) for t, s in zip(request.texts, request.stars)]
