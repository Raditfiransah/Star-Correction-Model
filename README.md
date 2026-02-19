# ⭐ Star-Correction: Multi-Task BERT for Sentiment & Star Rating

Multi-task learning pipeline that simultaneously predicts **sentiment** (Positive / Neutral / Negative) and **corrected star rating** (1–5) from Google Maps review texts of tourist attractions in the Malang region, utilizing a shared BERT encoder with two independent classification heads dedicated to each task.

---

## Table of Contents

- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
  - [Data Pipeline](#1-data-pipeline)
  - [Model Architecture](#2-model-architecture)
  - [Training Loop](#3-training-loop)
  - [Hyperparameter Tuning](#4-hyperparameter-tuning-optuna)
  - [Evaluation](#5-evaluation)
  - [Inference API](#6-inference-api-fastapi)
  - [Dashboard](#7-dashboard-streamlit)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Input Review Text                     │
│           "Tempat wisata yang sangat bagus!"             │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │     Tokenizer       │
            │  (IndoBERT BPE)     │
            │  max_len = 128      │
            └─────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │                       │
          │   IndoBERT Encoder    │
          │   (Shared Backbone)   │
          │   768-dim hidden      │
          │                       │
          └───────────┬───────────┘
                      │
              ┌───────▼───────┐
              │    Dropout    │
              │    (0.3)      │
              └───────┬───────┘
                      │
           ┌──────────┴──────────┐
           │                     │
           ▼                     ▼
  ┌─────────────────┐  ┌─────────────────┐
  │ Sentiment Head  │  │   Star Head     │
  │  Linear(768→3)  │  │  Linear(768→5)  │
  │                 │  │                 │
  │ Negative        │  │ ⭐ 1            │
  │ Neutral         │  │ ⭐⭐ 2          │
  │ Positive        │  │ ⭐⭐⭐ 3        │
  │                 │  │ ⭐⭐⭐⭐ 4      │
  │                 │  │ ⭐⭐⭐⭐⭐ 5    │
  └─────────────────┘  └─────────────────┘
```

**Kenapa Multi-Task?** Review "Pelayanan biasa aja" mungkin diberi bintang 5 oleh user (star yang salah), tapi sentiment-nya Neutral. Dengan multi-task learning, model bisa **mengoreksi star rating** berdasarkan pemahaman konteks dan sentimen secara bersamaan — kedua task saling membantu memperkuat representasi shared encoder.

---

## Project Structure

```
Star-Correction/
├── src/                          # Training pipeline
│   ├── config.py                 # Dataclass konfigurasi terpusat
│   ├── dataset.py                # Data loading, splitting, DataLoader
│   ├── model.py                  # MultiTaskBERT architecture
│   ├── train.py                  # Training loop + early stopping
│   ├── evaluate.py               # Metrics + confusion matrix
│   ├── optuna_tuning.py          # Hyperparameter tuning
│   ├── utils.py                  # Seed & device helpers
│   └── main.py                   # CLI entrypoint
│
├── inference/                    # Serving layer
│   ├── model_serve.py            # FastAPI REST API
│   └── app.py                    # Streamlit dashboard
│
├── data/processed/               # Input dataset (CSV)
├── outputs/                      # Trained model & artifacts
│   └── best_model.pt             # Saved model weights
│
├── Dockerfile                    # Multi-stage Docker build
├── docker-compose.yml            # API + Dashboard services
├── requirements.txt              # Python dependencies
└── README.md                     # ← You are here
```

---

## How It Works

### 1. Data Pipeline

**File:** `src/dataset.py`

```
CSV (text, sentiment_label, corrected_star)
         │
         ▼
  ┌──────────────┐
  │ Clean & Map  │  Drop NaN, map "Positive"→2, star 5.0→index 4
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Stratified  │  80% train / 10% val / 10% test
  │    Split     │  stratify on sentiment × star combo
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  Class Wt    │  sklearn compute_class_weight("balanced")
  │  Compute     │  → handle imbalanced data (Pos:16910, Neg:2467, Neu:2216)
  └──────┬───────┘
         │
         ▼
  ┌──────────────┐
  │  DataLoader  │  Tokenize via IndoBERT tokenizer
  │    Build     │  (max_len=128, batch_size=16)
  └──────────────┘
```

**Label Mapping:**

| Sentiment | Index | Star Rating  | Index |
| --------- | ----- | ------------ | ----- |
| Negative  | 0     | ⭐ 1         | 0     |
| Neutral   | 1     | ⭐⭐ 2       | 1     |
| Positive  | 2     | ⭐⭐⭐ 3     | 2     |
|           |       | ⭐⭐⭐⭐ 4   | 3     |
|           |       | ⭐⭐⭐⭐⭐ 5 | 4     |

### 2. Model Architecture

**File:** `src/model.py`

```python
class MultiTaskBERT(nn.Module):
    def __init__(self, model_name, num_sentiment_classes=3, num_star_classes=5, dropout=0.3):
        self.bert = BertModel.from_pretrained(model_name)       # Shared encoder
        self.dropout = nn.Dropout(dropout)                      # Regularization
        self.sentiment_head = nn.Linear(768, num_sentiment_classes)  # Task 1
        self.star_head = nn.Linear(768, num_star_classes)            # Task 2

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled = self.dropout(outputs.pooler_output)  # [CLS] token
        return self.sentiment_head(pooled), self.star_head(pooled)
```

- **Base model:** [`indobenchmark/indobert-base-p1`](https://huggingface.co/indobenchmark/indobert-base-p1) — pre-trained BERT khusus Bahasa Indonesia
- **Shared encoder** menghasilkan representasi 768-dim dari token `[CLS]`
- **Dua head independen** masing-masing memprediksi task berbeda dari representasi yang sama

### 3. Training Loop

**File:** `src/train.py`

```
For each epoch:
  ┌─ Forward pass ──────────────────────────────────────┐
  │  sentiment_logits, star_logits = model(ids, mask)   │
  │                                                     │
  │  loss_sent = CrossEntropy(sentiment_logits, labels)  │
  │  loss_star = CrossEntropy(star_logits, labels)       │
  │                                                     │
  │  total_loss = α × loss_sent + β × loss_star         │
  │               (α=0.5)          (β=0.5)              │
  └─────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─ Backward + Optimize ──────────────────────────────┐
  │  loss.backward()                                    │
  │  clip_grad_norm_(max_grad_norm=1.0)                │
  │  optimizer.step()  (AdamW, weight_decay=0.01)       │
  │  scheduler.step()  (linear warmup)                  │
  └─────────────────────────────────────────────────────┘
                         │
                         ▼
  ┌─ Early Stopping ───────────────────────────────────┐
  │  if val_loss < best → save model                    │
  │  else → patience_counter++                          │
  │  if patience_counter >= 3 → STOP                    │
  └─────────────────────────────────────────────────────┘
```

**Key features:**

- **Weighted CrossEntropyLoss** — balanced class weights mengatasi data imbalance
- **AdamW optimizer** — with decoupled weight decay
- **Linear warmup scheduler** — warmup 10% dari total steps
- **Gradient clipping** — max_grad_norm=1.0 untuk stabilitas
- **Early stopping** — patience=3 berdasarkan validation loss

### 4. Hyperparameter Tuning (Optuna)

**File:** `src/optuna_tuning.py`

Optuna mencarikan kombinasi hyperparameter terbaik secara otomatis:

| Parameter       | Search Space          |
| --------------- | --------------------- |
| `learning_rate` | 1e-5 → 5e-5 (log)     |
| `batch_size`    | {8, 16, 32}           |
| `alpha`         | 0.2 → 0.8 (step 0.1)  |
| `weight_decay`  | 0.0 → 0.1 (step 0.01) |
| `dropout`       | 0.1 → 0.5 (step 0.1)  |

**Objective function:**

```
score = 0.5 × val_sentiment_weighted_f1 + 0.5 × val_star_weighted_f1
```

Semua trial di-track via **MLflow** (nested runs).

### 5. Evaluation

**File:** `src/evaluate.py`

Setelah training selesai, model dievaluasi dengan:

- **Accuracy** — persentase prediksi benar
- **Macro F1** — rata-rata F1 semua kelas (tanpa weighting)
- **Weighted F1** — F1 dengan weighting jumlah sampel per kelas
- **Classification Report** — precision, recall, F1 per kelas
- **Confusion Matrix** — heatmap disimpan sebagai PNG

### 6. Inference API (FastAPI)

**File:** `inference/model_serve.py`

```
 Client Request                    FastAPI Server
 ┌──────────┐     POST /predict    ┌──────────────────────┐
 │  {"text": │ ──────────────────▶ │  1. Tokenize text    │
 │   "..."}  │                     │  2. Forward pass     │
 └──────────┘                     │  3. Softmax probs    │
                                   │  4. Return JSON      │
 ┌──────────┐                     └──────────────────────┘
 │ Response  │ ◀──────────────────
 │ sentiment │
 │ stars     │
 │ probs     │
 └──────────┘
```

**Lifecycle:**

1. **Startup** — Load tokenizer + model weights ke memory (sekali saja)
2. **Request** — Tokenize → forward pass → softmax → JSON response
3. **Shutdown** — Bersihkan memory

### 7. Dashboard (Streamlit)

**File:** `inference/app.py`

Dashboard interaktif dengan layout dua kolom:

| Kiri (Input)           | Kanan (Output)                     |
| ---------------------- | ---------------------------------- |
| Text area untuk review | Sentiment + confidence             |
| Slider bintang (1–5)   | Probability bars per kelas         |
| Tombol Predict         | Predicted star (★★★★☆)             |
|                        | Perbandingan star asli vs prediksi |

Dashboard memanggil FastAPI via HTTP (`/predict` endpoint).

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA GPU (opsional, tapi sangat direkomendasikan untuk training)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** Untuk environment lokal, install torch sesuai GPU driver kamu.
> Lihat: https://pytorch.org/get-started/locally/

### 2. Train Model

```bash
# Full training run
python -m src.main --mode train

# Smoke test (1 epoch, 200 sampel)
python -m src.main --mode train --epochs 1 --max_len 32 --sample_size 200

# Hyperparameter tuning (20 trials)
python -m src.main --mode tune --n_trials 20
```

### 3. Monitor Training

```bash
mlflow ui --port 5000
# → http://localhost:5000
```

### 4. Serve Model

```bash
# Start API
uvicorn inference.model_serve:app --host 0.0.0.0 --port 8000

# Start Dashboard
streamlit run inference/app.py --server.port 8501
```

### 5. Test API

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Tempat wisata yang sangat bagus dan menyenangkan!"}'
```

**Response:**

```json
{
  "text": "Tempat wisata yang sangat bagus dan menyenangkan!",
  "sentiment": "Positive",
  "sentiment_confidence": 0.9999,
  "sentiment_probabilities": {
    "Negative": 0.0001,
    "Neutral": 0.0,
    "Positive": 0.9999
  },
  "predicted_star": 5,
  "star_confidence": 0.9758,
  "star_probabilities": {
    "1": 0.0001,
    "2": 0.0001,
    "3": 0.0001,
    "4": 0.0238,
    "5": 0.9758
  }
}
```

---

## Docker Deployment

### Build & Run

```bash
docker compose up --build
```

### Services

| Service       | Port | URL                        |
| ------------- | ---- | -------------------------- |
| **API**       | 8000 | http://localhost:8000/docs |
| **Dashboard** | 8501 | http://localhost:8501      |

### Docker Architecture

```
docker-compose.yml
├── api (FastAPI)
│   ├── Build: multi-stage Dockerfile
│   ├── torch CPU-only (~700MB vs ~5GB)
│   ├── Volume: ./outputs → /app/outputs (model weights)
│   ├── Healthcheck: GET /health every 30s
│   └── Port: 8000
│
└── dashboard (Streamlit)
    ├── Depends on: api (healthy)
    ├── ENV: API_URL=http://api:8000
    └── Port: 8501
```

Model weights (`outputs/best_model.pt`) di-mount sebagai volume read-only — tidak masuk ke Docker image.

---

## API Reference

### `GET /health`

Cek apakah model sudah loaded.

```json
{ "status": "ok", "model_loaded": true, "device": "cpu" }
```

### `POST /predict`

Prediksi single text.

**Request:**

```json
{ "text": "Review text here" }
```

**Response:**

```json
{
  "text": "Review text here",
  "sentiment": "Positive",
  "sentiment_confidence": 0.95,
  "sentiment_probabilities": {
    "Negative": 0.02,
    "Neutral": 0.03,
    "Positive": 0.95
  },
  "predicted_star": 5,
  "star_confidence": 0.87,
  "star_probabilities": {
    "1": 0.01,
    "2": 0.02,
    "3": 0.03,
    "4": 0.07,
    "5": 0.87
  }
}
```

### `POST /predict/batch`

Prediksi batch (max 64 texts).

**Request:**

```json
{
  "texts": ["Review 1", "Review 2", "Review 3"]
}
```

**Response:** Array of `PredictResponse` objects.

---

## Configuration

Semua konfigurasi terpusat di `src/config.py` sebagai Python dataclass:

| Parameter       | Default                          | Description                 |
| --------------- | -------------------------------- | --------------------------- |
| `model_name`    | `indobenchmark/indobert-base-p1` | Pre-trained BERT model      |
| `max_len`       | 128                              | Max token sequence length   |
| `epochs`        | 10                               | Maximum training epochs     |
| `batch_size`    | 16                               | Batch size                  |
| `learning_rate` | 2e-5                             | AdamW learning rate         |
| `weight_decay`  | 0.01                             | L2 regularization           |
| `dropout`       | 0.3                              | Dropout rate                |
| `alpha`         | 0.5                              | Sentiment loss weight       |
| `beta`          | 0.5                              | Star loss weight            |
| `patience`      | 3                                | Early stopping patience     |
| `warmup_ratio`  | 0.1                              | LR warmup proportion        |
| `max_grad_norm` | 1.0                              | Gradient clipping threshold |
| `seed`          | 42                               | Random seed                 |

Override via CLI:

```bash
python -m src.main --mode train --learning_rate 3e-5 --batch_size 32 --epochs 20
```

---

## Tech Stack

| Component         | Technology                                              |
| ----------------- | ------------------------------------------------------- |
| **Base Model**    | IndoBERT (`indobenchmark/indobert-base-p1`)             |
| **Framework**     | PyTorch + Hugging Face Transformers                     |
| **API Server**    | FastAPI + Uvicorn                                       |
| **Dashboard**     | Streamlit                                               |
| **Experiment**    | MLflow                                                  |
| **HP Tuning**     | Optuna                                                  |
| **Container**     | Docker + Docker Compose                                 |
| **Data**          | pandas + scikit-learn (stratified split, class weights) |
| **Visualization** | matplotlib + seaborn (confusion matrices)               |

---

## License

This project is for educational and research purposes.
