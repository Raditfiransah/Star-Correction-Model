"""
Streamlit dashboard for the Star-Correction multi-task model.

Usage:
    streamlit run inference/app.py
"""

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Star-Correction Predictor",
    page_icon="⭐",
    layout="wide",
)

st.markdown(
    """
    <style>
    .stApp { max-width: 1100px; margin: 0 auto; }
    .result-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        border-radius: 16px;
        padding: 28px;
        margin-top: 8px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    .metric-label { color: #a0a0b8; font-size: 13px; margin-bottom: 2px; }
    .metric-value { font-size: 32px; font-weight: 700; margin-bottom: 12px; }
    .prob-bar-bg {
        background: rgba(255,255,255,0.08);
        border-radius: 6px;
        height: 10px;
        margin-bottom: 6px;
    }
    .prob-bar {
        height: 10px;
        border-radius: 6px;
    }
    .sentiment-Positive { color: #4ade80; }
    .sentiment-Neutral  { color: #facc15; }
    .sentiment-Negative { color: #f87171; }
    .star-display { font-size: 40px; letter-spacing: 4px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────

st.title("⭐ Star-Correction Predictor")
st.caption("Multi-task BERT — Sentiment & Corrected Star Rating")

# ── Check API health ─────────────────────────────────────────────────────────

try:
    health = requests.get(f"{API_URL}/health", timeout=3).json()
    if health["model_loaded"]:
        st.success(f"Model ready — device: `{health['device']}`", icon="✅")
    else:
        st.warning("Model is loading, please wait…", icon="⏳")
except Exception:
    st.error(
        "⚠️ Cannot connect to API. Pastikan server berjalan:\n\n"
        "```\nuvicorn inference.model_serve:app --host 0.0.0.0 --port 8000\n```"
    )
    st.stop()

st.divider()

# ── Layout: Input (kiri) | Output (kanan) ────────────────────────────────────

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📝 Input")

    text = st.text_area(
        "Review Text",
        height=180,
        placeholder="Contoh: Tempat wisata yang sangat bagus dan menyenangkan!",
    )

    star_input = st.slider("⭐ Star Rating (asli dari user)", 1, 5, 5)

    predict_btn = st.button("🔍 Predict", type="primary", use_container_width=True)

with col_output:
    st.subheader("📊 Output")

    if predict_btn:
        if not text.strip():
            st.warning("Masukkan teks review terlebih dahulu.")
        else:
            with st.spinner("Predicting…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/predict",
                        json={"text": text, "star": star_input},
                        timeout=30,
                    ).json()
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()

            # ── Sentiment ─────────────────────────────────────────────
            sent = resp["sentiment"]
            sent_conf = resp["sentiment_confidence"]
            color_map = {
                "Positive": "#4ade80",
                "Neutral": "#facc15",
                "Negative": "#f87171",
            }
            sent_color = color_map.get(sent, "#ffffff")

            st.markdown("##### Sentiment")
            st.markdown(
                f'<span style="color:{sent_color}; font-size:28px; font-weight:700;">'
                f"{sent}</span> "
                f'<span style="color:#a0a0b8; font-size:16px;">'
                f"({sent_conf:.1%} confidence)</span>",
                unsafe_allow_html=True,
            )

            # Probability bars
            for label, prob in resp["sentiment_probabilities"].items():
                c = color_map.get(label, "#888")
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                    f'<span style="width:70px;color:#a0a0b8;font-size:13px;">{label}</span>'
                    f'<div style="flex:1;background:rgba(255,255,255,0.08);border-radius:6px;height:10px;">'
                    f'<div style="width:{prob*100:.1f}%;background:{c};height:10px;border-radius:6px;"></div>'
                    f"</div>"
                    f'<span style="width:50px;text-align:right;color:#a0a0b8;font-size:13px;">{prob:.1%}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Star Rating ───────────────────────────────────────────
            pred_star = resp["predicted_star"]
            star_conf = resp["star_confidence"]

            st.markdown("##### Corrected Star Rating")

            stars_display = "★" * pred_star + "☆" * (5 - pred_star)
            st.markdown(
                f'<span style="font-size:36px;letter-spacing:4px;color:#facc15;">'
                f"{stars_display}</span> "
                f'<span style="color:#a0a0b8;font-size:16px;">'
                f"({star_conf:.1%} confidence)</span>",
                unsafe_allow_html=True,
            )

            # Probability bars for stars
            for label, prob in resp["star_probabilities"].items():
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">'
                    f'<span style="width:70px;color:#a0a0b8;font-size:13px;">⭐ {label}</span>'
                    f'<div style="flex:1;background:rgba(255,255,255,0.08);border-radius:6px;height:10px;">'
                    f'<div style="width:{prob*100:.1f}%;background:#facc15;height:10px;border-radius:6px;"></div>'
                    f"</div>"
                    f'<span style="width:50px;text-align:right;color:#a0a0b8;font-size:13px;">{prob:.1%}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Comparison ────────────────────────────────────────────
            st.markdown("##### Perbandingan Star")
            c1, c2, c3 = st.columns(3)
            c1.metric("⭐ Star Asli", f"{star_input}")
            c2.metric("🔮 Star Prediksi", f"{pred_star}")
            diff = pred_star - star_input
            c3.metric("📊 Selisih", f"{diff:+d}")

    else:
        st.info("Masukkan review text dan klik **Predict** untuk melihat hasil.", icon="👈")
