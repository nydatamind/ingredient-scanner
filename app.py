"""
app.py
======
Ingredient Safety Scanner – Camera & Image Vision Web Application.
DEVELOPED BY NITIN YADAV
"""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path
import time
import traceback
from typing import Any

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from PIL import Image

# ── Load environment keys ───────────────────────────────────────────────────
env_file = Path(__file__).parent / ".env"
env_example = Path(__file__).parent / ".env.example"

if env_file.exists():
    load_dotenv(dotenv_path=env_file, override=True)
elif env_example.exists():
    load_dotenv(dotenv_path=env_example, override=True)
else:
    load_dotenv()

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Ingredient Safety Scanner",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Declare Native Camera & Gallery Custom Component ─────────────────────────
_FRONTEND_DIR = Path(__file__).parent / "frontend"
_camera_gallery = components.declare_component(
    "camera_gallery_uploader",
    path=str(_FRONTEND_DIR),
)

# ─────────────────────────────────────────────────────────────────────────────
# Ultra-Modern Mobile Theme CSS
# ─────────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-main: #0a0d14;
    --bg-surface: #111622;
    --bg-card: #171d2b;
    --border: #232c3d;
    --border-glow: #3b82f6;
    --text-main: #f1f5f9;
    --text-sub: #94a3b8;
    --grade-a: #10b981;
    --grade-b: #34d399;
    --grade-c: #fbbf24;
    --grade-d: #ef4444;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
    background-color: var(--bg-main) !important;
    color: var(--text-main) !important;
}

.stApp {
    background: radial-gradient(circle at 50% 0%, #131a2a 0%, #0a0d14 80%) !important;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 1.2rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 0.8rem !important;
    padding-right: 0.8rem !important;
    max-width: 800px !important;
    margin: 0 auto !important;
}

/* ── Hero Box (Clean Header) ── */
.hero-box {
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.4rem 1rem;
    text-align: center;
    margin-bottom: 1.2rem;
    box-shadow: 0 8px 30px rgba(0,0,0,0.3);
}
.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    margin: 0 0 0.35rem 0;
    background: linear-gradient(135deg, #ffffff, #94a3b8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 0.9rem;
    color: var(--text-sub);
    margin: 0;
}

/* ── Grade Badge Card ── */
.grade-box {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 1.2rem;
    padding: 1.25rem;
    background: var(--bg-surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    margin-bottom: 0.85rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
}
.grade-circle {
    width: 80px;
    height: 80px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 2.6rem;
    font-weight: 900;
    flex-shrink: 0;
}
.grade-circle.A { background: rgba(16,185,129,0.15); border: 3px solid var(--grade-a); color: var(--grade-a); box-shadow: 0 0 20px rgba(16,185,129,0.3); }
.grade-circle.B { background: rgba(52,211,153,0.15); border: 3px solid var(--grade-b); color: var(--grade-b); box-shadow: 0 0 20px rgba(52,211,153,0.3); }
.grade-circle.C { background: rgba(251,191,36,0.15); border: 3px solid var(--grade-c); color: var(--grade-c); box-shadow: 0 0 20px rgba(251,191,36,0.3); }
.grade-circle.D { background: rgba(239,68,68,0.15); border: 3px solid var(--grade-d); color: var(--grade-d); box-shadow: 0 0 20px rgba(239,68,68,0.3); }

.grade-info { flex: 1; min-width: 140px; }
.grade-name { font-size: 1.2rem; font-weight: 800; }
.grade-score { font-size: 1.6rem; font-weight: 800; line-height: 1.1; margin: 0.15rem 0; }
.score-track { background: #1e2638; height: 8px; border-radius: 50px; overflow: hidden; margin-top: 0.4rem; }
.score-fill { height: 100%; border-radius: 50px; }

/* ── Mobile Metrics Grid ── */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 0.45rem;
    margin-bottom: 0.85rem;
}
@media (max-width: 580px) {
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
}
.stat-box {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 0.65rem 0.4rem;
    text-align: center;
}
.stat-val { font-size: 1.35rem; font-weight: 800; }
.stat-lbl { font-size: 0.68rem; color: var(--text-sub); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px; }

/* ── Summary & Recommendation ── */
.verdict-box {
    background: rgba(59,130,246,0.08);
    border: 1px solid rgba(59,130,246,0.25);
    border-radius: 14px;
    padding: 0.95rem;
    margin-bottom: 0.65rem;
    font-size: 0.88rem;
    line-height: 1.5;
}
.action-box {
    background: rgba(16,185,129,0.08);
    border: 1px solid rgba(16,185,129,0.25);
    border-radius: 14px;
    padding: 0.95rem;
    margin-bottom: 0.85rem;
    font-size: 0.88rem;
    line-height: 1.5;
}
.allergen-banner {
    background: rgba(239,68,68,0.1);
    border: 1px solid rgba(239,68,68,0.35);
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    margin-bottom: 0.85rem;
    font-size: 0.85rem;
    color: #fca5a5;
}

/* ── Ingredient Item Card ── */
.ing-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0.9rem;
    margin-bottom: 0.55rem;
}
.ing-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.35rem;
}
.ing-title { font-weight: 700; font-size: 0.92rem; color: #ffffff; }
.ing-cat { font-size: 0.75rem; color: var(--text-sub); }
.ing-details { font-size: 0.82rem; color: #cbd5e1; line-height: 1.4; margin-top: 0.3rem; }
.ing-status { font-size: 0.72rem; color: #64748b; margin-top: 0.25rem; font-family: monospace; }

/* ── Risk Chips ── */
.chip {
    padding: 0.2rem 0.55rem;
    border-radius: 50px;
    font-size: 0.72rem;
    font-weight: 700;
}
.chip-safe { background: rgba(16,185,129,0.18); color: #34d399; }
.chip-low { background: rgba(52,211,153,0.18); color: #6ee7b7; }
.chip-mod { background: rgba(251,191,36,0.18); color: #fde047; }
.chip-high { background: rgba(239,68,68,0.18); color: #f87171; }

/* ── Ultra-Modern Button Redesign ── */
div[data-testid="stButton"] button[kind="primary"],
.btn-primary-custom button {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 16px !important;
    font-weight: 800 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.3px !important;
    padding: 0.85rem !important;
    min-height: 52px !important;
    width: 100% !important;
    box-shadow: 0 4px 22px rgba(37,99,235,0.45) !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

div[data-testid="stButton"] button[kind="primary"]:hover,
.btn-primary-custom button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(124,58,237,0.6) !important;
    border-color: rgba(255,255,255,0.4) !important;
}

div[data-testid="stButton"] button[kind="secondary"],
.btn-secondary-custom button {
    background: rgba(23, 29, 43, 0.9) !important;
    color: #94a3b8 !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 0.85rem !important;
    min-height: 52px !important;
    width: 100% !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 2px 10px rgba(0,0,0,0.2) !important;
}

div[data-testid="stButton"] button[kind="secondary"]:hover,
.btn-secondary-custom button:hover {
    color: #ffffff !important;
    background: #1e2638 !important;
    border-color: #3b82f6 !important;
    transform: translateY(-1px) !important;
}

/* ── Uploader Dropzone Styling ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 2px dashed #2a3449 !important;
    border-radius: 18px !important;
    padding: 1rem !important;
    transition: all 0.25s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #3b82f6 !important;
    background: rgba(30,38,56,0.6) !important;
}

/* ── Developer Badge (Footer) ── */
.dev-banner {
    text-align: center;
    margin-top: 2.5rem;
    margin-bottom: 1rem;
}
.dev-badge {
    display: inline-block;
    background: linear-gradient(135deg, rgba(37,99,235,0.18), rgba(124,58,237,0.18));
    border: 1px solid rgba(124,58,237,0.45);
    color: #c4b5fd;
    font-size: 0.8rem;
    font-weight: 800;
    letter-spacing: 1.8px;
    padding: 0.45rem 1.4rem;
    border-radius: 50px;
    box-shadow: 0 0 18px rgba(124,58,237,0.25);
    text-transform: uppercase;
}
</style>
"""

# ─────────────────────────────────────────────────────────────────────────────
# Configurations
# ─────────────────────────────────────────────────────────────────────────────

GRADE_INFO = {
    "A": {"name": "Excellent – Safe", "color": "#10b981", "fill": "linear-gradient(90deg, #059669, #10b981)", "desc": "Clean, natural & safe ingredients"},
    "B": {"name": "Good – Generally Safe", "color": "#34d399", "fill": "linear-gradient(90deg, #047857, #34d399)", "desc": "Permitted additives in safe moderation"},
    "C": {"name": "Caution – Moderate Risk", "color": "#fbbf24", "fill": "linear-gradient(90deg, #d97706, #fbbf24)", "desc": "Contains artificial sweeteners or additives"},
    "D": {"name": "Hazardous – High Risk", "color": "#ef4444", "fill": "linear-gradient(90deg, #dc2626, #ef4444)", "desc": "Highly ultra-processed / harmful additives"},
}


def _get_key(name: str) -> str:
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, "").strip()


def get_ai_engine(custom_g: str = "", custom_q: str = ""):
    import importlib
    import ai_handler
    importlib.reload(ai_handler)
    from ai_handler import IngredientSafetyAI

    gk = custom_g or _get_key("GEMINI_API_KEY")
    qk = custom_q or _get_key("GROQ_API_KEY")
    sig = f"{gk}:{qk}:v3"  # v3 = reload & language-aware engine

    if "ai_engine" not in st.session_state or st.session_state.get("_sig") != sig:
        try:
            st.session_state["ai_engine"] = IngredientSafetyAI(gemini_key=gk, groq_key=qk)
            st.session_state["_sig"] = sig
            st.session_state.pop("ai_err", None)
        except Exception as exc:
            st.session_state["ai_err"] = str(exc)
            st.session_state.pop("ai_engine", None)
            return None

    return st.session_state.get("ai_engine")


# ─────────────────────────────────────────────────────────────────────────────
# Render Clean Results
# ─────────────────────────────────────────────────────────────────────────────

def render_results(data: dict) -> None:
    grade = data.get("overall_grade", "C").upper()
    score = data.get("overall_score", 50)
    info = GRADE_INFO.get(grade, GRADE_INFO["C"])

    total = data.get("total_ingredients_found", 0)
    concerning = data.get("concerning_ingredients_count", 0)
    allergens = data.get("allergens_detected", [])
    safe_count = max(0, total - concerning)

    # Grade Card
    st.markdown(f"""
    <div class="grade-box">
        <div class="grade-circle {grade}">{grade}</div>
        <div class="grade-info">
            <div class="grade-name" style="color:{info['color']}">{info['name']}</div>
            <div class="grade-score" style="color:{info['color']}">{score} <span style="font-size:0.85rem;color:#94a3b8">/ 100</span></div>
            <div style="font-size:0.78rem;color:#94a3b8">{info['desc']}</div>
            <div class="score-track">
                <div class="score-fill" style="width:{max(5, min(100, score))}%;background:{info['fill']}"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Metrics Grid
    st.markdown(f"""
    <div class="stats-grid">
        <div class="stat-box"><div class="stat-val" style="color:#60a5fa">{total}</div><div class="stat-lbl">Total</div></div>
        <div class="stat-box"><div class="stat-val" style="color:#34d399">{safe_count}</div><div class="stat-lbl">Safe</div></div>
        <div class="stat-box"><div class="stat-val" style="color:#f87171">{concerning}</div><div class="stat-lbl">Concerning</div></div>
        <div class="stat-box"><div class="stat-val" style="color:#fbbf24">{len(allergens)}</div><div class="stat-lbl">Allergens</div></div>
    </div>
    """, unsafe_allow_html=True)

    # Allergen Alert
    if allergens:
        st.markdown(f"""
        <div class="allergen-banner">
            ⚠️ <strong>Allergens Detected:</strong> {', '.join(allergens)}
        </div>
        """, unsafe_allow_html=True)

    # Summary
    summary = data.get("summary", "")
    rec = data.get("recommendation", "")
    if summary:
        st.markdown(f"""
        <div class="verdict-box">
            <strong style="color:#93c5fd">📋 Verdict:</strong> {summary}
        </div>
        """, unsafe_allow_html=True)

    if rec:
        st.markdown(f"""
        <div class="action-box">
            <strong style="color:#6ee7b7">💡 Advice:</strong> {rec}
        </div>
        """, unsafe_allow_html=True)

    # Ingredient Cards
    ingredients = data.get("ingredients", [])
    if ingredients:
        st.markdown("<h4 style='margin:1.2rem 0 0.6rem 0;font-size:1rem'>🧪 Ingredient Breakdown</h4>", unsafe_allow_html=True)

        for item in ingredients:
            name = item.get("name", "Unknown")
            cat = item.get("category", "General")
            risk = item.get("risk_level", "Safe")
            side_fx = item.get("side_effects", "Safe for regular consumption")
            reg = item.get("regulatory_status", "Approved")
            icon = item.get("icon", "✅")

            if "Safe" in risk:
                chip_class = "chip-safe"
            elif "Low" in risk:
                chip_class = "chip-low"
            elif "Moderate" in risk:
                chip_class = "chip-mod"
            else:
                chip_class = "chip-high"

            st.markdown(f"""
            <div class="ing-card">
                <div class="ing-header">
                    <div>
                        <span class="ing-title">{icon} {name}</span>
                        <span class="ing-cat"> · {cat}</span>
                    </div>
                    <span class="chip {chip_class}">{risk}</span>
                </div>
                <div class="ing-details">{side_fx}</div>
                <div class="ing-status">{reg}</div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[str, str]:
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:0.5rem 0 1rem;">
            <div style="font-size:1.5rem">🔬</div>
            <div style="font-weight:800;font-size:0.9rem;letter-spacing:1px">DEVELOPED BY NITIN YADAV</div>
            <div style="font-size:0.75rem;color:#94a3b8">Ingredient Scanner</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Engine Status
        ai = st.session_state.get("ai_engine")
        if ai:
            info = ai.get_status_info()
            g_icon = "🟢" if info["gemini_available"] else "🔴"
            q_icon = "⚡" if info.get("groq_available") else "⭕"
            st.markdown(f"{g_icon} **Gemini**: `{info['gemini_model']}`")
            st.markdown(f"{q_icon} **Groq**: `{info.get('groq_model', 'Not configured')}`")
            st.markdown(f"🔄 **Pipeline**: `{info.get('pipeline', 'Gemini Vision')}`")

        st.divider()

        # Optional Custom Keys
        with st.expander("🔑 Override API Keys", expanded=False):
            cg = st.text_input("Gemini API Key", type="password", value=st.session_state.get("c_gemini", _get_key("GEMINI_API_KEY")), key="c_gemini")
            cq = st.text_input("Groq API Key", type="password", value=st.session_state.get("c_groq", _get_key("GROQ_API_KEY")), key="c_groq")
            if st.button("Save Keys", use_container_width=True, type="secondary"):
                st.session_state.pop("ai_engine", None)
                st.session_state["_sig"] = None
                st.success("Keys updated!")
                st.rerun()

        st.markdown("""
        <div style="font-size:0.7rem;color:#64748b;text-align:center;margin-top:2rem">
            ⚡ <strong>DEVELOPED BY NITIN YADAV</strong>
        </div>
        """, unsafe_allow_html=True)

    return (
        st.session_state.get("c_gemini", "").strip(),
        st.session_state.get("c_groq", "").strip(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar
    custom_gemini, custom_groq = render_sidebar()

    # Clean Header
    st.markdown("""
    <div class="hero-box">
        <h1 class="hero-title">🔬 Ingredient Safety Scanner</h1>
        <p class="hero-subtitle">Capture with your Phone Camera or choose from Gallery for instant analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Language Selector ────────────────────────────────────────────────────
    st.markdown("""
    <div style="text-align:center;margin:-0.4rem 0 0.6rem;">
        <span style="font-size:0.78rem;color:#64748b;font-weight:600;letter-spacing:0.8px;text-transform:uppercase;">🌐 Response Language</span>
    </div>
    """, unsafe_allow_html=True)
    lang_col1, lang_col2, lang_col3 = st.columns(3)
    if "language" not in st.session_state:
        st.session_state["language"] = "Hinglish"
    with lang_col1:
        if st.button("🇬🇧 English",  key="lang_en",  use_container_width=True,
                     type="primary" if st.session_state["language"] == "English"  else "secondary"):
            st.session_state["language"] = "English";  st.rerun()
    with lang_col2:
        if st.button("🇮🇳 Hindi",    key="lang_hi",  use_container_width=True,
                     type="primary" if st.session_state["language"] == "Hindi"    else "secondary"):
            st.session_state["language"] = "Hindi";    st.rerun()
    with lang_col3:
        if st.button("🔀 Hinglish",  key="lang_hl",  use_container_width=True,
                     type="primary" if st.session_state["language"] == "Hinglish" else "secondary"):
            st.session_state["language"] = "Hinglish"; st.rerun()

    # Initialize AI
    ai = get_ai_engine(custom_gemini, custom_groq)

    if ai is None:
        st.markdown("""
        <div class="allergen-banner" style="color:#fde047;border-color:#f59e0b">
            ⚠️ <strong>API Key Required</strong><br>
            Please configure GEMINI_API_KEY in your environment or Streamlit Secrets.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Native Camera & Gallery Custom Component ──────────────────────────────
    # Camera button  → opens phone's original camera app (take photo → tick ✔)
    # Gallery button → opens device photo gallery / media picker directly
    uploader_val = _camera_gallery(key="native_cam_gal")

    # ── Process image received from the component ─────────────────────────────
    if uploader_val and isinstance(uploader_val, dict):
        ts = uploader_val.get("timestamp")
        if ts and ts != st.session_state.get("_last_img_ts"):
            st.session_state["_last_img_ts"] = ts
            raw_b64 = uploader_val.get("data", "")
            # Strip the data URL prefix (e.g. "data:image/jpeg;base64,")
            if "," in raw_b64:
                raw_b64 = raw_b64.split(",", 1)[1]
            try:
                img_bytes = base64.b64decode(raw_b64)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                st.session_state["active_image"] = img
                st.session_state["active_source"] = uploader_val.get("source", "camera")
                st.session_state.pop("scan_data", None)
                st.rerun()
            except Exception as e:
                st.error(f"Could not load image: {e}")

    # ── Active Image Preview & Scan ───────────────────────────────────────────
    active_img: Image.Image | None = st.session_state.get("active_image")

    if active_img is not None:
        source = st.session_state.get("active_source", "camera")
        source_label = "📷 Captured from Phone Camera" if source == "camera" else "🖼️ Selected from Gallery"

        st.markdown(f"""
        <div style="background:var(--bg-card);border:1px solid #3b82f6;border-radius:14px;
                    padding:0.65rem 1rem;margin:0.8rem 0 0.6rem;text-align:center;">
            <span style="font-size:0.82rem;font-weight:700;color:#60a5fa;letter-spacing:0.5px;">
                {source_label}
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.image(active_img, caption="Product Label – Ready for Analysis", use_container_width=True)

        col1, col2 = st.columns([3, 1])
        with col1:
            btn_scan = st.button(
                "🔍 Scan & Analyze Label Now",
                key="btn_scan_trigger",
                type="primary",
                use_container_width=True,
            )
        with col2:
            if st.button("🔄 Retake", key="btn_retake", type="secondary", use_container_width=True):
                st.session_state.pop("active_image", None)
                st.session_state.pop("scan_data", None)
                st.session_state.pop("_last_img_ts", None)
                st.session_state.pop("_fallback_name", None)
                st.rerun()

        if btn_scan:
            lang = st.session_state.get("language", "Hinglish")
            spinner_msg = {
                "English":  "🔬 Scanning label with Gemini Vision + Groq AI…",
                "Hindi":    "🔬 Gemini Vision + Groq AI से label scan हो रहा है…",
                "Hinglish": "🔬 Gemini Vision + Groq AI se label scan ho raha hai…",
            }.get(lang, "🔬 Scanning…")
            with st.spinner(spinner_msg):
                try:
                    res = ai.analyze_image(active_img, language=lang)
                    st.session_state["scan_data"] = res
                    st.rerun()
                except Exception as exc:
                    st.error(f"Scan failed: {exc}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())

    # ── Results Display ───────────────────────────────────────────────────────
    if "scan_data" in st.session_state:
        st.markdown("<hr style='border:none;border-top:1px solid #232c3d;margin:1.4rem 0;'>", unsafe_allow_html=True)
        render_results(st.session_state["scan_data"])

    # ── Bottom Developer Banner ───────────────────────────────────────────────
    st.markdown("""
    <div class="dev-banner">
        <span class="dev-badge">⚡ DEVELOPED BY NITIN YADAV</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
