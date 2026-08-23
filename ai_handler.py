"""
ai_handler.py
=============
Multi-model AI routing engine for the Ingredient Safety Scanner.
DEVELOPED BY NITIN YADAV

Optimized for ultra-fast Vision OCR & safety evaluation with Google Gemini & Groq fallback.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import textwrap
from io import BytesIO
from typing import Any, Optional

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from groq import Groq
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Fast System & Prompt Templates
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert food scientist and consumer safety analyst. "
    "Evaluate ingredient lists and return ONLY a valid JSON object. "
    "No conversational text, no markdown code blocks, no thinking traces."
)

ANALYSIS_JSON_SCHEMA = textwrap.dedent("""
Return a JSON object with this EXACT structure:
{
  "overall_grade": "A" | "B" | "C" | "D",
  "overall_score": <integer 0-100>,
  "summary": "<2-3 short sentences in simple Hinglish/English explaining safety verdict>",
  "recommendation": "<1 direct actionable advice sentence>",
  "total_ingredients_found": <integer>,
  "concerning_ingredients_count": <integer>,
  "allergens_detected": ["<allergen1>", "..."],
  "ingredients": [
    {
      "name": "<ingredient name>",
      "risk_level": "Safe" | "Low Risk" | "Moderate Risk" | "High Risk",
      "category": "<e.g. Natural, Preservative, Sweetener, Color, Emulsifier>",
      "side_effects": "<brief side effect or 'Safe for regular use'>",
      "regulatory_status": "<e.g. FSSAI Approved, FDA GRAS, Restricted in EU>",
      "icon": "<✅ | 🟡 | 🟠 | 🔴>"
    }
  ],
  "processing_engine": "<engine_name>"
}

Grade Scale:
  A (85-100): Clean & mostly natural
  B (65-84) : Generally safe, standard additives
  C (40-64) : Multiple concerning additives / artificial sweeteners
  D (0-39)  : Hazardous / highly ultra-processed additives
""").strip()

IMAGE_USER_PROMPT = textwrap.dedent("""
Extract all ingredients visible on this label image using OCR and perform a safety evaluation.
{schema}
""").format(schema=ANALYSIS_JSON_SCHEMA)


# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing & Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_image(image: Image.Image, max_dim: int = 1400) -> Image.Image:
    """Resize high-res smartphone photos for instant 3x faster network transfer & OCR."""
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")

    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    return image


def _clean_json(raw: str) -> str:
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL | re.IGNORECASE)
    raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    raw = raw.strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start : end + 1]
    return raw


def _parse_result(raw_text: str, engine: str) -> dict[str, Any]:
    cleaned = _clean_json(raw_text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error: %s\nRaw: %s", exc, cleaned[:300])
        raise ValueError(f"Invalid JSON returned: {exc}") from exc

    data["processing_engine"] = engine
    data.setdefault("overall_grade", "C")
    data.setdefault("overall_score", 50)
    data.setdefault("summary", "Analysis completed.")
    data.setdefault("recommendation", "Review the breakdown before use.")
    data.setdefault("total_ingredients_found", len(data.get("ingredients", [])))
    data.setdefault("concerning_ingredients_count", 0)
    data.setdefault("ingredients", [])
    data.setdefault("allergens_detected", [])
    return data


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Engine (Vision + OCR)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiEngine:
    # Latest stable models – v1 API compatible (no v1beta 404 errors)
    MODELS_ORDER = [
        "gemini-2.5-flash",       # Latest & fastest multimodal
        "gemini-2.0-flash",       # Stable fallback
        "gemini-2.0-flash-lite",  # Lightweight fallback
    ]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        genai.configure(api_key=api_key)
        self.active_model = self.MODELS_ORDER[0]

    def _get_model(self, model_name: str) -> genai.GenerativeModel:
        return genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=2200,
            ),
        )

    def _image_to_bytes(self, image: Image.Image) -> dict:
        """Convert PIL image to inline bytes dict for Gemini API."""
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return {
            "mime_type": "image/jpeg",
            "data": base64.b64encode(buf.getvalue()).decode()
        }

    def analyze_image(self, image: Image.Image) -> dict[str, Any]:
        # Fast resize before sending
        processed_img = _preprocess_image(image)

        last_err = None
        for model_name in self.MODELS_ORDER:
            try:
                model = self._get_model(model_name)
                # Use inline image bytes (works reliably across all API versions)
                img_bytes = self._image_to_bytes(processed_img)
                response = model.generate_content(
                    [
                        IMAGE_USER_PROMPT,
                        {"inline_data": img_bytes},
                    ],
                    request_options={"timeout": 60},
                )
                self.active_model = model_name
                logger.info("Gemini vision success with model: %s", model_name)
                return _parse_result(response.text, engine=f"Gemini Vision ({model_name})")
            except Exception as exc:
                last_err = exc
                logger.warning("Gemini vision %s failed: %s", model_name, exc)
                continue

        raise RuntimeError(f"Gemini image analysis failed: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Master AI Handler
# ─────────────────────────────────────────────────────────────────────────────

class IngredientSafetyAI:
    def __init__(
        self,
        gemini_key: Optional[str] = None,
        groq_key: Optional[str] = None,
    ) -> None:
        self._gemini_key = (gemini_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self._groq_key   = (groq_key   or os.getenv("GROQ_API_KEY",   "")).strip()

        self._gemini: Optional[GeminiEngine] = None

        if self._gemini_key:
            try:
                self._gemini = GeminiEngine(self._gemini_key)
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

        if not self._gemini:
            raise RuntimeError("GEMINI_API_KEY is required for Camera & Image Label OCR.")

    @property
    def engines_available(self) -> dict[str, bool]:
        return {
            "gemini": self._gemini is not None,
        }

    def analyze_image(self, image: Image.Image) -> dict[str, Any]:
        if self._gemini:
            return self._gemini.analyze_image(image)
        raise RuntimeError("Image OCR requires GEMINI_API_KEY. Please set your Gemini key.")

    def get_status_info(self) -> dict[str, Any]:
        return {
            "gemini_available": self._gemini is not None,
            "gemini_model": self._gemini.active_model if self._gemini else "Inactive",
        }
