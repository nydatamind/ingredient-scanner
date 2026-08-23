"""
ai_handler.py
=============
Multi-model AI routing engine for the Ingredient Safety Scanner.
DEVELOPED BY NITIN YADAV

Pipeline:
  Step 1 — Gemini Vision (latest flash model): OCR the image → extract ingredients text
  Step 2 — Groq LLaMA (ultra-fast): Analyze text → structured JSON  (if Groq key available)
           Fallback: Gemini does both steps in one shot if no Groq key.
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

from google import genai
from google.genai import types as genai_types
from groq import Groq
from PIL import Image

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Language Configuration
# ─────────────────────────────────────────────────────────────────────────────

LANGUAGE_INSTRUCTIONS = {
    "English":  "Write the 'summary' and 'recommendation' fields in clear, simple English.",
    "Hindi":    "'summary' और 'recommendation' fields को सरल और स्पष्ट हिंदी में लिखें।",
    "Hinglish": "'summary' और 'recommendation' fields को Hinglish में लिखें (Hindi + English mix, जैसे: 'Is product mein...').",
}

# ─────────────────────────────────────────────────────────────────────────────
# Prompt Templates (language-aware)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = (
    "You are an expert food scientist and consumer safety analyst. "
    "Evaluate ingredient lists and return ONLY a valid JSON object. "
    "No conversational text, no markdown code blocks, no thinking traces."
)

ANALYSIS_JSON_SCHEMA = textwrap.dedent("""
Return a JSON object with this EXACT structure:
{
  "overall_grade": "A" | "B" | "C" | "D",
  "overall_score": <integer 0-100>,
  "summary": "<2-3 short sentences explaining safety verdict in the specified language>",
  "recommendation": "<1 direct actionable advice sentence in the specified language>",
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

# Gemini OCR prompt — just extracts ingredients text from image (fast, minimal output)
OCR_ONLY_PROMPT = (
    "OCR this product label image. Extract ONLY the ingredients list as plain text. "
    "Output just the raw ingredient names separated by commas. No extra explanation."
)

def _build_analysis_prompt(ingredients_text: str, language: str) -> str:
    lang_instr = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["English"])
    return textwrap.dedent(f"""
    Analyze the following product ingredients for safety. {lang_instr}

    Ingredients:
    {ingredients_text}

    {ANALYSIS_JSON_SCHEMA}
    """).strip()

def _build_vision_prompt(language: str) -> str:
    lang_instr = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["English"])
    return textwrap.dedent(f"""
    Extract all ingredients visible on this label image using OCR and perform a safety evaluation.
    {lang_instr}
    {ANALYSIS_JSON_SCHEMA}
    """).strip()


# ─────────────────────────────────────────────────────────────────────────────
# Image Preprocessing & Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_image(image: Image.Image, max_dim: int = 1400) -> Image.Image:
    """Resize high-res smartphone photos for 3x faster network transfer & OCR."""
    if image.mode in ("RGBA", "P"):
        image = image.convert("RGB")
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        image = image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
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


def _extract_text_from_response(response) -> str:
    """Safely extract text from a Gemini response object."""
    raw = response.text
    if raw:
        return raw
    # Manual fallback through candidates/parts
    parts = []
    for cand in (response.candidates or []):
        for part in (cand.content.parts or []):
            if hasattr(part, "text") and part.text:
                parts.append(part.text)
    return "".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Engine — Vision OCR (latest models, new google-genai SDK)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiEngine:
    # Verified available models — ordered fastest/latest first (Aug 2026)
    VISION_MODELS = [
        "gemini-3.7-flash",       # Latest & fastest — primary
        "gemini-3.6-flash",       # Very recent stable
        "gemini-3.5-flash",       # Proven stable fallback
        "gemini-2.5-flash",       # Reliable fallback
        "gemini-flash-latest",    # Always-latest alias (final safety net)
    ]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not set.")
        self._client = genai.Client(api_key=api_key)
        self.active_model = self.VISION_MODELS[0]

    def _image_to_bytes(self, image: Image.Image) -> bytes:
        buf = BytesIO()
        image.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _call(self, model: str, contents: list, system: str) -> str:
        """Call Gemini model and return text response."""
        response = self._client.models.generate_content(
            model=model,
            contents=contents,
            config=genai_types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.1,
                max_output_tokens=2500,
                automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
        text = _extract_text_from_response(response)
        if not text:
            raise ValueError("Empty response from model")
        return text

    def ocr_extract_ingredients(self, image: Image.Image) -> str:
        """Step 1: Fast OCR — extract ingredients text only (no JSON analysis)."""
        img_bytes = self._image_to_bytes(_preprocess_image(image))
        last_err = None
        for model in self.VISION_MODELS:
            try:
                text = self._call(
                    model=model,
                    contents=[
                        OCR_ONLY_PROMPT,
                        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    ],
                    system="You are an OCR engine. Extract only the ingredient list as plain text.",
                )
                self.active_model = model
                logger.info("Gemini OCR success: %s", model)
                return text
            except Exception as exc:
                last_err = exc
                logger.warning("Gemini OCR %s failed: %s", model, exc)
        raise RuntimeError(f"Gemini OCR failed: {last_err}")

    def analyze_full(self, image: Image.Image, language: str = "English") -> dict[str, Any]:
        """Single-step: Gemini does OCR + analysis together (used when no Groq key)."""
        img_bytes = self._image_to_bytes(_preprocess_image(image))
        prompt = _build_vision_prompt(language)
        last_err = None
        for model in self.VISION_MODELS:
            try:
                text = self._call(
                    model=model,
                    contents=[
                        prompt,
                        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    ],
                    system=SYSTEM_PROMPT_BASE,
                )
                self.active_model = model
                logger.info("Gemini full-analysis success: %s", model)
                return _parse_result(text, engine=f"Gemini Vision ({model})")
            except Exception as exc:
                last_err = exc
                logger.warning("Gemini full-analysis %s failed: %s", model, exc)
        raise RuntimeError(f"Gemini image analysis failed: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Groq Engine — Ultra-fast Text Analysis (LLaMA)
# ─────────────────────────────────────────────────────────────────────────────

class GroqEngine:
    # Active Groq models for structured JSON output — ordered best first
    TEXT_MODELS = [
        "openai/gpt-oss-120b",           # Highest quality structured reasoning
        "openai/gpt-oss-20b",            # Ultra-fast lightweight
        "groq/compound",                 # Built-in fast Groq compound engine
        "groq/compound-mini",            # Lightweight compound engine
    ]

    def __init__(self, api_key: str) -> None:
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set.")
        self._client = Groq(api_key=api_key)
        self.active_model = self.TEXT_MODELS[0]

    def analyze_ingredients(self, ingredients_text: str, language: str = "English") -> dict[str, Any]:
        """Analyze extracted ingredient text and return structured JSON — extremely fast."""
        prompt = _build_analysis_prompt(ingredients_text, language)
        last_err = None
        for model in self.TEXT_MODELS:
            try:
                resp = self._client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT_BASE},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=2500,
                )
                raw_text = resp.choices[0].message.content
                if not raw_text:
                    raise ValueError("Empty Groq response")
                self.active_model = model
                logger.info("Groq analysis success: %s", model)
                return _parse_result(raw_text, engine=f"Gemini OCR + Groq ({model})")
            except Exception as exc:
                last_err = exc
                logger.warning("Groq %s failed: %s", model, exc)
        raise RuntimeError(f"Groq analysis failed: {last_err}")


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
        self._groq:   Optional[GroqEngine]   = None

        if self._gemini_key:
            try:
                self._gemini = GeminiEngine(self._gemini_key)
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

        if self._groq_key:
            try:
                self._groq = GroqEngine(self._groq_key)
            except Exception as exc:
                logger.warning("Groq init failed: %s", exc)

        if not self._gemini:
            raise RuntimeError("GEMINI_API_KEY is required for image label OCR.")

    @property
    def engines_available(self) -> dict[str, bool]:
        return {
            "gemini": self._gemini is not None,
            "groq":   self._groq   is not None,
        }

    def analyze_image(self, image: Image.Image, language: str = "English", **kwargs: Any) -> dict[str, Any]:
        """
        Two-step pipeline when Groq is available (fastest):
          1. Gemini Vision: OCR → extract ingredients text
          2. Groq LLaMA:   text → structured JSON analysis

        Falls back to single-step Gemini if Groq is unavailable.
        """
        if self._groq:
            # ── Fast two-step pipeline ──
            logger.info("Using 2-step pipeline: Gemini OCR → Groq analysis")
            ingredients_text = self._gemini.ocr_extract_ingredients(image)
            logger.info("OCR extracted: %s…", ingredients_text[:100])
            return self._groq.analyze_ingredients(ingredients_text, language=language)
        else:
            # ── Single-step Gemini (still uses latest model) ──
            logger.info("Using single-step Gemini Vision analysis")
            return self._gemini.analyze_full(image, language=language)

    def get_status_info(self) -> dict[str, Any]:
        pipeline = "⚡ Gemini OCR + Groq Analysis" if self._groq else "Gemini Vision"
        gemini_model = self._gemini.active_model if self._gemini else "Inactive"
        groq_model   = self._groq.active_model   if self._groq   else "Not configured"
        return {
            "gemini_available": self._gemini is not None,
            "groq_available":   self._groq   is not None,
            "gemini_model":     gemini_model,
            "groq_model":       groq_model,
            "pipeline":         pipeline,
        }
