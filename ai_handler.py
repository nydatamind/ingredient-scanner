"""
ai_handler.py
=============
Deterministic AI routing & ingredient safety engine for NutriScan · NY.
DEVELOPED BY NITIN YADAV

Features:
  1. Image fingerprinting (SHA-256) & normalization caching for consistent results.
  2. Canonical additive & ingredient database (E-numbers, INS codes, sweeteners, preservatives).
  3. Deterministic rule-based scoring & classification engine.
  4. Temperature = 0.0 deterministic OCR extraction.
  5. Multi-language output (English, Hindi, Hinglish) without altering safety scores.
  6. Unreadable label detection to prevent hallucinated ingredients.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import textwrap
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

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
# In-Memory Cache (Stable Analysis by Image & Ingredients Hash)
# ─────────────────────────────────────────────────────────────────────────────
_IMAGE_CACHE: Dict[str, Dict[str, Any]] = {}
_INGREDIENT_CACHE: Dict[str, Dict[str, Any]] = {}

# ─────────────────────────────────────────────────────────────────────────────
# Comprehensive Canonical Ingredient & Additive Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────
CANONICAL_DATABASE: List[Dict[str, Any]] = [
    # ── High Risk / Hazardous ──
    {
        "name": "Potassium Bromate (INS 924a)",
        "patterns": [r"potassium\s*bromate", r"ins\s*924\s*a?", r"e\s*924\s*a?"],
        "risk_level": "High Risk",
        "category": "Flour Treatment Agent",
        "side_effects": "Classified as potential carcinogen; banned in EU, India, UK, and multiple countries.",
        "regulatory_status": "Banned in Food",
        "icon": "🔴",
        "score_penalty": 35,
    },
    {
        "name": "Butylated Hydroxyanisole (BHA / INS 320)",
        "patterns": [r"\bbha\b", r"butylated\s*hydroxyanisole", r"ins\s*320", r"e\s*320"],
        "risk_level": "High Risk",
        "category": "Synthetic Antioxidant",
        "side_effects": "Potential endocrine disruptor; possible human carcinogen in long-term high doses.",
        "regulatory_status": "Restricted in EU / FDA Regulated",
        "icon": "🔴",
        "score_penalty": 25,
    },
    {
        "name": "Butylated Hydroxytoluene (BHT / INS 321)",
        "patterns": [r"\bbht\b", r"butylated\s*hydroxytoluene", r"ins\s*321", r"e\s*321"],
        "risk_level": "High Risk",
        "category": "Synthetic Antioxidant",
        "side_effects": "Linked to hormone disruption and potential organ toxicity with excessive intake.",
        "regulatory_status": "Strictly Limited Use",
        "icon": "🔴",
        "score_penalty": 25,
    },
    {
        "name": "Titanium Dioxide (INS 171)",
        "patterns": [r"titanium\s*dioxide", r"ins\s*171", r"e\s*171", r"ci\s*77891"],
        "risk_level": "High Risk",
        "category": "Artificial Color",
        "side_effects": "Banned in EU as food additive due to DNA damage (genotoxicity) concerns.",
        "regulatory_status": "Banned in EU Food",
        "icon": "🔴",
        "score_penalty": 30,
    },
    {
        "name": "Partially Hydrogenated Oil (Trans Fat)",
        "patterns": [r"partially\s*hydrogenated", r"trans\s*fat", r"vanaspati\s*ghee", r"hydrogenated\s*vegetable\s*oil"],
        "risk_level": "High Risk",
        "category": "Industrial Fat",
        "side_effects": "Elevates LDL bad cholesterol, lowers HDL, significantly increases heart disease risk.",
        "regulatory_status": "Prohibited / Heavily Restricted by WHO & FDA",
        "icon": "🔴",
        "score_penalty": 30,
    },
    {
        "name": "Azodicarbonamide (INS 927a)",
        "patterns": [r"azodicarbonamide", r"ins\s*927\s*a?", r"e\s*927\s*a?"],
        "risk_level": "High Risk",
        "category": "Dough Conditioner",
        "side_effects": "Respiratory sensitivity and asthma risk; breakdown products include semicarbazide.",
        "regulatory_status": "Banned in EU & Australia",
        "icon": "🔴",
        "score_penalty": 28,
    },
    {
        "name": "Red 40 / Allura Red (INS 129)",
        "patterns": [r"allura\s*red", r"red\s*40", r"ins\s*129", r"e\s*129", r"fd&c\s*red\s*no\.?\s*40"],
        "risk_level": "Moderate Risk",
        "category": "Artificial Color",
        "side_effects": "Synthetic azo dye; linked to hyperactivity in children and mild allergic sensitivities.",
        "regulatory_status": "Warning Label Required in EU",
        "icon": "🟠",
        "score_penalty": 15,
    },
    {
        "name": "Tartrazine / Yellow 5 (INS 102)",
        "patterns": [r"tartrazine", r"yellow\s*5", r"ins\s*102", r"e\s*102", r"fd&c\s*yellow\s*no\.?\s*5"],
        "risk_level": "Moderate Risk",
        "category": "Artificial Color",
        "side_effects": "May trigger urticaria (hives), asthma, and hyperactive behavior in sensitive children.",
        "regulatory_status": "Warning Label Required in EU",
        "icon": "🟠",
        "score_penalty": 15,
    },
    {
        "name": "Sunset Yellow FCF / Yellow 6 (INS 110)",
        "patterns": [r"sunset\s*yellow", r"yellow\s*6", r"ins\s*110", r"e\s*110"],
        "risk_level": "Moderate Risk",
        "category": "Artificial Color",
        "side_effects": "Synthetic azo dye; linked to hyperactivity and allergy symptoms in sensitive individuals.",
        "regulatory_status": "Warning Label Required in EU",
        "icon": "🟠",
        "score_penalty": 15,
    },
    {
        "name": "Brilliant Blue FCF (INS 133)",
        "patterns": [r"brilliant\s*blue", r"blue\s*1", r"ins\s*133", r"e\s*133"],
        "risk_level": "Moderate Risk",
        "category": "Artificial Color",
        "side_effects": "Synthetic dye; potential mild skin irritations or allergic reactions in rare cases.",
        "regulatory_status": "Approved with Limits",
        "icon": "🟠",
        "score_penalty": 12,
    },
    {
        "name": "Sodium Benzoate (INS 211)",
        "patterns": [r"sodium\s*benzoate", r"ins\s*211", r"e\s*211", r"preservative\s*\(?211\)?"],
        "risk_level": "Moderate Risk",
        "category": "Preservative",
        "side_effects": "Can form trace benzene when combined with Vitamin C (Ascorbic Acid).",
        "regulatory_status": "FSSAI / FDA Approved with Limits",
        "icon": "🟠",
        "score_penalty": 14,
    },
    {
        "name": "Potassium Sorbate (INS 202)",
        "patterns": [r"potassium\s*sorbate", r"ins\s*202", r"e\s*202", r"preservative\s*\(?202\)?"],
        "risk_level": "Low Risk",
        "category": "Preservative",
        "side_effects": "Widely used anti-fungal preservative; generally safe, rare contact allergic reactions.",
        "regulatory_status": "FDA GRAS / FSSAI Approved",
        "icon": "🟡",
        "score_penalty": 6,
    },
    {
        "name": "Sodium Nitrate / Nitrite (INS 250 / 251)",
        "patterns": [r"sodium\s*nitr[ia]te", r"potassium\s*nitr[ia]te", r"ins\s*250", r"ins\s*251", r"e\s*250", r"e\s*251"],
        "risk_level": "High Risk",
        "category": "Curing Agent / Preservative",
        "side_effects": "Forms carcinogenic nitrosamines during high heat cooking; linked to colon health risks.",
        "regulatory_status": "Strictly Regulated Preservative",
        "icon": "🔴",
        "score_penalty": 24,
    },
    {
        "name": "Monosodium Glutamate (MSG / INS 621)",
        "patterns": [r"monosodium\s*glutamate", r"\bmsg\b", r"ins\s*621", r"e\s*621", r"flavor\s*enhancer\s*\(?621\)?"],
        "risk_level": "Low Risk",
        "category": "Flavor Enhancer",
        "side_effects": "Flavor enhancer providing umami; may cause mild transient sensitivity in rare individuals.",
        "regulatory_status": "FDA GRAS / FSSAI Approved",
        "icon": "🟡",
        "score_penalty": 6,
    },
    {
        "name": "Disodium Inosinate & Guanylate (INS 627, 631)",
        "patterns": [r"disodium\s*inosinate", r"disodium\s*guanylate", r"ins\s*627", r"ins\s*631", r"ins\s*635", r"e\s*627", r"e\s*631"],
        "risk_level": "Low Risk",
        "category": "Flavor Enhancer",
        "side_effects": "Synergistic flavor enhancers; metabolizes to purines, consume with caution if prone to gout.",
        "regulatory_status": "Approved Additive",
        "icon": "🟡",
        "score_penalty": 5,
    },
    {
        "name": "High Fructose Corn Syrup (HFCS)",
        "patterns": [r"high\s*fructose\s*corn\s*syrup", r"\bhfcs\b", r"fructose\s*glucose\s*syrup", r"corn\s*syrup\s*solids"],
        "risk_level": "Moderate Risk",
        "category": "Refined Sweetener",
        "side_effects": "Linked to fatty liver risk, insulin resistance, obesity, and accelerated metabolic syndrome.",
        "regulatory_status": "Permitted Refined Caloric Sweetener",
        "icon": "🟠",
        "score_penalty": 15,
    },
    {
        "name": "Aspartame (INS 951)",
        "patterns": [r"aspartame", r"ins\s*951", r"e\s*951", r"artificial\s*sweetener\s*\(?951\)?"],
        "risk_level": "Moderate Risk",
        "category": "Artificial Sweetener",
        "side_effects": "Classified by IARC as possibly carcinogenic (Group 2B); unsafe for individuals with PKU.",
        "regulatory_status": "FDA Approved / PKU Warning Required",
        "icon": "🟠",
        "score_penalty": 16,
    },
    {
        "name": "Acesulfame Potassium (Ace-K / INS 950)",
        "patterns": [r"acesulfame\s*potassium", r"acesulfame\s*k", r"ace-?k", r"ins\s*950", r"e\s*950"],
        "risk_level": "Moderate Risk",
        "category": "Artificial Sweetener",
        "side_effects": "Zero-calorie synthetic sweetener; some studies show potential gut microbiome disruption.",
        "regulatory_status": "FDA / FSSAI Approved",
        "icon": "🟠",
        "score_penalty": 14,
    },
    {
        "name": "Sucralose (INS 955)",
        "patterns": [r"sucralose", r"ins\s*955", r"e\s*955"],
        "risk_level": "Low Risk",
        "category": "Non-Nutritive Sweetener",
        "side_effects": "Chlorinated sugar derivative; generally safe, high heat cooking may degrade compounds.",
        "regulatory_status": "FDA GRAS / FSSAI Approved",
        "icon": "🟡",
        "score_penalty": 7,
    },
    {
        "name": "Palm Oil / Fractionated Palm Olein",
        "patterns": [r"palm\s*oil", r"palm\s*olein", r"palmolein\s*oil", r"fractionated\s*palm"],
        "risk_level": "Moderate Risk",
        "category": "Saturated Fat",
        "side_effects": "High saturated palmitic acid content; frequent high consumption increases cardiovascular risk.",
        "regulatory_status": "Approved Cooking Oil",
        "icon": "🟠",
        "score_penalty": 12,
    },
    {
        "name": "Refined Palm Kernel Oil",
        "patterns": [r"palm\s*kernel\s*oil"],
        "risk_level": "Moderate Risk",
        "category": "Saturated Fat",
        "side_effects": "Contains over 80% saturated fat; regular excessive intake raises blood lipid levels.",
        "regulatory_status": "Approved Fat",
        "icon": "🟠",
        "score_penalty": 12,
    },
    {
        "name": "Caramel Color (Class III/IV - INS 150c/150d)",
        "patterns": [r"caramel\s*colou?r", r"ins\s*150[a-d]?", r"e\s*150[a-d]?"],
        "risk_level": "Moderate Risk",
        "category": "Food Coloring",
        "side_effects": "Ammonia-process caramel colors (150c/150d) may contain trace 4-MEI byproduct.",
        "regulatory_status": "Approved with Limits",
        "icon": "🟠",
        "score_penalty": 12,
    },
    {
        "name": "Soy Lecithin (INS 322)",
        "patterns": [r"soy\s*lecithin", r"soya\s*lecithin", r"lecithin", r"ins\s*322", r"e\s*322", r"emulsifier\s*\(?322\)?"],
        "risk_level": "Safe",
        "category": "Natural Emulsifier",
        "side_effects": "Plant-derived natural phospholipid; safe for consumption. Contains soy allergen.",
        "regulatory_status": "FDA GRAS / FSSAI Approved",
        "icon": "✅",
        "allergen": "Soy",
        "score_penalty": 0,
    },
    {
        "name": "Citric Acid (INS 330 / Acidity Regulator)",
        "patterns": [r"citric\s*acid", r"ins\s*330", r"e\s*330", r"acidity\s*regulator\s*\(?330\)?"],
        "risk_level": "Safe",
        "category": "Acidity Regulator",
        "side_effects": "Naturally occurring organic acid; completely safe in standard food portions.",
        "regulatory_status": "FDA GRAS / FSSAI Approved",
        "icon": "✅",
        "score_penalty": 0,
    },
    {
        "name": "Sodium Bicarbonate / Baking Soda (INS 500ii)",
        "patterns": [r"sodium\s*bicarbonate", r"baking\s*soda", r"ins\s*500", r"e\s*500"],
        "risk_level": "Safe",
        "category": "Leavening Agent",
        "side_effects": "Standard alkaline mineral leavening agent; safe for regular digestion.",
        "regulatory_status": "FDA GRAS",
        "icon": "✅",
        "score_penalty": 0,
    },
    {
        "name": "Ascorbic Acid / Vitamin C (INS 300)",
        "patterns": [r"ascorbic\s*acid", r"vitamin\s*c", r"ins\s*300", r"e\s*300"],
        "risk_level": "Safe",
        "category": "Antioxidant / Vitamin",
        "side_effects": "Essential nutrient and safe natural antioxidant for food preservation.",
        "regulatory_status": "FDA GRAS / Essential Nutrient",
        "icon": "✅",
        "score_penalty": 0,
    },
    {
        "name": "Tocopherols / Vitamin E (INS 307)",
        "patterns": [r"tocopherol", r"vitamin\s*e", r"ins\s*307", r"e\s*307"],
        "risk_level": "Safe",
        "category": "Natural Antioxidant",
        "side_effects": "Fat-soluble natural antioxidant; safe and beneficial nutrient.",
        "regulatory_status": "FDA GRAS",
        "icon": "✅",
        "score_penalty": 0,
    },
    {
        "name": "Maltodextrin",
        "patterns": [r"maltodextrin", r"malto\s*dextrin"],
        "risk_level": "Low Risk",
        "category": "Processed Carbohydrate",
        "side_effects": "High glycemic index (GI 110-185); rapidly spikes blood glucose levels.",
        "regulatory_status": "FDA GRAS",
        "icon": "🟡",
        "score_penalty": 6,
    },
    {
        "name": "Refined Sugar / Sucrose",
        "patterns": [r"\bsugar\b", r"refined\s*sugar", r"cane\s*sugar", r"sucrose", r"invert\s*sugar", r"liquid\s*glucose"],
        "risk_level": "Low Risk",
        "category": "Refined Sugar",
        "side_effects": "High caloric sweetener; frequent high intake contributes to tooth decay and weight gain.",
        "regulatory_status": "Safe in moderation",
        "icon": "🟡",
        "score_penalty": 5,
    },
    {
        "name": "Iodised Salt / Sodium Chloride",
        "patterns": [r"iodised\s*salt", r"iodized\s*salt", r"\bsalt\b", r"sodium\s*chloride", r"edible\s*common\s*salt"],
        "risk_level": "Safe",
        "category": "Mineral Seasoning",
        "side_effects": "Essential dietary mineral; safe for regular consumption in recommended daily amounts.",
        "regulatory_status": "Essential Food Standard",
        "icon": "✅",
        "score_penalty": 0,
    },
    {
        "name": "Refined Wheat Flour (Maida)",
        "patterns": [r"refined\s*wheat\s*flour", r"\bmaida\b", r"wheat\s*flour", r"enriched\s*flour"],
        "risk_level": "Low Risk",
        "category": "Refined Grain",
        "side_effects": "Stripped of natural bran and fiber; low satiety, contains gluten allergen.",
        "regulatory_status": "Standard Staple",
        "icon": "🟡",
        "allergen": "Wheat / Gluten",
        "score_penalty": 4,
    },
    {
        "name": "Whole Wheat Flour (Atta)",
        "patterns": [r"whole\s*wheat", r"\batta\b", r"whole\s*grain"],
        "risk_level": "Safe",
        "category": "Whole Grain",
        "side_effects": "Rich in dietary fiber and essential micronutrients. Contains gluten allergen.",
        "regulatory_status": "Natural Whole Food",
        "icon": "✅",
        "allergen": "Wheat / Gluten",
        "score_penalty": 0,
    },
    {
        "name": "Milk Solids / Whey / Dairy",
        "patterns": [r"milk\s*solids", r"whey\s*powder", r"skimmed\s*milk", r"dairy\s*whitener", r"milk\s*protein", r"casein"],
        "risk_level": "Safe",
        "category": "Dairy",
        "side_effects": "Source of natural protein and calcium. Contains lactose & dairy allergens.",
        "regulatory_status": "Natural Food",
        "icon": "✅",
        "allergen": "Milk / Dairy",
        "score_penalty": 0,
    },
    {
        "name": "Peanuts / Groundnuts",
        "patterns": [r"peanut", r"groundnut"],
        "risk_level": "Safe",
        "category": "Legume / Nut",
        "side_effects": "Nutrient-dense protein and healthy fats. Major common allergen.",
        "regulatory_status": "Natural Food",
        "icon": "✅",
        "allergen": "Peanuts",
        "score_penalty": 0,
    },
    {
        "name": "Almonds / Cashews / Tree Nuts",
        "patterns": [r"almond", r"cashew", r"walnut", r"pistachio", r"hazelnut", r"tree\s*nut"],
        "risk_level": "Safe",
        "category": "Tree Nut",
        "side_effects": "Heart-healthy unsaturated fats and antioxidants. Tree nut allergen.",
        "regulatory_status": "Natural Food",
        "icon": "✅",
        "allergen": "Tree Nuts",
        "score_penalty": 0,
    },
    {
        "name": "Guar Gum (INS 412)",
        "patterns": [r"guar\s*gum", r"ins\s*412", r"e\s*412", r"stabilizer\s*\(?412\)?"],
        "risk_level": "Safe",
        "category": "Natural Stabilizer",
        "side_effects": "Plant seed soluble fiber; safe and natural thickening agent.",
        "regulatory_status": "FDA GRAS",
        "icon": "✅",
        "score_penalty": 0,
    },
    {
        "name": "Xanthan Gum (INS 415)",
        "patterns": [r"xanthan\s*gum", r"ins\s*415", r"e\s*415", r"stabilizer\s*\(?415\)?"],
        "risk_level": "Safe",
        "category": "Fermented Stabilizer",
        "side_effects": "Safe fermented polysaccharide thickener; well-tolerated in normal food amounts.",
        "regulatory_status": "FDA GRAS",
        "icon": "✅",
        "score_penalty": 0,
    },
    {
        "name": "Mono- and Diglycerides of Fatty Acids (INS 471)",
        "patterns": [r"mono\s*and\s*diglycerides", r"ins\s*471", r"e\s*471", r"emulsifier\s*\(?471\)?"],
        "risk_level": "Low Risk",
        "category": "Emulsifier",
        "side_effects": "Common fat-derived emulsifier; may contain trace trans fats depending on oil source.",
        "regulatory_status": "FDA GRAS",
        "icon": "🟡",
        "score_penalty": 5,
    },
    {
        "name": "Cocoa Solids / Butter",
        "patterns": [r"cocoa\s*solids", r"cocoa\s*butter", r"cocoa\s*powder", r"cacao"],
        "risk_level": "Safe",
        "category": "Natural Cocoa",
        "side_effects": "Rich in polyphenol antioxidants and natural flavonoids.",
        "regulatory_status": "Natural Food",
        "icon": "✅",
        "score_penalty": 0,
    },
    {
        "name": "Sulfites / Sulphur Dioxide (INS 220-228)",
        "patterns": [r"sulphur\s*dioxide", r"sodium\s*metabisulphite", r"sodium\s*metabisulfite", r"ins\s*22[0-8]", r"e\s*22[0-8]"],
        "risk_level": "Moderate Risk",
        "category": "Preservative / Allergen",
        "side_effects": "Can trigger acute bronchospasms and allergic reactions in asthmatic individuals.",
        "regulatory_status": "Allergen Declaration Required",
        "icon": "🟠",
        "allergen": "Sulfites",
        "score_penalty": 15,
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Normalization Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_image_hash(image: Image.Image) -> str:
    """Compute a deterministic SHA-256 fingerprint from image pixels."""
    img_copy = image.convert("RGB").resize((400, 400), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img_copy.save(buf, format="JPEG", quality=85)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def normalize_ingredient_token(token: str) -> str:
    """Clean individual ingredient string (spaces, casing, parenthesis)."""
    token = re.sub(r"^\s*[\d\.\•\-\*]+\s*", "", token)
    token = re.sub(r"\b\d+(\.\d+)?\s*%\s*", "", token)
    token = re.sub(r"\s+", " ", token).strip()
    return token


def parse_raw_ingredients_text(raw_text: str) -> List[str]:
    """Parse comma/semicolon/newline-delimited text into distinct ingredient strings."""
    cleaned = re.sub(r"^(ingredients|contains|composition|samagri)\s*[:\-]\s*", "", raw_text, flags=re.IGNORECASE)
    
    tokens: List[str] = []
    raw_tokens = re.split(r",(?![^()]*\))|;(?![^()]*\))|\n|\|", cleaned)
    
    for rt in raw_tokens:
        tok = normalize_ingredient_token(rt)
        if len(tok) >= 2 and not tok.lower().startswith("may contain") and not tok.lower().startswith("allergen"):
            tokens.append(tok)
            
    return tokens


def match_ingredient_against_kb(raw_name: str) -> Dict[str, Any]:
    """Match an ingredient token against our canonical database using regex patterns."""
    raw_lower = raw_name.lower()
    
    for entry in CANONICAL_DATABASE:
        for pat in entry["patterns"]:
            if re.search(pat, raw_lower, re.IGNORECASE):
                res = {
                    "name": raw_name if len(raw_name) > 4 and not raw_name.startswith("INS") else entry["name"],
                    "canonical_name": entry["name"],
                    "risk_level": entry["risk_level"],
                    "category": entry["category"],
                    "side_effects": entry["side_effects"],
                    "regulatory_status": entry["regulatory_status"],
                    "icon": entry["icon"],
                    "score_penalty": entry["score_penalty"],
                }
                if "allergen" in entry:
                    res["allergen"] = entry["allergen"]
                return res
                
    # Fallback heuristic for unmatched ingredients
    penalty = 0
    cat = "Food Ingredient"
    risk = "Safe"
    icon = "✅"
    side_effects = "Standard food constituent; safe for general consumption."
    reg_status = "Approved Food"

    if re.search(r"artificial|synthetic|preservative|colour|color|stabilizer|emulsifier|flavo[u]?r", raw_lower):
        risk = "Moderate Risk" if re.search(r"artificial|synthetic", raw_lower) else "Low Risk"
        icon = "🟠" if risk == "Moderate Risk" else "🟡"
        penalty = 12 if risk == "Moderate Risk" else 5
        cat = "Food Additive"
        side_effects = "Additive used for texture, shelf-life or flavoring. Consume in moderation."
        reg_status = "Regulated Additive"
    elif re.search(r"oil|fat|hydrogenated", raw_lower):
        cat = "Fats & Oils"
        risk = "Low Risk"
        icon = "🟡"
        penalty = 4
        side_effects = "Dietary lipid; recommended in balanced moderation."
    elif re.search(r"water|flour|grain|salt|spice|herb|extract|milk|fruit|nut|seed|vegetable", raw_lower):
        cat = "Natural Ingredient"
        risk = "Safe"
        icon = "✅"
        penalty = 0
        side_effects = "Natural wholesome ingredient."

    return {
        "name": raw_name.title() if raw_name.isupper() or raw_name.islower() else raw_name,
        "canonical_name": raw_name.title(),
        "risk_level": risk,
        "category": cat,
        "side_effects": side_effects,
        "regulatory_status": reg_status,
        "icon": icon,
        "score_penalty": penalty,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Deterministic Scoring Engine
# ─────────────────────────────────────────────────────────────────────────────

def calculate_deterministic_safety(matched_ingredients: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Deterministically computes:
      - overall_score (0 - 100)
      - overall_grade (A, B, C, D)
      - total_ingredients_found
      - concerning_ingredients_count
      - allergens_detected
    """
    total_count = len(matched_ingredients)
    if total_count == 0:
        return {
            "overall_grade": "C",
            "overall_score": 50,
            "total_ingredients_found": 0,
            "concerning_ingredients_count": 0,
            "allergens_detected": [],
            "ingredients": [],
            "has_high_risk": False,
        }

    base_score = 100
    total_penalty = 0
    concerning_count = 0
    allergens: List[str] = []
    has_high_risk = False
    has_moderate_risk = False

    for ing in matched_ingredients:
        risk = ing.get("risk_level", "Safe")
        pen = ing.get("score_penalty", 0)
        total_penalty += pen

        if risk == "High Risk":
            concerning_count += 1
            has_high_risk = True
        elif risk == "Moderate Risk":
            concerning_count += 1
            has_moderate_risk = True
        elif risk == "Low Risk" and pen >= 5:
            concerning_count += 1

        if "allergen" in ing and ing["allergen"] not in allergens:
            allergens.append(ing["allergen"])

    # Calculate final score with penalty
    raw_score = base_score - total_penalty
    final_score = max(5, min(100, raw_score))

    # Cap grade deterministically based on risk presence
    if has_high_risk:
        if final_score >= 65:
            final_score = 62  # Cap to Caution
        if final_score < 40 or total_penalty >= 45:
            grade = "D"
        else:
            grade = "C"
    else:
        if final_score >= 85:
            grade = "A"
        elif final_score >= 65:
            grade = "B"
        elif final_score >= 40:
            grade = "C"
        else:
            grade = "D"

    return {
        "overall_grade": grade,
        "overall_score": int(final_score),
        "total_ingredients_found": total_count,
        "concerning_ingredients_count": concerning_count,
        "allergens_detected": allergens,
        "ingredients": matched_ingredients,
        "has_high_risk": has_high_risk,
        "has_moderate_risk": has_moderate_risk,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Localized Summary & Recommendation Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_deterministic_summary(
    score_data: Dict[str, Any], language: str = "English"
) -> Tuple[str, str]:
    """Produce deterministic summary & recommendation text in selected language."""
    grade = score_data.get("overall_grade", "C")
    score = score_data.get("overall_score", 50)
    concerning = score_data.get("concerning_ingredients_count", 0)
    allergens = score_data.get("allergens_detected", [])

    allergen_str = f" Contains allergens: {', '.join(allergens)}." if allergens else ""
    allergen_str_hi = f" इसमें {', '.join(allergens)} जैसे एलर्जेंस मौजूद हैं।" if allergens else ""
    allergen_str_hl = f" Isme {', '.join(allergens)} allergens shamil hain." if allergens else ""

    if language == "Hindi":
        if grade == "A":
            summary = f"यह उत्पाद पूरी तरह सुरक्षित और प्राकृतिक अवयवों से भरपूर है (सुरक्षा स्कोर: {score}/100)।{allergen_str_hi}"
            rec = "दैनिक उपयोग के लिए सुरक्षित और पौष्टिक विकल्प।"
        elif grade == "B":
            summary = f"यह उत्पाद सामान्य रूप से सुरक्षित है और इसमें केवल अनुमत खाद्य योजक मौजूद हैं (सुरक्षा स्कोर: {score}/100)।{allergen_str_hi}"
            rec = "संतुलित मात्रा में सेवन के लिए उपयुक्त।"
        elif grade == "C":
            summary = f"इस उत्पाद में {concerning} चिंताजनक घटक/कृत्रिम योजक पाए गए हैं (सुरक्षा स्कोर: {score}/100)।{allergen_str_hi}"
            rec = "नियमित रूप से अधिक मात्रा में सेवन करने से बचें।"
        else:
            summary = f"सावधान: इस उत्पाद में अत्यधिक प्रसंस्कृत या हानिकारक घटक मौजूद हैं (सुरक्षा स्कोर: {score}/100)।{allergen_str_hi}"
            rec = "स्वास्थ्य सुरक्षा के लिए इस उत्पाद का उपयोग सीमित रखें या प्राकृतिक विकल्प चुनें।"

    elif language == "Hinglish":
        if grade == "A":
            summary = f"Yeh product kaafi safe aur clean ingredients se bana hai (Safety Score: {score}/100).{allergen_str_hl}"
            rec = "Daily use ke liye safe aur healthy choice hai."
        elif grade == "B":
            summary = f"Yeh product generally safe hai aur standard permitted additives contain karta hai (Safety Score: {score}/100).{allergen_str_hl}"
            rec = "Moderate quantity me consume karna safe hai."
        elif grade == "C":
            summary = f"Is product me {concerning} concerning additives ya artificial sweeteners detect hue hain (Safety Score: {score}/100).{allergen_str_hl}"
            rec = "Regular excessive consumption avoid karein."
        else:
            summary = f"Warning: Is product me high-risk ya highly ultra-processed ingredients shamil hain (Safety Score: {score}/100).{allergen_str_hl}"
            rec = "Health protection ke liye iska intake minimize karein."

    else:  # English
        if grade == "A":
            summary = f"This product features clean, natural, and safe ingredients with minimal processing (Safety Score: {score}/100).{allergen_str}"
            rec = "Safe and suitable for regular, healthy consumption."
        elif grade == "B":
            summary = f"This product is generally safe and contains standard permitted food additives within safe limits (Safety Score: {score}/100).{allergen_str}"
            rec = "Suitable for consumption in balanced moderation."
        elif grade == "C":
            summary = f"This product contains {concerning} moderate-risk additive(s) or artificial ingredients (Safety Score: {score}/100).{allergen_str}"
            rec = "Consume occasionally and avoid excessive daily intake."
        else:
            summary = f"Caution: Contains hazardous or highly ultra-processed additives that may carry health concerns (Safety Score: {score}/100).{allergen_str}"
            rec = "Consider healthier whole-food alternatives with cleaner labels."

    return summary, rec


# ─────────────────────────────────────────────────────────────────────────────
# Gemini Engine for OCR Extraction (Deterministic Temperature = 0.0)
# ─────────────────────────────────────────────────────────────────────────────

class GeminiEngine:
    VISION_MODELS = [
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-2.5-flash",
        "gemini-flash-latest",
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

    def ocr_extract_ingredients(self, image: Image.Image) -> str:
        """
        Extract only visible ingredient text from product label image.
        Strict deterministic temperature = 0.0.
        """
        img_bytes = self._image_to_bytes(image)
        prompt = (
            "You are a specialized food packaging OCR reader. "
            "Carefully transcribe ONLY the exact ingredients list printed on this product package image. "
            "If no ingredients list is visible or text is completely blurred and illegible, return 'UNREADABLE_LABEL'. "
            "Do NOT guess, invent, or hallucinate missing ingredients. "
            "Output only the extracted ingredient names separated by commas."
        )

        last_err = None
        for model in self.VISION_MODELS:
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=[
                        prompt,
                        genai_types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                    ],
                    config=genai_types.GenerateContentConfig(
                        system_instruction="You are an exact OCR transcriber for ingredient labels. Output exact text only.",
                        temperature=0.0,
                        max_output_tokens=1500,
                        automatic_function_calling=genai_types.AutomaticFunctionCallingConfig(
                            disable=True
                        ),
                    ),
                )
                text = response.text or ""
                if text.strip():
                    self.active_model = model
                    logger.info("OCR extracted successfully with %s", model)
                    return text.strip()
            except Exception as exc:
                last_err = exc
                logger.warning("OCR attempt %s failed: %s", model, exc)

        raise RuntimeError(f"OCR label extraction failed: {last_err}")


# ─────────────────────────────────────────────────────────────────────────────
# Master NutriScan AI Handler
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
            raise RuntimeError("GEMINI_API_KEY is required for image label OCR.")

    def analyze_image(
        self, image: Image.Image, language: str = "English", **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Complete deterministic analysis pipeline:
          1. Image fingerprinting / Cache check
          2. Zero-temperature OCR text extraction
          3. Unreadable image guard
          4. Text normalization & INS/E-number mapping
          5. Deterministic scoring rule engine
          6. Multi-language summary formatting
        """
        # Step 1: Compute Image Fingerprint
        img_hash = compute_image_hash(image)

        # Check if we have cached results for this exact image
        if img_hash in _IMAGE_CACHE:
            logger.info("Cache hit for image hash: %s", img_hash)
            cached_data = _IMAGE_CACHE[img_hash]
            summary, rec = generate_deterministic_summary(cached_data, language=language)
            result = dict(cached_data)
            result["summary"] = summary
            result["recommendation"] = rec
            return result

        # Step 2: OCR Extraction
        ocr_text = self._gemini.ocr_extract_ingredients(image)
        logger.info("Raw OCR Text: %s", ocr_text[:120])

        # Step 3: Check for unreadable label
        if not ocr_text or "UNREADABLE_LABEL" in ocr_text.upper() or len(ocr_text.strip()) < 5:
            unreadable_msg = {
                "English":  "Some ingredients could not be read clearly. Please upload a sharper label photo.",
                "Hindi":    "सामग्रियों को स्पष्ट रूप से पढ़ा नहीं जा सका। कृपया अधिक स्पष्ट फोटो अपलोड करें।",
                "Hinglish": "Kuch ingredients clearly read nahi ho paye. Please ek sharper photo upload karein.",
            }.get(language, "Some ingredients could not be read clearly. Please upload a sharper label photo.")

            return {
                "is_unreadable": True,
                "overall_grade": "N/A",
                "overall_score": 0,
                "summary": unreadable_msg,
                "recommendation": "Ensure good lighting and focus on the ingredient list text.",
                "total_ingredients_found": 0,
                "concerning_ingredients_count": 0,
                "allergens_detected": [],
                "ingredients": [],
            }

        # Step 4: Parse & Normalize Ingredients
        tokens = parse_raw_ingredients_text(ocr_text)

        if not tokens:
            return {
                "is_unreadable": True,
                "overall_grade": "N/A",
                "overall_score": 0,
                "summary": "No identifiable ingredients detected on the label.",
                "recommendation": "Please capture a clear photo centered on the ingredient list.",
                "total_ingredients_found": 0,
                "concerning_ingredients_count": 0,
                "allergens_detected": [],
                "ingredients": [],
            }

        # Step 5: Match each token against Canonical Database
        matched = [match_ingredient_against_kb(tok) for tok in tokens]

        # Step 6: Deterministic Scoring & Grading
        score_data = calculate_deterministic_safety(matched)

        # Step 7: Deterministic Language Summary
        summary, rec = generate_deterministic_summary(score_data, language=language)

        final_result = {
            "is_unreadable": False,
            "overall_grade": score_data["overall_grade"],
            "overall_score": score_data["overall_score"],
            "summary": summary,
            "recommendation": rec,
            "total_ingredients_found": score_data["total_ingredients_found"],
            "concerning_ingredients_count": score_data["concerning_ingredients_count"],
            "allergens_detected": score_data["allergens_detected"],
            "ingredients": score_data["ingredients"],
            "raw_ocr": ocr_text,
        }

        # Store in cache for consistent recall
        _IMAGE_CACHE[img_hash] = final_result
        return final_result

    def get_status_info(self) -> Dict[str, Any]:
        return {
            "gemini_available": self._gemini is not None,
            "groq_available":   bool(self._groq_key),
            "status": "Ready",
        }
