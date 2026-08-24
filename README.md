# 🔬 NutriScan · NY

### ⚡ DEVELOPED BY NITIN YADAV

A production-ready, mobile-first **NutriScan · NY** web application providing instant, deterministic food ingredient safety analysis, additive classification, and allergen detection.

---

## 📱 Features

- 📷 **Instant Camera & Gallery Scan**: Capture or choose a label photo with automatic one-touch instant analysis.
- 🛡️ **Deterministic Safety Grading (A–D)**: Fixed rule-based scoring (A: Excellent, B: Good, C: Caution, D: Hazardous).
- 🧪 **Detailed Risk Breakdown**: Breakdown per ingredient with side effects, regulatory status, and INS/E-number mapping.
- ⚠️ **Allergen Alert System**: Highlights common allergens (Soy, Milk, Gluten, Nuts, Sulfites, etc.).
- 🌐 **Multi-Language Support**: Seamless reporting in English, Hindi, and Hinglish without altering safety decisions.
- 🔒 **Stable & Consistent Results**: Built-in image fingerprint caching and strict deterministic scoring rules.

---

## 🚀 Quick Setup

### 1. Configure Environment Keys
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=AIzaSy...your_gemini_api_key
GROQ_API_KEY=gsk_...your_groq_api_key
```

### 2. Run the App
```bash
streamlit run app.py
```

The app will launch at `http://localhost:8501`.

---

## 👨‍💻 Developer
**DEVELOPED BY NITIN YADAV**
