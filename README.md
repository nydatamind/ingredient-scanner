# 🔬 Ingredient Safety Scanner

### ⚡ DEVELOPED BY NITIN YADAV

A production-ready, mobile-first **Ingredient Safety Scanner** web application powered by **Google Gemini** (Vision OCR & Multimodal Safety Parsing) and **Groq Cloud LLMs** (Ultra-Fast LLaMA 3.3/3.1 Analysis) with automated zero-downtime failover.

---

## 📱 Mobile-First Features

- 📷 **Instant Smartphone Camera Scan**: Capture any food or cosmetic label directly from your phone.
- ✍️ **Paste Ingredient Text**: Instant analysis via Groq LLaMA models.
- 🛡️ **A–D Safety Grade & Radar**: Color-coded safety badge (A: Excellent, B: Good, C: Caution, D: Hazardous).
- 🧪 **Detailed Risk Breakdown**: Breakdown per ingredient with side effects and regulatory status.
- ⚠️ **Allergen Alert System**: Highlights common and hidden allergens automatically.
- 🔄 **Smart Multi-Model Routing**: Auto-failover across Gemini & Groq models to ensure 100% uptime.

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
