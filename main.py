import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict
import re
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI Backend!"}

@app.get("/api/hello")
def hello():
    return {"message": "Hello from the backend API!"}

@app.get("/test")
def test_database():
    """Test endpoint to check if database is available and accessible"""
    response = {
        "backend": "✅ Running",
        "database": "❌ Not Available",
        "database_url": None,
        "database_name": None,
        "connection_status": "Not Connected",
        "collections": []
    }
    
    try:
        # Try to import database module
        from database import db
        
        if db is not None:
            response["database"] = "✅ Available"
            response["database_url"] = "✅ Configured"
            response["database_name"] = db.name if hasattr(db, 'name') else "✅ Connected"
            response["connection_status"] = "Connected"
            
            # Try to list collections to verify connectivity
            try:
                collections = db.list_collection_names()
                response["collections"] = collections[:10]  # Show first 10 collections
                response["database"] = "✅ Connected & Working"
            except Exception as e:
                response["database"] = f"⚠️  Connected but Error: {str(e)[:50]}"
        else:
            response["database"] = "⚠️  Available but not initialized"
            
    except ImportError:
        response["database"] = "❌ Database module not found (run enable-database first)"
    except Exception as e:
        response["database"] = f"❌ Error: {str(e)[:50]}"
    
    # Check environment variables
    import os
    response["database_url"] = "✅ Set" if os.getenv("DATABASE_URL") else "❌ Not Set"
    response["database_name"] = "✅ Set" if os.getenv("DATABASE_NAME") else "❌ Not Set"
    
    return response

# ===============================
# AI-like Text Utilities (no external APIs)
# ===============================

sentence_end_re = re.compile(r"(?<=[.!?]) +")
word_re = re.compile(r"[A-Za-z']+")


def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    # Split on sentence boundaries while keeping punctuation
    sentences = re.split(sentence_end_re, text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> List[str]:
    return [w.lower() for w in word_re.findall(text)]


def count_syllables_word(word: str) -> int:
    word = word.lower()
    vowels = "aeiouy"
    syllables = 0
    prev_is_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_is_vowel:
            syllables += 1
        prev_is_vowel = is_vowel
    if word.endswith("e") and syllables > 1:
        syllables -= 1
    return max(1, syllables)


def text_stats(text: str) -> Dict[str, float]:
    sentences = split_sentences(text)
    words = tokenize(text)
    syllables = sum(count_syllables_word(w) for w in words)
    num_sentences = max(1, len(sentences))
    num_words = len(words)
    num_chars = len(re.sub(r"\s", "", text))

    words_per_sentence = num_words / num_sentences if num_sentences else 0
    syllables_per_word = syllables / num_words if num_words else 0

    # Flesch Reading Ease
    flesch = 206.835 - 1.015 * words_per_sentence - 84.6 * syllables_per_word if num_words else 0

    # Flesch-Kincaid Grade
    fk_grade = 0.39 * words_per_sentence + 11.8 * syllables_per_word - 15.59 if num_words else 0

    reading_time_min = num_words / 200.0  # ~200 wpm

    return {
        "sentences": len(sentences),
        "words": num_words,
        "characters": num_chars,
        "syllables": syllables,
        "flesch_reading_ease": round(flesch, 2),
        "flesch_kincaid_grade": round(fk_grade, 2),
        "reading_time_minutes": round(reading_time_min, 2),
    }


def summarize_text(text: str, max_sentences: int = 3) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return text.strip()
    words = tokenize(text)
    freq: Dict[str, int] = {}
    for w in words:
        if len(w) <= 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    # score sentences by sum of word frequencies
    scores = []
    for idx, s in enumerate(sentences):
        s_words = tokenize(s)
        score = sum(freq.get(w, 0) for w in s_words)
        # prefer earlier sentences slightly
        score += max(0, len(sentences) - idx) * 0.01
        scores.append((score, idx, s))
    top = sorted(scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    # restore original order
    top_sorted = sorted(top, key=lambda x: x[1])
    return " ".join(s for _, _, s in top_sorted)


formal_map = {
    "can't": "cannot",
    "won't": "will not",
    "don't": "do not",
    "I'm": "I am",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "we're": "we are",
    "you're": "you are",
}

casual_map = {
    "do not": "don't",
    "cannot": "can't",
    "will not": "won't",
    "it is": "it's",
    "that is": "that's",
    "there is": "there's",
    "we are": "we're",
    "you are": "you're",
    "I am": "I'm",
}


def apply_tone(text: str, tone: Literal["formal", "casual"]) -> str:
    mapping = formal_map if tone == "formal" else casual_map
    out = text
    # simple whole-word replacements
    for src, tgt in mapping.items():
        # case-insensitive, respect word boundaries
        pattern = re.compile(rf"\b{re.escape(src)}\b", re.IGNORECASE)
        out = pattern.sub(tgt, out)
    return out


def rewrite_text(text: str, mode: Literal["simplify", "expand", "tone_formal", "tone_casual"]) -> str:
    if mode == "simplify":
        # Shorten long sentences by splitting at commas/semicolons
        sentences = split_sentences(text)
        simplified = []
        for s in sentences:
            parts = re.split(r",|;|:\s", s)
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) > 1:
                simplified.extend(parts)
            else:
                simplified.append(s)
        return ". ".join(simplified)
    elif mode == "expand":
        # Add gentle elaborations to sentences (rule-based)
        sentences = split_sentences(text)
        expanded = []
        for s in sentences:
            if len(tokenize(s)) < 6:
                expanded.append(s + " In other words, this point highlights an essential idea.")
            else:
                expanded.append(s)
        return " ".join(expanded)
    elif mode == "tone_formal":
        return apply_tone(text, "formal")
    elif mode == "tone_casual":
        return apply_tone(text, "casual")
    return text


def suggest_titles(text: str) -> List[str]:
    # Extract top keywords by frequency
    words = tokenize(text)
    stop = set(
        [
            "the","is","and","a","an","to","of","in","on","for","with","as","by","it","this","that","are","was","be","or","from","at","we","you","i",
        ]
    )
    freq: Dict[str, int] = {}
    for w in words:
        if w in stop or len(w) <= 2:
            continue
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq.items(), key=lambda x: x[1], reverse=True)[:5]
    keywords = [w for w, _ in top]
    if not keywords:
        return [
            "Quick Notes",
            "Draft Document",
            "Thoughts and Ideas",
        ]
    base = " ".join(k.capitalize() for k in keywords)
    return [
        f"{base}: A Brief Overview",
        f"Understanding {' '.join(k.capitalize() for k in keywords[:3])}",
        f"Guide to {base}",
        f"{keywords[0].capitalize()} and Beyond",
    ]

# ===============================
# Request/Response Models
# ===============================

class TextPayload(BaseModel):
    text: str = Field("")

class SummarizeRequest(BaseModel):
    text: str
    max_sentences: int = 3

class RewriteRequest(BaseModel):
    text: str
    mode: Literal["simplify", "expand", "tone_formal", "tone_casual"]

class TitlesRequest(BaseModel):
    text: str

# ===============================
# API Routes
# ===============================

@app.post("/api/analyze")
def api_analyze(payload: TextPayload):
    stats = text_stats(payload.text)
    return {"stats": stats}


@app.post("/api/summarize")
def api_summarize(req: SummarizeRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    summary = summarize_text(req.text, max_sentences=max(1, min(7, req.max_sentences)))
    return {"summary": summary}


@app.post("/api/rewrite")
def api_rewrite(req: RewriteRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")
    rewritten = rewrite_text(req.text, req.mode)
    return {"result": rewritten}


@app.post("/api/titles")
def api_titles(req: TitlesRequest):
    suggestions = suggest_titles(req.text)
    return {"titles": suggestions}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
