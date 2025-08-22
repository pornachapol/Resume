# resume-chatbot-backend/chatbot.py
import os, re, time, io
from typing import List, Tuple, Optional, Dict

import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import pdfplumber

# ===================== ENV =====================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RESUME_URL = os.getenv("RESUME_URL")  # e.g. https://raw.githubusercontent.com/<user>/<repo>/main/assets/Resume.pdf
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS", "")).split(",") if o.strip()]
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set")

genai.configure(api_key=GOOGLE_API_KEY)

# ===================== APP =====================
app = FastAPI(title="Resume-only Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],  # เปิดกว้างช่วงทดสอบ; ขึ้นจริงให้กำหนดโดเมน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Schemas =====================
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    sources: List[Tuple[int, str]] = []

# ===================== In-memory store =====================
CHUNKS: List[str] = []
VECTORIZER = None
MATRIX = None
LAST_FETCH_AT = 0

PROFILE: Dict = {
    "name": None,
    "contacts": {},      # {"email": "...", "phone": "...", "location": "...", "links": [...]}
    "skills": [],
    "experience": [],
    "education": [],
    "etc": []
}

# ===================== Helpers =====================
def clean(txt: str) -> str:
    return re.sub(r"\s+", " ", txt or "").strip()

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript"]):
        t.decompose()
    return clean(soup.get_text(" "))

def chunk_text(text: str, size=1200, overlap=200) -> List[str]:
    out, i = [], 0
    if not text:
        return out
    while i < len(text):
        out.append(text[i:i+size])
        i += max(1, size - overlap)
    return out

async def fetch_resume_text() -> str:
    """
    ดึงเนื้อหาเรซูเม่จาก RESUME_URL
    - ถ้าเป็น PDF: ใช้ pdfplumber แปลง PDF -> ข้อความ
    - ถ้าเป็น HTML: ใช้ BeautifulSoup แปลง HTML -> ข้อความ
    """
    if not RESUME_URL:
        return ""
    headers = {"User-Agent": "resume-bot/1.0 (+https://render.com)"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(RESUME_URL, follow_redirects=True, headers=headers)
            r.raise_for_status()
            content_type = (r.headers.get("Content-Type") or "").lower()

            # PDF
            if "application/pdf" in content_type or RESUME_URL.lower().endswith(".pdf"):
                with pdfplumber.open(io.BytesIO(r.content)) as pdf:
                    pages = []
                    for page in pdf.pages:
                        txt = page.extract_text() or ""
                        if txt:
                            pages.append(txt)
                return clean("\n".join(pages))

            # HTML
            return html_to_text(r.text)
    except Exception as e:
        print(f"[fetch_resume_text] error: {e}")
        return ""

def parse_profile(full_text: str):
    """
    สกัดข้อมูลหลักจากข้อความเรซูเม่เป็น PROFILE dict:
    - name / contacts(email/phone/location/links)
    - skills / experience / education / etc
    """
    global PROFILE
    text = full_text or ""
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    # --- Name heuristic (ดูบรรทัดต้น ๆ) ---
    name = None
    for line in lines[:30]:
        ws = line.split()
        if 2 <= len(ws) <= 4 and all(w[:1].isupper() for w in ws if w.isalpha()):
            name = line
            break

    # --- Contacts ---
    email = None
    phone = None
    location = None
    links: List[str] = []

    m = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    if m: email = m.group(0)

    m = re.search(r"(\+?\d[\d \-]{7,}\d)", text)
    if m: phone = m.group(1)

    m = re.search(r"(Bangkok|Thailand|Address\s*:\s*.+|Location\s*:\s*.+)", text, flags=re.IGNORECASE)
    if m: location = m.group(0)

    links = re.findall(r"https?://\S+", text)

    # --- Section grouping (หยาบ) ---
    blocks = {"skills": [], "experience": [], "education": [], "etc": []}
    current = None
    for line in lines:
        low = line.lower()
        if re.match(r"skills\b", low) or "skills &" in low:
            current = "skills"; continue
        if "experience" in low or "professional experience" in low:
            current = "experience"; continue
        if "education" in low or "certification" in low:
            current = "education"; continue
        if current is None:
            current = "etc"
        blocks[current].append(line)

    # skills → list
    skills: List[str] = []
    if blocks["skills"]:
        joined = " ".join(blocks["skills"])
        parts = re.split(r"[•\u2022,;|/]", joined)
        for p in parts:
            s = p.strip()
            if 1 < len(s) <= 40:
                skills.append(s)
        # uniq
        skills = list(dict.fromkeys(skills))

    PROFILE = {
        "name": name,
        "contacts": {
            "email": email,
            "phone": phone,
            "location": location,
            "links": links[:6],
        },
        "skills": skills,
        "experience": blocks["experience"][:80],
        "education": blocks["education"][:40],
        "etc": blocks["etc"][:40]
    }

def build_index(chunks: List[str]):
    global VECTORIZER, MATRIX
    VECTORIZER = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    MATRIX = VECTORIZER.fit_transform(chunks)

async def ensure_index(force=False):
    global CHUNKS, LAST_FETCH_AT, PROFILE
    if CHUNKS and not force and (time.time() - LAST_FETCH_AT < 3600):
        return
    text = await fetch_resume_text()
    parse_profile(text)
    CHUNKS = chunk_text(text) if text else []
    if CHUNKS:
        build_index(CHUNKS)
    LAST_FETCH_AT = time.time()

def normalize_query(q: str) -> str:
    ql = q.lower()
    mapping = {
        "ชื่อเต็ม": "full name",
        "ชื่อจริง": "first name",
        "นามสกุล": "surname",
        "ชื่อ": "name",
        "what is your name": "full name",
        "your name": "full name",
        "name?": "name",
        "ติดต่อ": "contact",
    }
    for k, v in mapping.items():
        ql = ql.replace(k, v)
    return ql

def retrieve(q: str, k=5):
    if not CHUNKS or VECTORIZER is None:
        return []
    qv = VECTORIZER.transform([q])
    sims = cosine_similarity(qv, MATRIX).ravel()
    order = sims.argsort()[::-1][:k]
    return [(int(i), CHUNKS[int(i)]) for i in order if sims[int(i)] > 0.01]

def try_extract_name_heuristic(full_text: str) -> Optional[str]:
    if not full_text:
        return None
    head = full_text[:2000]
    lines = [l.strip() for l in head.split("\n") if l.strip()]
    candidates = []
    for line in lines[:40]:
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0:1].isupper() for w in words if w.isalpha()):
            candidates.append(line)
    if not candidates:
        for idx, line in enumerate(lines[:50]):
            if "resume" in line.lower() and idx > 0:
                candidates.append(lines[idx-1])
    return max(candidates, key=len) if candidates else None

def answer_from_profile(q: str) -> Optional[str]:
    ql = q.lower()

    # Name
    if any(k in ql for k in ["ชื่ออะไร", "ชื่อคืออะไร", "full name", "your name", "name?"]):
        if PROFILE.get("name"):
            return f"ชื่อ-นามสกุล: {PROFILE['name']}"

    # Contact
    if any(k in ql for k in ["ติดต่อ", "contact", "email", "อีเมล", "โทร", "phone", "linkedin", "github", "location", "ที่อยู่"]):
        c = PROFILE.get("contacts", {})
        parts = []
        if c.get("email"): parts.append(f"อีเมล: {c['email']}")
        if c.get("phone"): parts.append(f"โทร: {c['phone']}")
        if c.get("location"): parts.append(f"พื้นที่: {c['location']}")
        if c.get("links"): parts.append("ลิงก์: " + ", ".join(c['links']))
        if parts:
            return " | ".join(parts)

    # Skills
    if any(k in ql for k in ["ทักษะ", "skills", "skill", "สกิล"]):
        if PROFILE.get("skills"):
            return "ทักษะหลัก: " + ", ".join(PROFILE["skills"][:30])

    # Experience
    if any(k in ql for k in ["ประสบการณ์", "experience", "เคยทำงาน", "ทำงานที่ไหน"]):
        exp = PROFILE.get("experience") or []
        if exp:
            return "ประสบการณ์ (สรุป):\n- " + "\n- ".join(exp[:8])

    # Education
    if any(k in ql for k in ["การศึกษา", "education", "เรียนที่ไหน", "จบจาก"]):
        edu = PROFILE.get("education") or []
        if edu:
            return "การศึกษา:\n- " + "\n- ".join(edu[:8])

    return None

def ask_gemini(question: str, contexts: List[str]) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    ctx = "\n\n---\n".join(contexts) if contexts else ""
    prompt = f"""
คุณเป็นผู้ช่วยตอบคำถามจากเรซูเม่เท่านั้น
ห้ามเดา ถ้าไม่พบในบริบทให้ตอบข้อมูลที่อ่านได้แบบสรุป
ตอบภาษาไทยสั้น กระชับ

[บริบทจากเรซูเม่]
{ctx}

[คำถาม]
{question}
"""
    try:
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip() or "ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่ของฉัน"
    except Exception as e:
        print(f"[ask_gemini] error: {e}")
        return "ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่ของฉัน"

# ===================== Routes =====================
@app.get("/")
def home():
    return {"service": "resume-chatbot", "status": "ok", "docs": "/docs", "health": "/health"}

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/debug")
async def debug():
    await ensure_index()
    return {
        "chunks": len(CHUNKS),
        "resume_url": RESUME_URL,
        "model": MODEL_NAME,
        "profile": {
            "name": PROFILE.get("name"),
            "contacts": PROFILE.get("contacts"),
            "skills_count": len(PROFILE.get("skills", [])),
            "experience_items": len(PROFILE.get("experience", [])),
            "education_items": len(PROFILE.get("education", [])),
        }
    }

@app.get("/refresh")
async def refresh():
    await ensure_index(force=True)
    return {"ok": True, "chunks": len(CHUNKS)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    await ensure_index()

    # 1) ลองตอบจาก PROFILE ก่อน (เร็วและชัวร์)
    direct = answer_from_profile(req.message)
    if direct:
        return ChatResponse(reply=direct, sources=[])

    # 2) ไม่พอ → ใช้ retrieval + โมเดล
    q_norm = normalize_query(req.message)
    hits = retrieve(q_norm, k=5)

    reply: Optional[str] = None

    # เสริม heuristics กรณีถามชื่อแบบสั้นมาก
    if ("name" in q_norm or "surname" in q_norm or "full name" in q_norm or "first name" in q_norm) and not hits:
        text = " ".join(CHUNKS) if CHUNKS else ""
        maybe_name = try_extract_name_heuristic(text)
        if maybe_name:
            reply = f"ชื่อ-นามสกุล (จากเรซูเม่): {maybe_name}"

    contexts = [c for _, c in hits][:3]
    if not reply:
        if contexts:
            reply = ask_gemini(req.message, contexts)
        else:
            reply = "ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่ของฉัน"

    previews = [(i, (c[:140] + ("..." if len(c) > 140 else ""))) for i, c in hits[:3]]
    return ChatResponse(reply=reply, sources=previews)
