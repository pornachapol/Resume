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
import fitz  # PyMuPDF
import numpy as np

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

def chunk_text(text: str, size=1800, overlap=350) -> List[str]:
    blocks = smart_split(text)   # แยกเป็นบล็อกก่อน
    chunks = []
    for blk in blocks:
        i = 0
        while i < len(blk):
            chunks.append(blk[i:i+size])
            i += max(1, size - overlap)
    return chunks

def embed_texts(texts: List[str]) -> np.ndarray:
    """
    ใช้ Google AI Studio: text-embedding-004
    คืนค่าเป็น np.ndarray shape (N, D)
    """
    if not texts:
        return np.zeros((0, 1))
    try:
        resp = genai.embed_content(
            model="text-embedding-004",
            content=texts,
            task_type="retrieval_document"  # บอกว่างานนี้สำหรับดึงเอกสาร
        )
        # resp["embedding"] เมื่อส่ง 1 รายการ / resp["embeddings"] เมื่อส่งหลายรายการ (SDK อาจต่างเวอร์ชัน)
        embs = resp.get("embeddings") or resp.get("embedding")
        if isinstance(embs, list) and isinstance(embs[0], dict) and "values" in embs[0]:
            vecs = np.array([e["values"] for e in embs], dtype="float32")
        elif isinstance(embs, dict) and "values" in embs:
            vecs = np.array([embs["values"]], dtype="float32")
        else:
            # บาง SDK: resp เป็น list ของ float[]
            vecs = np.array(embs, dtype="float32")
        # ปกติ vectorize เพื่อความปลอดภัย
        return vecs
    except Exception as e:
        print(f"[embed_texts] error: {e}")
        return np.zeros((len(texts), 1), dtype="float32")

EMB_MATRIX = None  # np.ndarray (num_chunks, dim)

async def fetch_resume_text() -> str:
    """
    ดึงเนื้อหาจาก RESUME_URL
    - ถ้าเป็น PDF: ใช้ PyMuPDF (fitz) ดึงข้อความ (แม่นกับเลย์เอาต์แน่น ๆ)
    - ถ้าเป็น HTML: ใช้ BeautifulSoup
    """
    if not RESUME_URL:
        return ""
    headers = {"User-Agent": "resume-bot/1.0 (+https://render.com)"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(RESUME_URL, follow_redirects=True, headers=headers)
            r.raise_for_status()
            content_type = (r.headers.get("Content-Type") or "").lower()

            if "application/pdf" in content_type or RESUME_URL.lower().endswith(".pdf"):
                # อ่าน PDF ด้วย PyMuPDF
                doc = fitz.open(stream=r.content, filetype="pdf")
                pages = []
                for p in doc:
                    # ใช้ textpage.extractTEXT() ที่เรียงลำดับตาม layout ได้ดีกว่า get_text("text")
                    tp = p.get_textpage()
                    txt = tp.extractTEXT() or ""
                    if txt.strip():
                        pages.append(txt)
                return clean("\n".join(pages))

            # HTML
            return html_to_text(r.text)
    except Exception as e:
        print(f"[fetch_resume_text] error: {e}")
        return ""

SECTION_HINTS = [
    r"\b(PROFESSIONAL EXPERIENCE|EXPERIENCE)\b",
    r"\b(EDUCATION|CERTIFICATIONS?)\b",
    r"\b(KEY ACHIEVEMENTS|ACHIEVEMENTS)\b",
    r"\b(AREA OF EXPERTISE|SKILLS?)\b",
    r"\b(ADDITIONAL INFORMATION|LANGUAGES?)\b",
]

def smart_split(text: str) -> List[str]:
    # แยกเป็นบล็อกเมื่อเจอหัวข้อใหญ่ ๆ
    if not text:
        return []
    lines = text.splitlines()
    blocks, cur = [], []
    import re as _re
    for ln in lines:
        if any(_re.search(pat, ln.strip(), flags=_re.IGNORECASE) for pat in SECTION_HINTS):
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
        cur.append(ln)
    if cur:
        blocks.append("\n".join(cur).strip())
    # ถ้า detect ไม่ได้ ก็คืนทั้งก้อน
    return [b for b in blocks if b] or [text]
    
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
    global VECTORIZER, MATRIX, EMB_MATRIX
    VECTORIZER = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    MATRIX = VECTORIZER.fit_transform(chunks)
    EMB_MATRIX = embed_texts(chunks)  # <- เพิ่มฝั่งเวคเตอร์
    # L2 normalize เพื่อง่ายต่อคอสไซน์
    if EMB_MATRIX is not None and EMB_MATRIX.size > 0:
        norms = np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True) + 1e-12
        EMB_MATRIX = EMB_MATRIX / norms


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

def retrieve_hybrid(q: str, k=5, alpha=0.6):
    """
    alpha: น้ำหนักของ embedding (semantic)
    (1-alpha): น้ำหนัก TF-IDF (keyword)
    """
    if not CHUNKS or VECTORIZER is None:
        return []
    # TF-IDF
    qv = VECTORIZER.transform([q])
    sims_tfidf = cosine_similarity(qv, MATRIX).ravel()

    # Embedding
    emb_q = embed_texts([q])
    if emb_q is not None and emb_q.size > 0 and EMB_MATRIX is not None and EMB_MATRIX.size > 0:
        qn = emb_q / (np.linalg.norm(emb_q, axis=1, keepdims=True) + 1e-12)
        sims_vec = (EMB_MATRIX @ qn.T).ravel()
    else:
        sims_vec = np.zeros_like(sims_tfidf)

    # Normalize ทั้งสองฝั่ง
    def norm(x):
        x = x - x.min()
        m = x.max() or 1.0
        return x / m
    s1 = norm(sims_tfidf)
    s2 = norm(sims_vec)

    hybrid = alpha * s2 + (1 - alpha) * s1
    order = np.argsort(hybrid)[::-1][:k]
    return [(int(i), CHUNKS[int(i)], float(hybrid[int(i)])) for i in order if hybrid[int(i)] > 0.01]


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
คุณเป็นผู้ช่วยตอบคำถามจากไฟล์เรซูเม่ (PDF) เท่านั้น
- อ้างอิงเฉพาะข้อมูลที่อยู่ใน [บริบท] ด้านล่าง
- ถ้าคำถามกว้าง/ปลายเปิด ให้สรุปเป็นหัวข้อย่อ ๆ อ่านง่าย
- ถ้าคำถามเฉพาะ ให้ดึงคำตอบตรงจากบริบท
- ถ้าไม่มีข้อมูลในบริบท ให้ตอบว่า: "คำถามนี้ไม่มีอยู่ใน PDF"

[บริบท]
{ctx}

[คำถาม]
{question}

โปรดตอบเป็นภาษาไทย กระชับ ชัดเจน
"""
    try:
        resp = model.generate_content(prompt)
        return (getattr(resp, "text", "") or "").strip() or "คำถามนี้ไม่มีอยู่ใน PDF"
    except Exception as e:
        print(f"[ask_gemini] error: {e}")
        return "คำถามนี้ไม่มีอยู่ใน PDF"
def is_summary_query(q: str) -> bool:
    ql = q.lower()
    keys = ["สรุป", "เล่าภาพรวม", "overview", "summary", "โดยรวม", "แนะนำตัว"]
    return any(k in ql for k in keys)

def summarize_all_chunks(chunks: List[str]) -> str:
    # map-reduce เบา ๆ: สรุปทีละก้อน แล้วรวม
    model = genai.GenerativeModel(MODEL_NAME)
    partials = []
    for c in chunks[:8]:  # จำกัดเพื่อความเร็ว
        resp = model.generate_content(f"สรุปสาระสำคัญของข้อความนี้เป็น bullet ไทยสั้น ๆ:\n\n{c}")
        partials.append((resp.text or "").strip())
    # reduce
    joined = "\n".join(partials)
    resp2 = model.generate_content(f"รวมสรุปเหล่านี้ให้เป็นภาพรวมอ่านง่าย ไม่ซ้ำซ้อน:\n\n{joined}")
    return (resp2.text or "").strip()


@app.get("/context")
async def get_context():
    await ensure_index()
    # คืนตัวอย่าง 3 ชิ้นแรกให้ตรวจว่าอ่านชื่อ/ส่วนหัว ๆ มาไหม
    return {"count": len(CHUNKS), "sample": CHUNKS[:3]}

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

    # 1) โจทย์พื้นฐาน → ตอบจาก PROFILE
    direct = answer_from_profile(req.message)
    if direct:
        return ChatResponse(reply=direct, sources=[])

    # 2) คำถามสรุป/ปลายเปิด
    if is_summary_query(req.message):
        reply = summarize_all_chunks(CHUNKS)
        # ไม่ต้องแสดง sources ก็ได้ เพราะเป็นการสรุปรวม
        return ChatResponse(reply=reply, sources=[])

    # 3) คำถามเฉพาะเจาะจง → Hybrid retrieval
    q_norm = normalize_query(req.message)
    hits = retrieve_hybrid(q_norm, k=5, alpha=0.6)
    contexts = [c for _, c, _ in hits][:3]

    if contexts:
        reply = ask_gemini(req.message, contexts)
    else:
        reply = "คำถามนี้ไม่มีอยู่ใน PDF"

    previews = [(i, (c[:140] + ("..." if len(c) > 140 else ""))) for i, c, _ in hits[:3]]
    return ChatResponse(reply=reply, sources=previews)
