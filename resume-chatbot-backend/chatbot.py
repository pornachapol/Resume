# resume-chatbot-backend/chatbot.py
import os, re, time
from typing import List, Tuple
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import pdfplumber

# ===== ENV =====
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RESUME_URL = os.getenv("RESUME_URL")  # ex: https://nachapol-resume.streamlit.app
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS","")).split(",") if o.strip()]
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-flash")  # ใช้ flash-2.0 ถ้าบัญชีคุณเปิดใช้ได้

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set")

genai.configure(api_key=GOOGLE_API_KEY)

# ===== APP =====
app = FastAPI(title="Resume-only Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],  # ช่วงทดสอบเปิดได้, ขึ้นจริงควรกำหนดโดเมน
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== Schemas =====
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    sources: List[Tuple[int, str]] = []

# ===== Index In-Memory =====
CHUNKS: List[str] = []
VECTORIZER = None
MATRIX = None
LAST_FETCH_AT = 0

# ===== Helpers =====
def clean(txt: str) -> str:
    return re.sub(r"\s+", " ", txt or "").strip()

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script","style","noscript"]):
        t.decompose()
    return clean(soup.get_text(" "))

def chunk_text(text: str, size=1200, overlap=200) -> List[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i:i+size])
        i += (size - overlap)
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
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(RESUME_URL, follow_redirects=True, headers=headers)
        r.raise_for_status()
        content_type = (r.headers.get("Content-Type") or "").lower()

        # PDF (เช็กทั้ง Content-Type และนามสกุลไฟล์)
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


def retrieve(q: str, k=5):
    if not CHUNKS or VECTORIZER is None:
        return []
    qv = VECTORIZER.transform([q])
    sims = cosine_similarity(qv, MATRIX).ravel()
    order = sims.argsort()[::-1][:k]
    return [(int(i), CHUNKS[int(i)]) for i in order if sims[int(i)] > 0.01]

def try_extract_name_heuristic(full_text: str) -> str | None:
    """
    ดึงชื่อ-นามสกุลจากช่วงต้น ๆ ของหน้า (ภาษาอังกฤษ) แบบ heuristic
    ใช้เฉพาะข้อมูลที่มีอยู่บนหน้าเรซูเม่เท่านั้น
    """
    head = full_text[:2000]
    lines = [l.strip() for l in head.split("\n") if l.strip()]
    # เลือกบรรทัดที่ดูเหมือนชื่อ (2-4 คำ ตัวแรกขึ้นต้นด้วยตัวใหญ่)
    candidates = []
    for line in lines[:40]:
        words = line.split()
        if 2 <= len(words) <= 4 and all(w[0:1].isupper() for w in words if w.isalpha()):
            candidates.append(line)
    # ถ้าไม่มี pattern ข้างบน ลองหาบรรทัดที่มี 'Resume' ใกล้ ๆ แล้วดูบรรทัดเหนือ
    if not candidates:
        for idx, line in enumerate(lines[:50]):
            if "resume" in line.lower() and idx > 0:
                candidates.append(lines[idx-1])
    # คืนค่าที่น่าจะเป็นชื่อมากที่สุด (ยาวสุด)
    if candidates:
        return max(candidates, key=len)
    return None

def build_messages(question: str, contexts: List[str]) -> list[dict]:
    sys = (
        "คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับเรซูเม่ โดยใช้เฉพาะข้อมูลจากบริบทที่ให้เท่านั้น "
        "ห้ามเดา/ห้ามเพิ่ม ถ้าไม่พบข้อมูล ให้ตอบว่า 'ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่ของฉัน' "
        "รองรับคำถามภาษาไทยที่กล่าวถึงชื่อ/นามสกุล โดยแม็พกับคำภาษาอังกฤษ เช่น name, full name, surname"
    )
    ctx = "\n\n---\n".join(contexts) if contexts else ""
    user = (
        f"Question: {question}\n\n"
        f"Context from resume page:\n---\n{ctx}\n---\n"
        "Answer in Thai, short and factual, using ONLY the context above."
    )
    return [{"role":"system","content":sys},{"role":"user","content":user}]

def ask_gemini(question: str, contexts: List[str]) -> str:
    model = genai.GenerativeModel(MODEL_NAME)
    ctx = "\n\n---\n".join(contexts) if contexts else ""
    prompt = f"""
คุณเป็นผู้ช่วยตอบคำถามจากเรซูเม่เท่านั้น
ห้ามเดา ถ้าไม่พบในบริบทให้ตอบว่า "ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่ของฉัน"
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


# ===== Routes =====
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/refresh")
async def refresh():
    await ensure_index(force=True)
    return {"ok": True, "chunks": len(CHUNKS)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    await ensure_index()
    q_norm = normalize_query(req.message)
    hits = retrieve(q_norm, k=5)

    # กรณีพิเศษ: ถามชื่อ/นามสกุล แต่ retrieval ว่าง ลอง heuristic จาก full text
    reply = None
    if ("name" in q_norm or "surname" in q_norm or "full name" in q_norm or "first name" in q_norm) and not hits:
        # ดึง full text ตรง ๆ แล้วลองเดาชื่อจากหัวกระดาษ
        text = " ".join(CHUNKS) if CHUNKS else ""
        maybe_name = try_extract_name_heuristic(text)
        if maybe_name:
            reply = f"ชื่อ-นามสกุล (จากหน้าเรซูเม่): {maybe_name}"

    # ปกติ: ถ้ามี context ให้โมเดลตอบ
    contexts = [c for _, c in hits][:3]
    if not reply:
        if contexts:
            reply = ask_gemini(req.message, contexts)
        else:
            reply = "ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่ของฉัน"

    previews = [(i, (c[:140] + ("..." if len(c) > 140 else ""))) for i, c in hits[:3]]
    return ChatResponse(reply=reply, sources=previews)
