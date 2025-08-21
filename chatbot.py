# chatbot.py
import os, re, time
import httpx
import asyncio
from typing import List, Tuple
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ====== CONFIG ======
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
RESUME_URL = os.getenv("RESUME_URL", "https://your-domain.com/resume")  # <- ตั้งลิงก์หน้าเรซูเม่
ALLOWED_ORIGINS = [
    "https://your-domain.com",     # <- แก้เป็นโดเมนหน้าเว็บของคุณ
    "http://localhost:5500",
]

# ====== APP ======
app = FastAPI(title="Resume QA Bot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    sources: List[Tuple[int, str]] = []  # (chunk_index, preview)

# ====== Simple in-memory store ======
CHUNKS: List[str] = []
VECTORIZER: TfidfVectorizer = None
MATRIX = None
LAST_FETCH_AT = 0

# ====== Utils ======
def clean_text(txt: str) -> str:
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()

def html_to_visible_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")

    # ตัด script/style/nav/footer ที่มักไม่ใช่เนื้อหา
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # เก็บ text
    text = soup.get_text(separator=" ")
    return clean_text(text)

def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> List[str]:
    out = []
    i = 0
    while i < len(text):
        out.append(text[i:i+size])
        i += (size - overlap)
    return out

async def fetch_resume() -> str:
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(RESUME_URL, follow_redirects=True)
        r.raise_for_status()
        return r.text

def build_index(chunks: List[str]):
    global VECTORIZER, MATRIX
    VECTORIZER = TfidfVectorizer(min_df=1, ngram_range=(1,2))
    MATRIX = VECTORIZER.fit_transform(chunks)

def retrieve(query: str, k: int = 5) -> List[Tuple[int, str]]:
    if not CHUNKS:
        return []
    qv = VECTORIZER.transform([query])
    sims = cosine_similarity(qv, MATRIX).ravel()
    topk = sims.argsort()[::-1][:k]
    return [(int(i), CHUNKS[int(i)]) for i in topk if sims[int(i)] > 0.01]

async def ensure_index(force: bool = False):
    global CHUNKS, LAST_FETCH_AT
    if CHUNKS and not force and (time.time() - LAST_FETCH_AT < 3600):
        return
    html = await fetch_resume()
    text = html_to_visible_text(html)
    CHUNKS = chunk_text(text, size=1200, overlap=200)
    build_index(CHUNKS)
    LAST_FETCH_AT = time.time()

def build_prompt(question: str, contexts: List[str]) -> List[dict]:
    rules = (
        "คุณเป็นผู้ช่วยตอบคำถามเกี่ยวกับเรซูเม่ **โดยยึดตามบริบทที่ให้เท่านั้น** "
        "ห้ามเดา ห้ามเพิ่มข้อมูลนอกเหนือจากบริบท ถ้าไม่พบข้อมูล ให้ตอบว่า "
        "“ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่ของฉัน” และแนะนำหัวข้อที่มีให้ค้นหาต่อได้อย่างสุภาพ"
    )
    context_block = "\n\n---\n".join(contexts) if contexts else ""
    user_msg = (
        f"คำถาม: {question}\n\n"
        f"นี่คือบริบทจากหน้าเรซูเม่ (บางส่วนที่เกี่ยวข้อง):\n---\n{context_block}\n---\n\n"
        "โปรดตอบเป็นภาษาไทย กระชับ ชัด ใช้เฉพาะข้อมูลในบริบทเท่านั้น"
    )
    return [
        {"name": "system", "role": "system", "content": rules},
        {"role": "user", "content": user_msg},
    ]

async def ask_llm(messages: List[dict]) -> str:
    if not OPENAI_API_KEY:
        return "Server error: OPENAI_API_KEY not set."
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": MODEL, "messages": messages, "temperature": 0.2, "max_tokens": 400}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

# ====== Routes ======
@app.on_event("startup")
async def warmup():
    try:
        await ensure_index(force=True)
    except Exception as e:
        print("Warmup error:", e)

@app.get("/refresh")
async def refresh():
    await ensure_index(force=True)
    return {"ok": True, "chunks": len(CHUNKS)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    await ensure_index()
    hits = retrieve(req.message, k=5)
    contexts = [c for _, c in hits][:3]  # ให้ 2–3 บล็อกพอ เพื่อโฟกัส
    msgs = build_prompt(req.message, contexts)
    reply = await ask_llm(msgs)

    # ถ้า LLM เผลอหลุด ให้กรอง fallback
    if not contexts or ("ไม่พบ" in reply and "เรซูเม่" in reply):
        # โครงสร้าง fallback ชัดเจน
        reply = "ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่ของฉัน"

    # แนบตัวอย่าง source preview สั้นๆ
    previews = []
    for idx, chunk in hits[:3]:
        previews.append((idx, chunk[:140] + ("..." if len(chunk) > 140 else "")))

    return ChatResponse(reply=reply, sources=previews)
