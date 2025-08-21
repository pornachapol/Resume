# chatbot.py (Backend)
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

# ===== CONFIG =====
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RESUME_URL     = os.getenv("RESUME_URL")   # ex: https://yourdomain.com/resume
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS","")).split(",") if o.strip()]
MODEL_NAME     = os.getenv("MODEL_NAME", "gemini-1.5-flash")  # หรือ flash-2.0 ถ้ามีสิทธิ์

genai.configure(api_key=GOOGLE_API_KEY)

# ===== FastAPI App =====
app = FastAPI(title="Resume Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    sources: List[Tuple[int, str]] = []

# ===== Text Index =====
CHUNKS: List[str] = []
VECTORIZER: TfidfVectorizer | None = None
MATRIX = None
LAST_FETCH_AT = 0

def clean(txt): return re.sub(r"\s+"," ",txt or "").strip()

def html_to_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script","style","noscript"]): t.decompose()
    return clean(soup.get_text(" "))

def chunk(text: str, size=1200, overlap=200):
    out=[]; i=0
    while i < len(text):
        out.append(text[i:i+size]); i+=(size-overlap)
    return out

async def fetch_resume():
    async with httpx.AsyncClient(timeout=60) as client:
        r=await client.get(RESUME_URL, follow_redirects=True)
        r.raise_for_status()
        return r.text

def build_index(chunks: List[str]):
    global VECTORIZER,MATRIX
    VECTORIZER=TfidfVectorizer(min_df=1,ngram_range=(1,2))
    MATRIX=VECTORIZER.fit_transform(chunks)

async def ensure_index(force=False):
    global CHUNKS,LAST_FETCH_AT
    if CHUNKS and not force and (time.time()-LAST_FETCH_AT<3600): return
    html=await fetch_resume()
    text=html_to_text(html)
    CHUNKS=chunk(text) if text else []
    if CHUNKS: build_index(CHUNKS)
    LAST_FETCH_AT=time.time()

def search(q,k=5):
    if not CHUNKS or not VECTORIZER: return []
    qv=VECTORIZER.transform([q])
    sims=cosine_similarity(qv,MATRIX).ravel()
    order=sims.argsort()[::-1][:k]
    return [(int(i),CHUNKS[int(i)]) for i in order if sims[int(i)]>0.01]

def ask_gemini(question,contexts):
    ctx="\n\n".join(contexts)
    prompt=f"""
คุณเป็นผู้ช่วยตอบคำถามจากเรซูเม่เท่านั้น
คำถาม: {question}

บริบทจากเรซูเม่:
{ctx}

ห้ามเดา ถ้าไม่พบข้อมูล ให้ตอบว่า 'ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่ของฉัน'
ตอบเป็นภาษาไทยสั้น กระชับ
"""
    model=genai.GenerativeModel(MODEL_NAME)
    resp=model.generate_content(prompt)
    return resp.text

@app.get("/health")
def health(): return {"ok":True}

@app.post("/chat",response_model=ChatResponse)
async def chat(req:ChatRequest):
    await ensure_index()
    hits=search(req.message,k=5)
    if not hits: 
        return ChatResponse(reply="ขออภัย ไม่พบข้อมูลนี้ในเรซูเม่าของฉัน",sources=[])
    contexts=[c for _,c in hits][:3]
    reply=ask_gemini(req.message,contexts)
    previews=[(i,(c[:140]+"...")) for i,c in hits[:3]]
    return ChatResponse(reply=reply,sources=previews)
