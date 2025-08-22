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
ALLOW_OPINION = (os.getenv("ALLOW_OPINION", "false").lower() == "true")


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

# ===================== Helpers =====================
# ===== Helpers needed by /chat =====
from typing import Optional

# โครง PROFILE พื้นฐาน (ป้องกัน KeyError)
PROFILE = {
    "name": None,
    "name_th": None,
    "contacts": {"email": None, "phone": None, "location": None, "links": []},
    "skills": [],
    "experience": [],
    "education": [],
    "etc": []
}

def is_summary_query(q: str) -> bool:
    """
    ตรวจว่าคำถามเป็นแนว 'สรุปภาพรวม' / 'overview'
    """
    ql = (q or "").lower()
    keys = [
        "สรุป", "เล่าภาพรวม", "overview", "summary",
        "โดยรวม", "แนะนำตัว", "ภาพรวม", "แนะนำ", "สรุปเรซูเม่", "ย่อ", "profile"
    ]
    return any(k in ql for k in keys)

def opinion_footer() -> str:
    """
    สร้าง footer ช่องทางติดต่อจาก PROFILE จริง
    """
    c = PROFILE.get("contacts", {}) if isinstance(PROFILE, dict) else {}
    parts = []
    if c.get("email"):
        parts.append(f"อีเมล: {c['email']}")
    if c.get("phone"):
        parts.append(f"โทร: {c['phone']}")
    tail = "💡 หากต้องการข้อมูลเพิ่มเติม กรุณาติดต่อเจ้าของเรซูเม่โดยตรง"
    if parts:
        tail += " — " + " | ".join(parts)
    return tail

def answer_from_profile(q: str) -> Optional[str]:
    """
    ตอบจาก PROFILE โดยตรงสำหรับคำถามพื้นฐาน (รวดเร็ว ไม่ต้องเรียกโมเดล)
    รองรับทั้งไทย/อังกฤษ และกรณีระบุชื่อภาษาไทย
    """
    ql = (q or "").strip().lower()
    ql_nospace = ql.replace(" ", "")

    c = PROFILE.get("contacts", {}) if isinstance(PROFILE, dict) else {}
    skills = PROFILE.get("skills") or []
    exp = PROFILE.get("experience") or []
    edu = PROFILE.get("education") or []
    links = c.get("links") or []

    def find_link(domain_keyword: str) -> Optional[str]:
        for u in links:
            if domain_keyword in (u or "").lower():
                return u
        return None

    # ชื่อภาษาไทยก่อน
    if any(k in ql for k in ["ชื่อภาษาไทย", "ชื่อไทย", "thai name", "thai fullname"]):
        if PROFILE.get("name_th"):
            return f"ชื่อภาษาไทย: {PROFILE['name_th']}"
        if PROFILE.get("name"):
            return f"ชื่อ-นามสกุล: {PROFILE['name']}"

    # ชื่อทั่วไป
    if any(k in ql for k in ["ชื่ออะไร", "ชื่อคืออะไร", "ชื่อจริง", "นามสกุล", "ชื่อ", "full name", "your name", "name", "name?"]) \
       or "ชื่ออะไร" in ql_nospace:
        if PROFILE.get("name"):
            if PROFILE.get("name_th"):
                return f"ชื่อ-นามสกุล: {PROFILE['name']} (TH: {PROFILE['name_th']})"
            return f"ชื่อ-นามสกุล: {PROFILE['name']}"

    # Contact เฉพาะฟิลด์
    if any(k in ql for k in ["email", "อีเมล", "อีเมล์"]):
        if c.get("email"): return f"อีเมล: {c['email']}"
    if any(k in ql for k in ["phone", "โทร", "เบอร์", "เบอร์โทร", "เบอร์โทรศัพท์"]):
        if c.get("phone"): return f"เบอร์โทร: {c['phone']}"
    if any(k in ql for k in ["location", "ที่อยู่", "อยู่ที่ไหน", "อยู่จังหวัดอะไร"]):
        if c.get("location"): return f"ที่ตั้ง: {c['location']}"
    if "linkedin" in ql:
        li = find_link("linkedin")
        if li: return f"LinkedIn: {li}"
    if "github" in ql:
        gh = find_link("github")
        if gh: return f"GitHub: {gh}"

    # Contact สรุปรวม
    if any(k in ql for k in ["ติดต่อ", "contact", "ช่องทางติดต่อ", "คอนแทค"]):
        parts = []
        if c.get("email"): parts.append(f"อีเมล: {c['email']}")
        if c.get("phone"): parts.append(f"โทร: {c['phone']}")
        if c.get("location"): parts.append(f"ที่ตั้ง: {c['location']}")
        li = find_link("linkedin")
        gh = find_link("github")
        if li: parts.append(f"LinkedIn: {li}")
        if gh: parts.append(f"GitHub: {gh}")
        if parts: return " | ".join(parts)

    # Skills
    if any(k in ql for k in ["ทักษะ", "skills", "skill", "สกิล"]):
        if skills:
            if any(k in ql for k in ["หลัก", "top", "เด่น", "core"]):
                return "ทักษะหลัก: " + ", ".join(skills[:10])
            return "ทักษะ: " + ", ".join(skills[:30])

    # Experience
    if any(k in ql for k in ["ประสบการณ์", "experience", "เคยทำงาน", "ทำงานที่ไหน", "งานที่ผ่านมา"]):
        if exp:
            return "ประสบการณ์ (สรุป):\n- " + "\n- ".join(exp[:8])

    # Education
    if any(k in ql for k in ["การศึกษา", "education", "เรียนที่ไหน", "จบจาก", "วุฒิการศึกษา"]):
        if edu:
            return "การศึกษา:\n- " + "\n- ".join(edu[:8])

    return None

# ====== Opinion / Summary helpers ======
def ask_opinion(question: str) -> str:
    """
    ใช้เมื่อตอบจาก PDF ไม่ได้จริง ๆ และ ALLOW_OPINION=true
    จะตอบเป็นมุมมองทั่วไปที่ไม่อ้างข้อเท็จจริงเฉพาะเจาะจง
    """
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
คุณจะตอบ 'ความเห็นส่วนตัว (Opinion)' ต่อคำถามด้านล่างนี้
กติกา:
- หลีกเลี่ยงการอ้างชื่อ/องค์กร/วันที่/ตัวเลขเฉพาะเจาะจง
- ให้เป็นแนวปฏิบัติ/แนวคิดทั่วไปที่เป็นประโยชน์
- เขียนสั้น กระชับ เป็นข้อ ๆ ได้ยิ่งดี
- เริ่มด้วย "ความเห็นส่วนตัว (Opinion):"

[คำถาม]
{question}
"""
    try:
        resp = model.generate_content(prompt)
        ans = (getattr(resp, "text", "") or "").strip()
        return ans or "ความเห็นส่วนตัว (Opinion): คำถามนี้อยู่นอกเหนือ PDF จึงขอตอบเชิงมุมมองทั่วไป"
    except Exception as e:
        print(f"[ask_opinion] error: {e}")
        return "ความเห็นส่วนตัว (Opinion): ขัดข้องชั่วคราว จึงตอบเชิงมุมมองทั่วไป"

def summarize_all_chunks(chunks: List[str]) -> str:
    """
    สรุปทั้งเอกสารแบบ map-reduce:
    - มีเนื้อหา → ส่งสรุปโดยใช้คำว่า 'ข้อมูลในระบบ'
    - ไม่มีเนื้อหา → ถ้า ALLOW_OPINION true → ความเห็นส่วนตัว (Opinion)
    """
    if not chunks:
        if ALLOW_OPINION:
            return ask_opinion("ช่วยสรุปภาพรวมโปรไฟล์/เรซูเม่แบบทั่วไป") + "\n\n" + opinion_footer()
        return "คำถามนี้ไม่มีอยู่ในข้อมูลในระบบ"

    model = genai.GenerativeModel(MODEL_NAME)

    partials = []
    for c in chunks[:8]:
        try:
            r = model.generate_content(
                "สรุปสาระสำคัญของข้อความนี้เป็น bullet ภาษาไทยสั้น ๆ "
                "โดยห้ามขึ้นต้นด้วยคำว่า 'สรุปจากข้อมูลในระบบ' หรือคำคล้ายกัน:\n\n" + c
            )
            partials.append((r.text or "").strip())
        except Exception as e:
            print(f"[summarize] chunk error: {e}")

    joined = "\n".join([p for p in partials if p])
    if not joined:
        if ALLOW_OPINION:
            return ask_opinion("ช่วยสรุปภาพรวมโปรไฟล์/เรซูเม่แบบทั่วไป") + "\n\n" + opinion_footer()
        return "คำถามนี้ไม่มีอยู่ในข้อมูลในระบบ"

    try:
        r2 = model.generate_content(
            "รวมสรุปต่อไปนี้ให้เป็นภาพรวมอ่านง่าย ไม่ซ้ำซ้อน เป็นภาษาไทย "
            "และอย่าใส่คำขึ้นต้นเช่น 'สรุปจากข้อมูลในระบบ':\n\n" + joined
        )
        final_sum = (r2.text or "").strip()

        # cleanup prefix ถ้าโมเดลเผลอใส่มา
        if final_sum:
            bad_prefixes = [
                "สรุปจากข้อมูลในระบบ:", "สรุปจากข้อมูลในระบบ",
                "จากข้อมูลในระบบ:", "จากข้อมูลในระบบ"
            ]
            fs_lower = final_sum.lower().lstrip()
            for bp in bad_prefixes:
                if fs_lower.startswith(bp):
                    cut_len = len(final_sum) - len(fs_lower)
                    final_sum = final_sum[cut_len + len(bp):].lstrip(": \n-")
                    break

        if not final_sum:
            if ALLOW_OPINION:
                return ask_opinion("ช่วยสรุปภาพรวมโปรไฟล์/เรซูเม่แบบทั่วไป") + "\n\n" + opinion_footer()
            return "คำถามนี้ไม่มีอยู่ในข้อมูลในระบบ"

        return final_sum

    except Exception as e:
        print(f"[summarize reduce] error: {e}")
        if ALLOW_OPINION:
            return ask_opinion("ช่วยสรุปภาพรวมโปรไฟล์/เรซูเม่แบบทั่วไป") + "\n\n" + opinion_footer()
        return "คำถามนี้ไม่มีอยู่ในข้อมูลในระบบ"

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
    # Experience / Career
    r"\b(PROFESSIONAL EXPERIENCE|EXPERIENCE|WORK EXPERIENCE|CAREER HISTORY|EMPLOYMENT HISTORY|WORK HISTORY|PROFESSIONAL BACKGROUND)\b",

    # Education / Certifications
    r"\b(EDUCATION|ACADEMIC BACKGROUND|QUALIFICATIONS|CERTIFICATION|CERTIFICATIONS|TRAINING|COURSES|CREDENTIALS)\b",

    # Achievements / Awards
    r"\b(KEY ACHIEVEMENTS|ACHIEVEMENTS|AWARDS|HONORS|DISTINCTIONS|RECOGNITION|MILESTONES|ACCOMPLISHMENTS)\b",

    # Skills / Expertise
    r"\b(AREA OF EXPERTISE|SKILLS?|TECHNICAL SKILLS|CORE COMPETENCIES|EXPERTISE|SPECIALTIES|CAPABILITIES|PROFICIENCIES|KNOWLEDGE AREAS)\b",

    # Languages & Additional Info
    r"\b(ADDITIONAL INFORMATION|LANGUAGES?|HOBBIES|INTERESTS|PERSONAL DETAILS|CONTACT INFORMATION|PROFILE|SUMMARY|OVERVIEW)\b",

    # Projects
    r"\b(PROJECTS|KEY PROJECTS|SELECTED PROJECTS|PROJECT EXPERIENCE|CASE STUDIES)\b",

    # Objectives / Goals
    r"\b(CAREER OBJECTIVE|OBJECTIVES|GOAL|FUTURE GOALS|PROFESSIONAL GOALS|MISSION|VISION|PASSION)\b",

    # Strengths / Weaknesses / Self
    r"\b(STRENGTHS|WEAKNESSES|AREAS FOR IMPROVEMENT|PERSONAL ATTRIBUTES|CHARACTERISTICS|VALUES)\b",

    # Frameworks / Methods
    r"\b(METHODOLOGIES|FRAMEWORKS|TOOLS|PROCESS IMPROVEMENT|LEAN SIX SIGMA|PRA|SWOT|FISHBONE|IPSO|PDCA)\b",
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



    # -------- Name (TH/EN) --------
    # กรณีระบุ "ชื่อภาษาไทย" / "ชื่อไทย" ก่อน
    if any(k in ql for k in ["ชื่อภาษาไทย", "ชื่อไทย", "thai name", "thai fullname"]):
        if PROFILE.get("name_th"):
            return f"ชื่อภาษาไทย: {PROFILE['name_th']}"
        # ไม่มี name_th ก็ fallback เป็น EN
        if PROFILE.get("name"):
            return f"ชื่อ-นามสกุล: {PROFILE['name']}"

    # คำถามทั่วไปเรื่องชื่อ
    if any(k in ql for k in [
        "ชื่ออะไร", "ชื่อคืออะไร", "ชื่อจริง", "นามสกุล", "ชื่อ", 
        "full name", "your name", "name", "name?"
    ]) or "ชื่ออะไร" in ql_nospace:
        if PROFILE.get("name"):
            # ถ้ามีชื่อไทยด้วย ให้บอกทั้งคู่
            if PROFILE.get("name_th"):
                return f"ชื่อ-นามสกุล: {PROFILE['name']} (TH: {PROFILE['name_th']})"
            return f"ชื่อ-นามสกุล: {PROFILE['name']}"

    # -------- Contacts (ตอบเฉพาะสิ่งที่ถาม) --------
    # Email only
    if any(k in ql for k in ["email", "อีเมล", "อีเมล์"]):
        if c.get("email"):
            return f"อีเมล: {c['email']}"

    # Phone only
    if any(k in ql for k in ["phone", "โทร", "เบอร์", "เบอร์โทร", "เบอร์โทรศัพท์"]):
        if c.get("phone"):
            return f"เบอร์โทร: {c['phone']}"

    # Location only
    if any(k in ql for k in ["location", "ที่อยู่", "อยู่ที่ไหน", "อยู่จังหวัดอะไร"]):
        if c.get("location"):
            return f"ที่ตั้ง: {c['location']}"

    # LinkedIn only
    if "linkedin" in ql:
        li = find_link("linkedin")
        if li:
            return f"LinkedIn: {li}"

    # GitHub only
    if "github" in ql:
        gh = find_link("github")
        if gh:
            return f"GitHub: {gh}"

    # Contact summary
    if any(k in ql for k in ["ติดต่อ", "contact", "ช่องทางติดต่อ", "คอนแทค"]):
        parts = []
        if c.get("email"): parts.append(f"อีเมล: {c['email']}")
        if c.get("phone"): parts.append(f"โทร: {c['phone']}")
        if c.get("location"): parts.append(f"ที่ตั้ง: {c['location']}")
        li = find_link("linkedin")
        gh = find_link("github")
        if li: parts.append(f"LinkedIn: {li}")
        if gh: parts.append(f"GitHub: {gh}")
        if parts:
            return " | ".join(parts)

    # -------- Skills --------
    if any(k in ql for k in ["ทักษะ", "skills", "skill", "สกิล"]):
        if skills:
            # ถ้าถามว่า top/หลัก ให้ตัดเหลือสั้น ๆ
            if any(k in ql for k in ["หลัก", "top", "เด่น", "core"]):
                return "ทักษะหลัก: " + ", ".join(skills[:10])
            return "ทักษะ: " + ", ".join(skills[:30])

    # -------- Experience --------
    if any(k in ql for k in ["ประสบการณ์", "experience", "เคยทำงาน", "ทำงานที่ไหน", "งานที่ผ่านมา"]):
        if exp:
            # แสดงไม่เกิน 8 บรรทัด พอให้อ่านง่าย
            return "ประสบการณ์ (สรุป):\n- " + "\n- ".join(exp[:8])

    # -------- Education --------
    if any(k in ql for k in ["การศึกษา", "education", "เรียนที่ไหน", "จบจาก", "วุฒิการศึกษา"]):
        if edu:
            return "การศึกษา:\n- " + "\n- ".join(edu[:8])

    return None

def ask_gemini(question: str, contexts: List[str]) -> str:
    """
    พยายามตอบจากบริบท (PDF) ก่อน
    - ถ้าพบข้อมูล → ตอบปกติ และให้ขึ้นต้นด้วย 'คำตอบจาก PDF:' เพื่อความชัดเจน
    - ถ้าไม่พบข้อมูลในบริบท → ให้ตอบแบบ 'ความเห็นส่วนตัว (Opinion): ...'
      โดยหลีกเลี่ยงการระบุชื่อ/ตัวเลข/วันที่จำเพาะเจาะจง
    """
    model = genai.GenerativeModel(MODEL_NAME)
    ctx = "\n\n---\n".join(contexts) if contexts else ""
    prompt = f"""
คุณเป็นผู้ช่วยที่ตอบคำถามจากไฟล์เรซูเม่ (PDF) เป็นหลัก

กฎสำคัญ:
- ใช้เฉพาะข้อมูลใน [บริบท] เมื่อตอบข้อเท็จจริงเกี่ยวกับผู้สมัคร
- ถ้าคำตอบ "ไม่มีอยู่ในบริบท" ให้ตอบเป็น 'ความเห็นส่วนตัว (Opinion): ...'
  โดยเป็นคำแนะนำ/มุมมองทั่วไป หลีกเลี่ยงการสร้างชื่อ-วันเวลา-ตัวเลขที่เฉพาะเจาะจง
- ห้ามทำให้ผู้อ่านสับสนว่าเป็นข้อเท็จจริงจาก PDF

รูปแบบคำตอบ:
- ถ้าพบข้อมูลใน PDF: เริ่มด้วย "คำตอบจาก PDF: ..."
- ถ้าไม่พบข้อมูลใน PDF: เริ่มด้วย "ความเห็นส่วนตัว (Opinion): ..."

[บริบทจาก PDF]
{ctx}

[คำถาม]
{question}

โปรดตอบเป็นภาษาไทย กระชับ ชัดเจน
"""
    try:
        resp = model.generate_content(prompt)
        ans = (getattr(resp, "text", "") or "").strip()
        # safety net: ถ้าโมเดลเงียบ ให้ใส่ opinion ตาม flag
        if not ans:
            return "ความเห็นส่วนตัว (Opinion): คำถามนี้ไม่มีใน PDF จึงขอตอบเชิงมุมมองทั่วไปแบบไม่อ้างอิงข้อเท็จจริงเฉพาะเจาะจง"
        return ans
    except Exception as e:
        print(f"[ask_gemini] error: {e}")
        return "ความเห็นส่วนตัว (Opinion): เกิดข้อผิดพลาดขณะประมวลผล จึงตอบเป็นมุมมองทั่วไปโดยไม่อ้างอิงข้อเท็จจริงเฉพาะเจาะจง"

def opinion_footer() -> str:
    c = PROFILE.get("contacts", {}) if isinstance(PROFILE, dict) else {}
    email = c.get("email")
    phone = c.get("phone")
    tail = "💡 หากต้องการข้อมูลเพิ่มเติม กรุณาติดต่อเจ้าของเรซูเม่โดยตรง"
    ways = []
    if email: ways.append(f"อีเมล: {email}")
    if phone: ways.append(f"โทร: {phone}")
    if ways:
        tail += " — " + " | ".join(ways)
    return tail


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

    # กันเคสข้อความว่าง
    text_q = (req.message or "").strip()
    if not text_q:
        return ChatResponse(
            reply="โปรดพิมพ์คำถาม (เช่น: ชื่ออะไร, อีเมลคืออะไร, สรุปเรซูเม่)",
            sources=[]
        )

    # 1) ตอบจาก PROFILE ก่อน (ชื่อ/อีเมล/เบอร์/ทักษะฯลฯ)
    direct = answer_from_profile(text_q)
    if direct:
        return ChatResponse(reply=direct, sources=[])

    # 2) คำถามสรุป/ปลายเปิด → summarize จาก PDF
    if is_summary_query(text_q):
        reply = summarize_all_chunks(CHUNKS)
        return ChatResponse(reply=reply, sources=[])

    # 3) Retrieval ปกติ (ถ้ามี retrieve_hybrid ให้ใช้ก่อน)
    q_norm = normalize_query(text_q)
    try:
        hits = retrieve_hybrid(q_norm, k=5, alpha=0.6)  # ถ้ามีฟังก์ชันนี้
        # hits: [(idx, chunk, score), ...]
        contexts = [h[1] for h in hits[:3] if len(h) >= 2]
        previews = [(h[0], (h[1][:140] + ("..." if len(h[1]) > 140 else ""))) for h in hits[:3] if len(h) >= 2]
    except NameError:
        hits = retrieve(q_norm, k=5)  # โหมดเดิม
        # hits: [(idx, chunk), ...]
        contexts = [c for _, c in hits[:3]]
        previews = [(i, (c[:140] + ("..." if len(c) > 140 else ""))) for i, c in hits[:3]]

    if contexts:
        reply = ask_gemini(text_q, contexts)
        # ถ้าโมเดลตอบเป็น Opinion (ไม่ได้ยึด PDF) → เติมช่องทางติดต่อจริง
        if reply.startswith("ความเห็นส่วนตัว (Opinion):"):
            reply += "\n\n" + opinion_footer()
    else:
        # ❗ ไม่มีคอนเท็กซ์จาก PDF
        if ALLOW_OPINION:
            reply = ask_opinion(text_q)
            reply += "\n\n" + opinion_footer()
        else:
            reply = "คำถามนี้ไม่มีอยู่ใน PDF"

    return ChatResponse(reply=reply, sources=previews)

