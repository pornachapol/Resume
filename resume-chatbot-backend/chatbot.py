import os, re, time, io
from typing import List, Tuple, Optional, Dict, Any
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
RESUME_URL = os.getenv("RESUME_URL")
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS", "")).split(",") if o.strip()]
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set")
genai.configure(api_key=GOOGLE_API_KEY)

# ===================== APP =====================
app = FastAPI(title="Enhanced Resume Chatbot")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
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

# ===================== Enhanced Data Structures =====================
class ResumeChunk:
    def __init__(self, content: str, section: str = "general", metadata: Dict = None):
        self.content = content
        self.section = section
        self.metadata = metadata or {}
        
class ProfileData:
    def __init__(self):
        self.name_en = None
        self.name_th = None
        self.email = None
        self.phone = None
        self.location = None
        self.linkedin = None
        self.github = None
        self.skills = []
        self.experience = []
        self.education = []
        self.strengths = []
        self.achievements = []

# ===================== Global Variables =====================
RESUME_CHUNKS: List[ResumeChunk] = []
PROFILE: ProfileData = ProfileData()
VECTORIZER = None
TFIDF_MATRIX = None
EMB_MATRIX = None
LAST_FETCH_AT = 0

# ===================== Enhanced Parsing =====================
def parse_structured_resume(text: str) -> Tuple[ProfileData, List[ResumeChunk]]:
    """Parse resume with better structure awareness"""
    profile = ProfileData()
    chunks = []
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    current_section = "general"
    section_content = []
    
    # Section patterns
    section_patterns = {
        'basic_info': r'(basic information|personal|contact)',
        'summary': r'(professional summary|summary|profile)',
        'skills': r'(skills|expertise|competencies|technical)',
        'experience': r'(professional experience|experience|work history|career)',
        'education': r'(education|academic|qualification)',
        'achievements': r'(achievements|accomplishments|key results)',
        'strengths': r'(strengths|weaknesses)',
        'goals': r'(goals|objectives|future|passion)'
    }
    
    for line in lines:
        line_lower = line.lower()
        
        # Detect section changes
        new_section = None
        for section, pattern in section_patterns.items():
            if re.search(pattern, line_lower):
                new_section = section
                break
        
        if new_section:
            # Save previous section
            if section_content:
                chunk = ResumeChunk(
                    content='\n'.join(section_content),
                    section=current_section,
                    metadata={'line_count': len(section_content)}
                )
                chunks.append(chunk)
            
            current_section = new_section
            section_content = [line]
        else:
            section_content.append(line)
    
    # Save last section
    if section_content:
        chunk = ResumeChunk(
            content='\n'.join(section_content),
            section=current_section,
            metadata={'line_count': len(section_content)}
        )
        chunks.append(chunk)
    
    # Extract profile data
    full_text = text.lower()
    
    # Names
    name_match = re.search(r'name \(en\):\s*([^\n]+)', text, re.IGNORECASE)
    if name_match:
        profile.name_en = name_match.group(1).strip()
    
    th_name_match = re.search(r'ชื่อภาษาไทย:\s*([^\n]+)', text)
    if th_name_match:
        profile.name_th = th_name_match.group(1).strip()
    
    # Contact info
    email_match = re.search(r'email:\s*([^\s\n]+)', text, re.IGNORECASE)
    if email_match:
        profile.email = email_match.group(1).strip()
        
    phone_match = re.search(r'phone:\s*([^\s\n]+)', text, re.IGNORECASE)
    if phone_match:
        profile.phone = phone_match.group(1).strip()
        
    location_match = re.search(r'location:\s*([^\n]+)', text, re.IGNORECASE)
    if location_match:
        profile.location = location_match.group(1).strip()
    
    # Skills extraction (improved)
    skills_section = next((c.content for c in chunks if c.section == 'skills'), '')
    if skills_section:
        # Extract skills from bullet points and comma-separated lists
        skill_lines = re.findall(r'[•·-]\s*([^\n]+)', skills_section)
        for line in skill_lines:
            skills = re.split(r'[,/|]', line)
            profile.skills.extend([s.strip() for s in skills if s.strip()])
    
    return profile, chunks

# ===================== Enhanced Query Processing =====================
def expand_query_for_context(query: str) -> List[str]:
    """Expand query to catch more relevant context"""
    queries = [query]
    query_lower = query.lower()
    
    # Job suitability queries
    if any(word in query_lower for word in ['เหมาะ', 'suitable', 'fit', 'match']):
        if 'data' in query_lower:
            queries.extend([
                'data analysis skills experience',
                'SQL Python Power BI analytics',
                'business analysis reporting'
            ])
        if 'project' in query_lower:
            queries.extend([
                'project management experience',
                'leadership team management',
                'coordination stakeholder'
            ])
    
    # Skill-related queries
    if any(word in query_lower for word in ['skills', 'ทักษะ', 'ability']):
        queries.extend([
            'technical automation tools',
            'business process improvement',
            'leadership management'
        ])
    
    # Experience queries
    if any(word in query_lower for word in ['experience', 'ประสบการณ์', 'work']):
        queries.extend([
            'professional experience achievements',
            'manager supervisor role',
            'projects implementation'
        ])
    
    return queries

def multi_query_retrieval(query: str, k: int = 5) -> List[Tuple[int, str, float]]:
    """Enhanced retrieval with query expansion"""
    if not RESUME_CHUNKS or not VECTORIZER:
        return []
    
    all_results = {}  # chunk_idx -> max_score
    
    queries = expand_query_for_context(query)
    
    for q in queries:
        # TF-IDF retrieval
        qv = VECTORIZER.transform([q])
        tfidf_scores = cosine_similarity(qv, TFIDF_MATRIX).ravel()
        
        # Embedding retrieval
        emb_scores = np.zeros_like(tfidf_scores)
        if EMB_MATRIX is not None and EMB_MATRIX.size > 0:
            emb_q = embed_texts([q])
            if emb_q is not None and emb_q.size > 0:
                qn = emb_q / (np.linalg.norm(emb_q, axis=1, keepdims=True) + 1e-12)
                emb_scores = (EMB_MATRIX @ qn.T).ravel()
        
        # Hybrid scoring with section weighting
        for i, (tfidf_score, emb_score) in enumerate(zip(tfidf_scores, emb_scores)):
            # Normalize scores
            hybrid_score = 0.4 * tfidf_score + 0.6 * emb_score
            
            # Section-based boosting
            section = RESUME_CHUNKS[i].section
            if section in ['skills', 'experience', 'achievements']:
                hybrid_score *= 1.3
            elif section in ['summary', 'strengths']:
                hybrid_score *= 1.1
            
            # Keep max score across all queries
            all_results[i] = max(all_results.get(i, 0), hybrid_score)
    
    # Sort and return top k
    sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for idx, score in sorted_results[:k]:
        if score > 0.01:  # Minimum threshold
            results.append((idx, RESUME_CHUNKS[idx].content, score))
    
    return results

# ===================== Enhanced Response Generation =====================
def generate_enhanced_response(question: str, contexts: List[str]) -> str:
    """Generate response with clear fact vs analysis separation"""
    if not contexts:
        return "❌ ไม่พบข้อมูลที่เกี่ยวข้องในเรซูเม่ กรุณาระบุคำถามที่เฉพาะเจาะจงมากขึ้น"
    
    model = genai.GenerativeModel(MODEL_NAME)
    
    # Check if question requires analysis
    analysis_keywords = ['เหมาะ', 'suitable', 'จุดแข็ง', 'จุดอ่อน', 'strengths', 'weaknesses', 
                        'แนะนำ', 'recommend', 'ควร', 'should', 'เปรียบเทียบ', 'compare']
    
    needs_analysis = any(keyword in question.lower() for keyword in analysis_keywords)
    
    if needs_analysis:
        prompt = f"""
คุณเป็น AI ที่วิเคราะห์เรซูเม่อย่างชาญฉลาด

งานของคุณ:
1. ตอบจากข้อเท็จจริงในเรซูเม่เป็นหลัก
2. วิเคราะห์และให้ความเห็นที่อิงจากข้อมูลจริง
3. แยกชัดเจนระหว่าง "ข้อเท็จจริง" และ "การวิเคราะห์"

รูปแบบตอบ:
📋 **จากข้อมูลในเรซูเม่:**
[สรุปข้อเท็จจริงที่เกี่ยวข้อง]

💡 **การวิเคราะห์:**
[ความเห็นและข้อเสนอแนะที่อิงจากข้อเท็จจริงข้างต้น]

**บริบทจากเรซูเม่:**
{chr(10).join(contexts)}

**คำถาม:** {question}

กรุณาตอบเป็นภาษาไทยที่อ่านเข้าใจง่าย
"""
    else:
        prompt = f"""
คุณเป็นผู้ช่วยที่ตอบคำถามจากเรซูเม่

งานของคุณ:
- ตอบจากข้อมูลในเรซูเม่เป็นหลัก
- หากไม่มีข้อมูลตรงตัว ให้บอกว่า "ไม่ระบุในเรซูเม่"
- ตอบกระชับ ชัดเจน

**บริบทจากเรซูเม่:**
{chr(10).join(contexts)}

**คำถาม:** {question}

กรุณาตอบเป็นภาษาไทย
"""
    
    try:
        response = model.generate_content(prompt)
        answer = (getattr(response, "text", "") or "").strip()
        
        if not answer:
            return "❌ ไม่สามารถประมวลผลคำตอบได้ กรุณาลองใหม่"
            
        return answer
    except Exception as e:
        print(f"[generate_enhanced_response] error: {e}")
        return "❌ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่"

# ===================== Profile-based Quick Answers =====================
def get_quick_answer(question: str) -> Optional[str]:
    """Quick answers for basic profile questions"""
    q_lower = question.lower().replace(" ", "")
    
    # Name questions
    if any(k in q_lower for k in ["ชื่ออะไร", "ชื่อคืออะไร", "name", "fullname"]):
        if "thai" in q_lower or "ไทย" in q_lower:
            return f"ชื่อภาษาไทย: {PROFILE.name_th}" if PROFILE.name_th else None
        elif PROFILE.name_en:
            result = f"ชื่อ: {PROFILE.name_en}"
            if PROFILE.name_th:
                result += f" (ภาษาไทย: {PROFILE.name_th})"
            return result
    
    # Contact info
    if "email" in q_lower or "อีเมล" in q_lower:
        return f"อีเมล: {PROFILE.email}" if PROFILE.email else None
        
    if "phone" in q_lower or "โทร" in q_lower:
        return f"เบอร์โทร: {PROFILE.phone}" if PROFILE.phone else None
        
    if "location" in q_lower or "ที่อยู่" in q_lower:
        return f"ที่ตั้ง: {PROFILE.location}" if PROFILE.location else None
    
    return None

# ===================== Utility Functions =====================
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts using Google AI"""
    if not texts:
        return np.zeros((0, 1))
    try:
        resp = genai.embed_content(
            model="text-embedding-004",
            content=texts,
            task_type="retrieval_document"
        )
        
        embs = resp.get("embeddings") or resp.get("embedding")
        if isinstance(embs, list) and isinstance(embs[0], dict) and "values" in embs[0]:
            vecs = np.array([e["values"] for e in embs], dtype="float32")
        elif isinstance(embs, dict) and "values" in embs:
            vecs = np.array([embs["values"]], dtype="float32")
        else:
            vecs = np.array(embs, dtype="float32")
        
        return vecs
    except Exception as e:
        print(f"[embed_texts] error: {e}")
        return np.zeros((len(texts), 1), dtype="float32")

async def fetch_resume_text() -> str:
    """Fetch resume content from URL"""
    if not RESUME_URL:
        return ""
        
    headers = {"User-Agent": "resume-bot/1.0"}
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(RESUME_URL, follow_redirects=True, headers=headers)
            r.raise_for_status()
            
            content_type = (r.headers.get("Content-Type") or "").lower()
            
            if "application/pdf" in content_type or RESUME_URL.lower().endswith(".pdf"):
                doc = fitz.open(stream=r.content, filetype="pdf")
                pages = []
                for page in doc:
                    text = page.get_text()
                    if text.strip():
                        pages.append(text)
                return "\n".join(pages)
            else:
                soup = BeautifulSoup(r.text, "html.parser")
                return soup.get_text()
                
    except Exception as e:
        print(f"[fetch_resume_text] error: {e}")
        return ""

def build_search_index():
    """Build TF-IDF and embedding indices"""
    global VECTORIZER, TFIDF_MATRIX, EMB_MATRIX
    
    if not RESUME_CHUNKS:
        return
    
    contents = [chunk.content for chunk in RESUME_CHUNKS]
    
    # TF-IDF
    VECTORIZER = TfidfVectorizer(min_df=1, ngram_range=(1,2), max_features=1000)
    TFIDF_MATRIX = VECTORIZER.fit_transform(contents)
    
    # Embeddings
    EMB_MATRIX = embed_texts(contents)
    if EMB_MATRIX is not None and EMB_MATRIX.size > 0:
        norms = np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True) + 1e-12
        EMB_MATRIX = EMB_MATRIX / norms

async def ensure_data_loaded(force: bool = False):
    """Ensure resume data is loaded and indexed"""
    global PROFILE, RESUME_CHUNKS, LAST_FETCH_AT
    
    if RESUME_CHUNKS and not force and (time.time() - LAST_FETCH_AT < 3600):
        return
    
    text = await fetch_resume_text()
    if text:
        PROFILE, RESUME_CHUNKS = parse_structured_resume(text)
        build_search_index()
        LAST_FETCH_AT = time.time()
        print(f"Loaded {len(RESUME_CHUNKS)} chunks from resume")

# ===================== API Routes =====================
@app.get("/")
def home():
    return {"service": "enhanced-resume-chatbot", "status": "ready"}

@app.get("/health")
def health():
    return {"ok": True, "chunks": len(RESUME_CHUNKS)}

@app.get("/debug")
async def debug():
    await ensure_data_loaded()
    return {
        "chunks_count": len(RESUME_CHUNKS),
        "sections": [chunk.section for chunk in RESUME_CHUNKS],
        "profile": {
            "name_en": PROFILE.name_en,
            "name_th": PROFILE.name_th,
            "email": PROFILE.email,
            "skills_count": len(PROFILE.skills)
        }
    }

@app.post("/refresh")
async def refresh():
    await ensure_data_loaded(force=True)
    return {"ok": True, "chunks": len(RESUME_CHUNKS)}

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    await ensure_data_loaded()
    
    question = (req.message or "").strip()
    if not question:
        return ChatResponse(
            reply="โปรดพิมพ์คำถาม เช่น: ชื่ออะไร, ทักษะอะไรบ้าง, เหมาะกับตำแหน่ง Data Analyst ไหม",
            sources=[]
        )
    
    # Try quick answer first
    quick = get_quick_answer(question)
    if quick:
        return ChatResponse(reply=quick, sources=[])
    
    # Retrieve relevant contexts
    hits = multi_query_retrieval(question, k=5)
    contexts = [hit[1] for hit in hits[:3]]
    sources = [(hit[0], hit[1][:150] + "..." if len(hit[1]) > 150 else hit[1]) for hit in hits[:3]]
    
    # Generate response
    reply = generate_enhanced_response(question, contexts)
    
    return ChatResponse(reply=reply, sources=sources)

# ===================== Additional Helper Functions =====================

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    return re.sub(r'\s+', ' ', text or '').strip()

def is_summary_query(question: str) -> bool:
    """Check if question asks for overall summary"""
    q_lower = question.lower()
    summary_keywords = [
        'สรุป', 'ภาพรวม', 'overview', 'summary', 'profile',
        'แนะนำตัว', 'โดยรวม', 'เล่าเกี่ยวกับ'
    ]
    return any(keyword in q_lower for keyword in summary_keywords)

def generate_profile_summary() -> str:
    """Generate comprehensive profile summary"""
    if not RESUME_CHUNKS:
        return "❌ ไม่มีข้อมูลเรซูเม่ในระบบ"
    
    # Get summary content from all sections
    summary_content = []
    for chunk in RESUME_CHUNKS:
        if chunk.section in ['summary', 'basic_info', 'skills', 'experience', 'achievements']:
            summary_content.append(chunk.content)
    
    if not summary_content:
        summary_content = [chunk.content for chunk in RESUME_CHUNKS[:3]]
    
    model = genai.GenerativeModel(MODEL_NAME)
    prompt = f"""
สรุปเรซูเม่ต่อไปนี้ให้เป็นภาพรวมที่น่าสนใจและครบถ้วน:

รูปแบบที่ต้องการ:
👤 **ข้อมูลพื้นฐาน**
📊 **ความเชี่ยวชาญ** 
💼 **ประสบการณ์สำคัญ**
🎯 **จุดเด่น**

ข้อมูลจากเรซูเม่:
{chr(10).join(summary_content)}

กรุณาเขียนเป็นภาษาไทยที่อ่านง่าย กระชับแต่ครอบคลุม
"""
    
    try:
        response = model.generate_content(prompt)
        answer = (getattr(response, "text", "") or "").strip()
        return answer or "❌ ไม่สามารถสรุปข้อมูลได้"
    except Exception as e:
        print(f"[generate_profile_summary] error: {e}")
        return "❌ เกิดข้อผิดพลาดในการสรุปข้อมูล"

# ===================== Enhanced Chat Endpoint =====================
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    await ensure_data_loaded()
    
    question = clean_text(req.message)
    if not question:
        return ChatResponse(
            reply="👋 สวัสดีครับ! สามารถถามคำถามเกี่ยวกับเรซูเม่ได้เลย\n\nตัวอย่าง:\n• ชื่ออะไร?\n• มีทักษะอะไรบ้าง?\n• เหมาะกับตำแหน่ง Data Analyst ไหม?\n• สรุปเรซูเม่ให้หน่อย",
            sources=[]
        )
    
    # Check for summary query first
    if is_summary_query(question):
        summary = generate_profile_summary()
        return ChatResponse(reply=summary, sources=[])
    
    # Try quick answer for basic info
    quick = get_quick_answer(question)
    if quick:
        return ChatResponse(reply=quick, sources=[])
    
    # Enhanced retrieval
    hits = multi_query_retrieval(question, k=5)
    contexts = [hit[1] for hit in hits[:3]]
    
    # Prepare source previews
    sources = []
    for i, (chunk_idx, content, score) in enumerate(hits[:3]):
        preview = content[:150].replace('\n', ' ') + "..." if len(content) > 150 else content
        section = RESUME_CHUNKS[chunk_idx].section if chunk_idx < len(RESUME_CHUNKS) else "unknown"
        sources.append((chunk_idx, f"[{section.title()}] {preview}"))
    
    # Generate enhanced response
    reply = generate_enhanced_response(question, contexts)
    
    # Add contact info footer for analysis responses
    if "💡" in reply and (PROFILE.email or PROFILE.phone):
        contact_info = []
        if PROFILE.email:
            contact_info.append(f"📧 {PROFILE.email}")
        if PROFILE.phone:
            contact_info.append(f"📱 {PROFILE.phone}")
        
        if contact_info:
            reply += f"\n\n---\n💬 **ติดต่อเพิ่มเติม:** {' | '.join(contact_info)}"
    
    return ChatResponse(reply=reply, sources=sources)

# ===================== Additional API Endpoints =====================
@app.get("/profile")
async def get_profile():
    """Get structured profile data"""
    await ensure_data_loaded()
    return {
        "name_en": PROFILE.name_en,
        "name_th": PROFILE.name_th,
        "contact": {
            "email": PROFILE.email,
            "phone": PROFILE.phone,
            "location": PROFILE.location
        },
        "skills": PROFILE.skills[:20],  # Top 20 skills
        "sections": {
            section: len([c for c in RESUME_CHUNKS if c.section == section])
            for section in set(chunk.section for chunk in RESUME_CHUNKS)
        }
    }

@app.get("/search/{query}")
async def search_resume(query: str):
    """Search in resume content"""
    await ensure_data_loaded()
    hits = multi_query_retrieval(query, k=5)
    return {
        "query": query,
        "results": [
            {
                "chunk_id": chunk_idx,
                "section": RESUME_CHUNKS[chunk_idx].section if chunk_idx < len(RESUME_CHUNKS) else "unknown",
                "content": content[:300] + "..." if len(content) > 300 else content,
                "score": float(score)
            }
            for chunk_idx, content, score in hits
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
