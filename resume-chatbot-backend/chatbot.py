import os, re, time, io
from typing import List, Tuple, Optional, Dict, Any
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai
import fitz  # PyMuPDF
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== ENV =====================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
RESUME_URL = os.getenv("RESUME_URL")
ALLOWED_ORIGINS = [o.strip() for o in (os.getenv("ALLOWED_ORIGINS", "")).split(",") if o.strip()]
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.0-flash")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not set")
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Google AI configured successfully")
    except Exception as e:
        logger.error(f"Failed to configure Google AI: {e}")

# ===================== APP =====================
app = FastAPI(title="Enhanced Resume Chatbot", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===================== Schemas =====================
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)

class ChatResponse(BaseModel):
    reply: str
    sources: List[Tuple[int, str]] = []
    opinion_enabled: bool = True  # เพิ่ม flag สำหรับ opinion mode

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
    
    try:
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
        
        # Extract profile data safely
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
        
        logger.info(f"Successfully parsed resume: {len(chunks)} chunks, profile name: {profile.name_en}")
        return profile, chunks
        
    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
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
        logger.warning("No resume chunks or vectorizer available")
        return []
    
    all_results = {}  # chunk_idx -> max_score
    
    try:
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
        
        logger.info(f"Retrieved {len(results)} relevant chunks for query: {query}")
        return results
        
    except Exception as e:
        logger.error(f"Error in multi_query_retrieval: {e}")
        return []

# ===================== Enhanced Response Generation =====================
def generate_enhanced_response(question: str, contexts: List[str], enable_opinion: bool = True) -> str:
    """Generate response with opinion control and better error handling"""
    if not contexts:
        return "❌ ไม่พบข้อมูลที่เกี่ยวข้องในเรซูเม่ กรุณาระบุคำถามที่เฉพาะเจาะจงมากขึ้น"
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Check if question requires analysis
        analysis_keywords = ['เหมาะ', 'suitable', 'จุดแข็ง', 'จุดอ่อน', 'strengths', 'weaknesses', 
                            'แนะนำ', 'recommend', 'ควร', 'should', 'เปรียบเทียบ', 'compare']
        
        needs_analysis = any(keyword in question.lower() for keyword in analysis_keywords)
        
        if needs_analysis and enable_opinion:
            prompt = f"""
คุณเป็น AI ที่วิเคราะห์เรซูเม่อย่างชาญฉลาด

**สำคัญ: คุณสามารถให้ความเห็นและคำแนะนำได้ตามที่ต้องการ**

งานของคุณ:
1. ตอบจากข้อเท็จจริงในเรซูเม่เป็นหลัก
2. วิเคราะห์และให้ความเห็นที่อิงจากข้อมูลจริง
3. แยกชัดเจนระหว่าง "ข้อเท็จจริง" และ "การวิเคราะห์"
4. ให้คำแนะนำที่เป็นประโยชน์

รูปแบบตอบ:
📋 **จากข้อมูลในเรซูเม่:**
[สรุปข้อเท็จจริงที่เกี่ยวข้อง]

💡 **การวิเคราะห์และความเห็น:**
[ความเห็นและข้อเสนอแนะที่อิงจากข้อเท็จจริงข้างต้น]

**บริบทจากเรซูเม่:**
{chr(10).join(contexts)}

**คำถาม:** {question}

กรุณาตอบเป็นภาษาไทยที่อ่านเข้าใจง่าย และให้ความเห็นที่เป็นประโยชน์
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
        
        # Generate with safety settings for opinion
        generation_config = genai.types.GenerationConfig(
            temperature=0.7 if enable_opinion else 0.3,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
        
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        answer = (getattr(response, "text", "") or "").strip()
        
        if not answer:
            return "❌ ไม่สามารถประมวลผลคำตอบได้ กรุณาลองใหม่"
            
        logger.info(f"Generated response for question: {question[:50]}...")
        return answer
        
    except Exception as e:
        logger.error(f"Error in generate_enhanced_response: {e}")
        return "❌ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่หรือติดต่อผู้ดูแลระบบ"

# ===================== Profile-based Quick Answers =====================
def get_quick_answer(question: str) -> Optional[str]:
    """Quick answers for basic profile questions"""
    try:
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
        
    except Exception as e:
        logger.error(f"Error in get_quick_answer: {e}")
        return None

# ===================== Utility Functions =====================
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts using Google AI with better error handling"""
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
        
        logger.info(f"Successfully embedded {len(texts)} texts")
        return vecs
        
    except Exception as e:
        logger.error(f"Error in embed_texts: {e}")
        return np.zeros((len(texts), 1), dtype="float32")

async def fetch_resume_text() -> str:
    """Fetch resume content from URL with better error handling"""
    if not RESUME_URL:
        logger.warning("No RESUME_URL provided")
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
                text_content = "\n".join(pages)
                logger.info(f"Successfully fetched PDF content: {len(text_content)} characters")
                return text_content
            else:
                soup = BeautifulSoup(r.text, "html.parser")
                text_content = soup.get_text()
                logger.info(f"Successfully fetched HTML content: {len(text_content)} characters")
                return text_content
                
    except Exception as e:
        logger.error(f"Error fetching resume: {e}")
        return ""

def build_search_index():
    """Build TF-IDF and embedding indices with error handling"""
    global VECTORIZER, TFIDF_MATRIX, EMB_MATRIX
    
    if not RESUME_CHUNKS:
        logger.warning("No resume chunks to index")
        return
    
    try:
        contents = [chunk.content for chunk in RESUME_CHUNKS]
        
        # TF-IDF
        VECTORIZER = TfidfVectorizer(min_df=1, ngram_range=(1,2), max_features=1000)
        TFIDF_MATRIX = VECTORIZER.fit_transform(contents)
        
        # Embeddings
        EMB_MATRIX = embed_texts(contents)
        if EMB_MATRIX is not None and EMB_MATRIX.size > 0:
            norms = np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True) + 1e-12
            EMB_MATRIX = EMB_MATRIX / norms
        
        logger.info(f"Built search index with {len(contents)} chunks")
        
    except Exception as e:
        logger.error(f"Error building search index: {e}")

async def ensure_data_loaded(force: bool = False):
    """Ensure resume data is loaded and indexed with better error handling"""
    global PROFILE, RESUME_CHUNKS, LAST_FETCH_AT
    
    if RESUME_CHUNKS and not force and (time.time() - LAST_FETCH_AT < 3600):
        return
    
    try:
        text = await fetch_resume_text()
        if text:
            PROFILE, RESUME_CHUNKS = parse_structured_resume(text)
            build_search_index()
            LAST_FETCH_AT = time.time()
            logger.info(f"Data loaded successfully: {len(RESUME_CHUNKS)} chunks")
        else:
            logger.warning("No resume text fetched")
    except Exception as e:
        logger.error(f"Error ensuring data loaded: {e}")

# ===================== API Routes =====================
@app.get("/")
def home():
    return {
        "service": "enhanced-resume-chatbot", 
        "status": "ready",
        "version": "2.0.0",
        "opinion_enabled": True
    }

@app.get("/health")
async def health():
    try:
        await ensure_data_loaded()
        return {
            "ok": True, 
            "chunks": len(RESUME_CHUNKS),
            "profile_loaded": bool(PROFILE.name_en or PROFILE.name_th),
            "vectorizer_ready": VECTORIZER is not None,
            "embeddings_ready": EMB_MATRIX is not None
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug")
async def debug():
    try:
        await ensure_data_loaded()
        return {
            "chunks_count": len(RESUME_CHUNKS),
            "sections": [chunk.section for chunk in RESUME_CHUNKS],
            "profile": {
                "name_en": PROFILE.name_en,
                "name_th": PROFILE.name_th,
                "email": PROFILE.email,
                "skills_count": len(PROFILE.skills)
            },
            "last_fetch": LAST_FETCH_AT,
            "opinion_enabled": True
        }
    except Exception as e:
        logger.error(f"Debug endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/refresh")
async def refresh():
    try:
        await ensure_data_loaded(force=True)
        return {
            "ok": True, 
            "chunks": len(RESUME_CHUNKS),
            "message": "Data refreshed successfully"
        }
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        await ensure_data_loaded()
        
        question = (req.message or "").strip()
        if not question:
            return ChatResponse(
                reply="👋 สวัสดีครับ! สามารถถามคำถามเกี่ยวกับเรซูเม่ได้เลย\n\nตัวอย่าง:\n• ชื่ออะไร?\n• มีทักษะอะไรบ้าง?\n• เหมาะกับตำแหน่ง Data Analyst ไหม?\n• สรุปเรซูเม่ให้หน่อย",
                sources=[],
                opinion_enabled=True
            )
        
        # Try quick answer first
        quick = get_quick_answer(question)
        if quick:
            return ChatResponse(
                reply=quick, 
                sources=[],
                opinion_enabled=True
            )
        
        # Retrieve relevant contexts
        hits = multi_query_retrieval(question, k=5)
        contexts = [hit[1] for hit in hits[:3]]
        sources = [(hit[0], hit[1][:150] + "..." if len(hit[1]) > 150 else hit[1]) for hit in hits[:3]]
        
        # Generate response with opinion enabled
        reply = generate_enhanced_response(question, contexts, enable_opinion=True)
        
        return ChatResponse(
            reply=reply, 
            sources=sources,
            opinion_enabled=True
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        return ChatResponse(
            reply=f"❌ เกิดข้อผิดพลาดในระบบ: {str(e)}\n\nกรุณาลองใหม่อีกครั้ง หรือติดต่อผู้ดูแลระบบ",
            sources=[],
            opinion_enabled=True
        )

# ===================== Additional API Endpoints =====================
@app.get("/profile")
async def get_profile():
    """Get structured profile data"""
    try:
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
    except Exception as e:
        logger.error(f"Profile endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{query}")
async def search_resume(query: str):
    """Search in resume content"""
    try:
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
    except Exception as e:
        logger.error(f"Search endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Add a test endpoint to verify opinion functionality
@app.post("/test-opinion")
async def test_opinion():
    """Test endpoint to verify opinion functionality"""
    try:
        await ensure_data_loaded()
        
        test_question = "คุณ Nachapol เหมาะกับตำแหน่ง Data Analyst ไหม"
        hits = multi_query_retrieval(test_question, k=3)
        contexts = [hit[1] for hit in hits[:3]]
        
        response_with_opinion = generate_enhanced_response(test_question, contexts, enable_opinion=True)
        response_without_opinion = generate_enhanced_response(test_question, contexts, enable_opinion=False)
        
        return {
            "test_question": test_question,
            "with_opinion": response_with_opinion,
            "without_opinion": response_without_opinion,
            "opinion_enabled": True,
            "contexts_found": len(contexts)
        }
    except Exception as e:
        logger.error(f"Test opinion endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
