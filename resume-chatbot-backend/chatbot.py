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
from scipy.sparse import hstack

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
    response_type: str = "mixed"  # factual, opinion, mixed
    confidence: float = 0.0  # Confidence in factual information

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

class QuestionClassifier:
    """Classify questions to determine appropriate response strategy"""
    
    @staticmethod
    def classify_question(question: str) -> Dict[str, Any]:
        q_lower = question.lower()
        
        # Direct factual questions
        factual_patterns = [
            r'ชื่ออะไร|what.*name',
            r'อีเมล|email',
            r'เบอร์|โทร|phone',
            r'ที่อยู่|location|address',
            r'ประสบการณ์กี่ปี|years.*experience',
            r'เรียนจบ|graduated|education',
            r'ทำงานที่|work.*at|company'
        ]
        
        # Opinion/analysis questions
        opinion_patterns = [
            r'เหมาะ|suitable|fit|match',
            r'จุดแข็ง|จุดอ่อน|strength|weakness',
            r'แนะนำ|recommend|suggest',
            r'คิดว่า|think|opinion',
            r'ประเมิน|evaluate|assess',
            r'เปรียบเทียบ|compare',
            r'ควร|should|would',
            r'โอกาส|opportunity|potential'
        ]
        
        # Skill/capability questions (semi-factual)
        skill_patterns = [
            r'ทักษะ|skill|ability|competent',
            r'สามารถ|can|able',
            r'มีประสบการณ์.*ใน|experience.*in',
            r'เคยทำ|have.*done|worked.*on'
        ]
        
        # Interview simulation questions
        interview_patterns = [
            r'ทำไม.*สนใจ|why.*interested',
            r'motivat|แรงจูงใจ',
            r'goal|เป้าหมาย',
            r'expect.*salary|เงินเดือน.*คาด',
            r'weakness|จุดอ่อน.*คุณ',
            r'challenge|ความท้าทาย'
        ]
        
        question_type = "factual"  # default
        needs_opinion = False
        interview_mode = False
        confidence_threshold = 0.7
        
        for pattern in factual_patterns:
            if re.search(pattern, q_lower):
                question_type = "factual"
                confidence_threshold = 0.9
                break
                
        for pattern in opinion_patterns:
            if re.search(pattern, q_lower):
                question_type = "opinion"
                needs_opinion = True
                confidence_threshold = 0.5
                break
                
        for pattern in skill_patterns:
            if re.search(pattern, q_lower):
                question_type = "capability"
                needs_opinion = True
                confidence_threshold = 0.6
                break
                
        for pattern in interview_patterns:
            if re.search(pattern, q_lower):
                question_type = "interview"
                needs_opinion = True
                interview_mode = True
                confidence_threshold = 0.3
                break
        
        return {
            "type": question_type,
            "needs_opinion": needs_opinion,
            "interview_mode": interview_mode,
            "confidence_threshold": confidence_threshold
        }

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

# ===== [เพิ่ม] enrich metadata =====
for chunk in chunks:
    # basic section name ให้ชัด
    chunk.metadata.setdefault('section', chunk.section)

    # เดา employer/role/dates ใน section=experience (pattern อาจปรับตามเรซูเม่จริง)
    if chunk.section == 'experience':
        m = re.search(r'(?P<company>[^\n]+?)\s+[-–]\s+(?P<role>[^\(]+?)\s*\((?P<dates>[^)]+)\)', chunk.content)
        if m:
            chunk.metadata.update({
                'employer': m.group('company').strip(),
                'role': m.group('role').strip(),
                'dates': m.group('dates').strip()
            })

    # ดึง skills จาก bullets/รายการ
    if chunk.section == 'skills':
        skill_lines = re.findall(r'[•·\-]\s*([^\n]+)', chunk.content)
        skills = []
        for line in skill_lines:
            skills.extend([s.strip() for s in re.split(r'[,/|]', line) if s.strip()])
        if skills:
            # เก็บแบบ unique
            chunk.metadata['skills'] = sorted(list({s for s in skills}))


# ===================== Enhanced Query Processing =====================
def expand_query_for_context(query: str, question_class: Dict) -> List[str]:
    """Expand query based on question classification"""
    queries = [query]
    query_lower = query.lower()
    
    if question_class["type"] == "opinion":
        # For opinion questions, get broader context
        if 'data' in query_lower:
            queries.extend([
                'data analysis experience',
                'SQL Python analytics',
                'business intelligence reporting',
                'statistical analysis'
            ])
        if 'project' in query_lower:
            queries.extend([
                'project management',
                'team leadership',
                'stakeholder management'
            ])
        if 'management' in query_lower:
            queries.extend([
                'leadership experience',
                'team management',
                'people management'
            ])
    
    elif question_class["type"] == "capability":
        # For capability questions, focus on skills and experience
        queries.extend([
            'technical skills experience',
            'professional experience',
            'achievements results'
        ])
    
    elif question_class["type"] == "interview":
        # For interview questions, get personal insights
        queries.extend([
            'goals objectives motivation',
            'strengths achievements',
            'challenges learning'
        ])
    
    return queries

def multi_query_retrieval(query: str, question_class: Dict, k: int = 5) -> List[Tuple[int, str, float]]:
    """Enhanced retrieval with question-aware strategy"""
    if not RESUME_CHUNKS or not VECTORIZER:
        logger.warning("No resume chunks or vectorizer available")
        return []
    
    all_results = {}  # chunk_idx -> max_score
    
    try:
        queries = expand_query_for_context(query, question_class)
        
        for q in queries:
            # TF-IDF retrieval
            if isinstance(VECTORIZER, tuple):
                vec_word, vec_char = VECTORIZER
                q_word = vec_word.transform([q]) * 0.5
                q_char = vec_char.transform([q]) * 0.5
                qv = hstack([q_word, q_char])
            else:
                qv = VECTORIZER.transform([q])  # เผื่อกรณีเก่า
            tfidf_scores = cosine_similarity(qv, TFIDF_MATRIX).ravel()
            
            # Embedding retrieval
            emb_scores = np.zeros_like(tfidf_scores)
            if EMB_MATRIX is not None and EMB_MATRIX.size > 0:
                emb_q = embed_texts([q])
                if emb_q is not None and emb_q.size > 0:
                    qn = emb_q / (np.linalg.norm(emb_q, axis=1, keepdims=True) + 1e-12)
                    emb_scores = (EMB_MATRIX @ qn.T).ravel()
            
            # Question-type aware scoring
            for i, (tfidf_score, emb_score) in enumerate(zip(tfidf_scores, emb_scores)):
                hybrid_score = 0.4 * tfidf_score + 0.6 * emb_score
                
                sec = RESUME_CHUNKS[i].section
                md = RESUME_CHUNKS[i].metadata or {}
                
                # บูสต์ตามประเภทคำถาม
                if question_class["type"] == "factual" and sec in ['basic_info', 'education', 'experience']:
                    hybrid_score *= 1.3
                elif question_class["type"] in ["opinion", "capability"] and sec in ['achievements', 'experience', 'skills']:
                    hybrid_score *= 1.2
                elif question_class["type"] == "interview" and sec in ['goals', 'strengths', 'summary']:
                    hybrid_score *= 1.2
                
                # บูสต์ตาม metadata keyword
                ql = q.lower()
                if "sap" in ql and any("sap" in s.lower() for s in md.get("skills", [])):
                    hybrid_score *= 1.2
                if ("ngg" in ql or "kubota" in ql) and md.get("employer", "").lower() in ql:
                    hybrid_score *= 1.2
                
                all_results[i] = max(all_results.get(i, 0), hybrid_score)
        
        # Sort and return top k
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        
        # ตัดทิ้งตัวที่คะแนนต่ำกว่าค่ากลาง/ขั้นต่ำ 0.2
        scores_arr = np.array([s for _, s in sorted_results]) if sorted_results else np.array([0.0])
        cut = float(max(np.percentile(scores_arr, 50), 0.2))
        
        results = []
        for idx, score in sorted_results[:k * 2]:
            if score >= cut:
                results.append((idx, RESUME_CHUNKS[idx].content, float(score)))
            if len(results) >= k:
                break
        
        logger.info(f"Retrieved {len(results)} relevant chunks for {question_class['type']} question")
        return results
        
    except Exception as e:
        logger.error(f"Error in multi_query_retrieval: {e}")
        return []

# ===================== Smart Response Generation =====================
def generate_smart_response(question: str, contexts: List[str], question_class: Dict) -> Tuple[str, str]:
    """Generate intelligent response based on question type and available context"""
    if not contexts:
        return handle_no_context_response(question, question_class)
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        if question_class["type"] == "factual":
            prompt = f"""
คุณเป็น AI Assistant ที่ตอบคำถามจากเรซูเม่อย่างแม่นยำ

**หลักการตอบ:**
- ตอบจากข้อเท็จจริงในเรซูเม่เป็นหลัก
- หากไม่มีข้อมูลตรงตัว ให้บอกชัดเจนว่า "ไม่ระบุในเรซูเม่"
- ตอบกระชับ ตรงประเด็น

**ข้อมูลจากเรซูเม่:**
{chr(10).join(contexts)}

**คำถาม:** {question}

กรุณาตอบเป็นภาษาไทยที่ชัดเจน
"""
            response_type = "factual"
            
        elif question_class["type"] == "opinion" or question_class["type"] == "capability":
            prompt = f"""
คุณเป็น AI Recruiter ที่วิเคราะห์เรซูเม่อย่างเชี่ยวชาญ

**การตอบ:**
1. แยกชัดเจนระหว่างข้อเท็จจริงและความเห็น
2. ให้ความเห็นที่สมเหตุสมผลตามข้อมูลจริง
3. ประเมินจุดแข็งและโอกาสพัฒนา

**รูปแบบ:**
📋 **ข้อเท็จจริงจากเรซูเม่:**
[สรุปข้อมูลที่เกี่ยวข้อง]

💡 **ความเห็นและการประเมิน:**
[วิเคราะห์และคำแนะนำ]

**ข้อมูลจากเรซูเม่:**
{chr(10).join(contexts)}

**คำถาม:** {question}

ตอบเป็นภาษาไทยที่เข้าใจง่าย
"""
            response_type = "mixed"
            
        elif question_class["type"] == "interview":
            prompt = f"""
คุณกำลังจำลองการตอบคำถามสัมภาษณ์งานในนาม Nachapol

**หลักการ:**
- ตอบในบุคคลที่ 1 (ผม/ดิฉัน)
- อ้างอิงข้อมูลจริงจากเรซูเม่
- แสดงบุคลิกที่เหมาะสมกับตำแหน่งงาน
- เพิ่มความน่าเชื่อถือด้วยตัวอย่างจริง

**ข้อมูลจากเรซูเม่:**
{chr(10).join(contexts)}

**คำถามสัมภาษณ์:** {question}

*หมายเหตุ: นี่คือการจำลองคำตอบตามข้อมูลในเรซูเม่*

ตอบในลักษณะผู้สมัครงานที่มั่นใจและมืออาชีพ
"""
            response_type = "interview_simulation"
            
        else:
            # Default mixed response
            prompt = f"""
คุณเป็น AI Assistant ที่ช่วยตอบคำถามเกี่ยวกับเรซูเม่

**ข้อมูลจากเรซูเม่:**
{chr(10).join(contexts)}

**คำถาม:** {question}

ตอบอย่างเป็นธรรมชาติและเป็นประโยชน์
"""
            response_type = "mixed"
        
        # Generate response
        generation_config = genai.types.GenerationConfig(
            temperature=0.3 if question_class["type"] == "factual" else 0.7,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
        
        response = model.generate_content(prompt, generation_config=generation_config)
        answer = (getattr(response, "text", "") or "").strip()
        
        if not answer:
            return "❌ ไม่สามารถประมวลผลคำตอบได้ กรุณาลองใหม่", response_type
            
        return answer, response_type
        
    except Exception as e:
        logger.error(f"Error in generate_smart_response: {e}")
        return "❌ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่", "error"

def handle_no_context_response(question: str, question_class: Dict) -> Tuple[str, str]:
    """Handle cases where no relevant context is found"""
    
    if question_class["type"] == "factual":
        return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเรซูเม่ กรุณาลองถามในรูปแบบอื่น", "no_context"
    
    elif question_class["type"] == "interview":
        # Try to give a general interview-style response
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""
คุณกำลังตอบคำถามสัมภาษณ์ในฐานะผู้สมัครงาน แต่คำถามนี้ไม่มีข้อมูลเฉพาะในเรซูเม่

ให้ตอบเป็นคำตอบทั่วไปที่เหมาะสมสำหรับผู้สมัครงาน พร้อมระบุว่าเป็นคำตอบทั่วไป

**คำถาม:** {question}

*หมายเหตุ: คำตอบนี้เป็นการจำลองทั่วไป ไม่ใช่ข้อมูลเฉพาะจากเรซูเม่*
"""
        try:
            response = model.generate_content(prompt)
            answer = getattr(response, "text", "").strip()
            return answer or "ขออภัย ไม่สามารถตอบคำถามนี้ได้", "general_interview"
        except:
            return "ขออภัย คำถามนี้ต้องการข้อมูลเฉพาะที่ไม่มีในเรซูเม่", "no_context"
    
    else:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในเรซูเม่ กรุณาลองถามในรูปแบบอื่น หรือถามคำถามที่เฉพาะเจาะจงมากขึ้น", "no_context"

# ===================== Profile-based Quick Answers =====================
def get_quick_answer(question: str) -> Optional[Tuple[str, str]]:
    """Quick answers for basic profile questions with response type"""
    try:
        q_lower = question.lower().replace(" ", "")
        
        # Name questions
        if any(k in q_lower for k in ["ชื่ออะไร", "ชื่อคืออะไร", "name", "fullname"]):
            if "thai" in q_lower or "ไทย" in q_lower:
                if PROFILE.name_th:
                    return f"ชื่อภาษาไทย: {PROFILE.name_th}", "factual"
            elif PROFILE.name_en:
                result = f"ชื่อ: {PROFILE.name_en}"
                if PROFILE.name_th:
                    result += f" (ภาษาไทย: {PROFILE.name_th})"
                return result, "factual"
        
        # Contact info
        if "email" in q_lower or "อีเมล" in q_lower:
            if PROFILE.email:
                return f"อีเมล: {PROFILE.email}", "factual"
            
        if "phone" in q_lower or "โทร" in q_lower:
            if PROFILE.phone:
                return f"เบอร์โทร: {PROFILE.phone}", "factual"
            
        if "location" in q_lower or "ที่อยู่" in q_lower:
            if PROFILE.location:
                return f"ที่ตั้ง: {PROFILE.location}", "factual"
        
        return None
        
    except Exception as e:
        logger.error(f"Error in get_quick_answer: {e}")
        return None

# ===================== Utility Functions =====================
def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed ทีละข้อความ (loop) เพื่อหลีกเลี่ยงปัญหา batch schema ต่างเวอร์ชัน"""
    if not texts:
        return np.zeros((0, 1), dtype="float32")
    vecs = []
    try:
        for t in texts:
            resp = genai.embed_content(
                model="text-embedding-004",
                content=t,
                task_type="retrieval_document"
            )
            emb = resp.get("embedding") or resp.get("embeddings")
            if isinstance(emb, dict) and "values" in emb:
                vecs.append(np.array(emb["values"], dtype="float32"))
            elif isinstance(emb, list) and len(emb) > 0 and isinstance(emb[0], dict) and "values" in emb[0]:
                vecs.append(np.array(emb[0]["values"], dtype="float32"))
            else:
                # fallback ป้องกันหลุด schema
                vecs.append(np.zeros((1024,), dtype="float32"))
        return np.vstack(vecs)
    except Exception as e:
        logger.error(f"Error in embed_texts: {e}")
        return np.zeros((len(texts), 1), dtype="float32")

def calculate_response_confidence(contexts: List[str], question: str, question_class: Dict) -> float:
    """Calculate confidence score for the response"""
    if not contexts:
        return 0.0
    
    # Base confidence from context quality
    context_length = sum(len(c.split()) for c in contexts)
    base_confidence = min(context_length / 100, 1.0)  # Normalize by expected context length
    
    # Adjust by question type
    if question_class["type"] == "factual":
        # Factual questions need precise information
        confidence_multiplier = 1.0
    elif question_class["type"] == "opinion":
        # Opinion questions are inherently less certain
        confidence_multiplier = 0.7
    elif question_class["type"] == "interview":
        # Interview simulations are moderate confidence
        confidence_multiplier = 0.8
    else:
        confidence_multiplier = 0.6
    
    return base_confidence * confidence_multiplier

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
    """Build TF-IDF (word + char) และ embeddings พร้อม normalization"""
    global VECTORIZER, TFIDF_MATRIX, EMB_MATRIX

    if not RESUME_CHUNKS:
        logger.warning("No resume chunks to index")
        return

    try:
        contents = [chunk.content for chunk in RESUME_CHUNKS]

        # 1) Word n-gram (เหมาะอังกฤษ/ทับศัพท์)
        vec_word = TfidfVectorizer(min_df=1, ngram_range=(1, 2), max_features=20000)
        X_word = vec_word.fit_transform(contents)

        # 2) Char n-gram (เหมาะภาษาไทย/ติดคำ)
        vec_char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 6), min_df=1, max_features=30000)
        X_char = vec_char.fit_transform(contents)

        # รวมสองมุมมอง (ถ่วงเท่า ๆ กันก่อน)
        VECTORIZER = (vec_word, vec_char)
        TFIDF_MATRIX = hstack([X_word * 0.5, X_char * 0.5])

        # 3) Embeddings + L2 normalize
        EMB_MATRIX = embed_texts(contents)
        if EMB_MATRIX is not None and EMB_MATRIX.size > 0:
            norms = np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True) + 1e-12
            EMB_MATRIX = EMB_MATRIX / norms

        logger.info(f"Built search index with {len(contents)} chunks (word+char TFIDF, embeddings)")

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
        "features": ["intelligent_classification", "interview_simulation", "confidence_scoring"]
    }

@app.get("/health")
async def health():
    try:
        await ensure_data_loaded()
        emb_ready = EMB_MATRIX is not None and hasattr(EMB_MATRIX, "size") and EMB_MATRIX.size > 10
        tfidf_ready = TFIDF_MATRIX is not None and getattr(TFIDF_MATRIX, "shape", [0])[0] > 0
        return {
            "ok": True,
            "chunks": len(RESUME_CHUNKS),
            "profile_loaded": bool(PROFILE.name_en or PROFILE.name_th),
            "vectorizer_ready": tfidf_ready,
            "embeddings_ready": emb_ready,
            "ai_ready": bool(GOOGLE_API_KEY)
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
            "intelligent_features": True
        }
    except Exception as e:
        logger.error(f"Debug endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/debug-metadata")
async def debug_metadata(sample: int = 5):
    """ดู metadata ของชังก์ตัวอย่าง เพื่อยืนยันว่ามี employer/role/dates/skills ไหลเข้า index จริง"""
    try:
        await ensure_data_loaded()
        data = []
        for i, c in enumerate(RESUME_CHUNKS[:sample]):
            data.append({
                "idx": i,
                "section": c.section,
                "metadata": c.metadata,
                "preview": c.content[:120] + ("..." if len(c.content) > 120 else "")
            })
        return {"count": len(RESUME_CHUNKS), "samples": data}
    except Exception as e:
        logger.error(f"debug_metadata failed: {e}")
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
                reply="👋 สวัสดีครับ! ผมคือ AI Assistant ที่จะช่วยตอบคำถามเกี่ยวกับเรซูเม่ของคุณ Nachapol\n\n🔍 **ตัวอย่างคำถาม:**\n• ข้อมูลพื้นฐาน: ชื่ออะไร? อีเมลคืออะไร?\n• ทักษะและความสามารถ: มีทักษะอะไรบ้าง?\n• การประเมิน: เหมาะกับตำแหน่ง Data Analyst ไหม?\n• สัมภาษณ์งาน: ทำไมสนใจตำแหน่งนี้?\n\n💡 ระบบจะแยกแยะระหว่างข้อเท็จจริงในเรซูเม่กับความเห็นส่วนตัวให้ชัดเจน",
                sources=[],
                response_type="greeting",
                confidence=1.0
            )
        
        # Classify the question
        question_class = QuestionClassifier.classify_question(question)
        
        # Try quick answer first for factual questions
        if question_class["type"] == "factual":
            quick = get_quick_answer(question)
            if quick:
                return ChatResponse(
                    reply=quick[0], 
                    sources=[],
                    response_type=quick[1],
                    confidence=0.95
                )
        
        # Retrieve relevant contexts
        hits = multi_query_retrieval(question, question_class, k=6)
        contexts = [hit[1] for hit in hits[:4]]  # Use top 4 for better context
        sources = [(hit[0], hit[1][:200] + "..." if len(hit[1]) > 200 else hit[1]) for hit in hits[:3]]
        
        # Calculate confidence
        confidence = calculate_response_confidence(contexts, question, question_class)
        
        # Generate intelligent response
        reply, response_type = generate_smart_response(question, contexts, question_class)
        
        # Add metadata for transparency
        if response_type in ["mixed", "interview_simulation"]:
            if not any(indicator in reply.lower() for indicator in ["📋", "💡", "หมายเหตุ"]):
                # Add transparency note if not already included
                if response_type == "interview_simulation":
                    reply += "\n\n*หมายเหตุ: คำตอบนี้เป็นการจำลองการสัมภาษณ์ตามข้อมูลในเรซูเม่*"
                elif confidence < 0.6:
                    reply += "\n\n*หมายเหตุ: คำตอบบางส่วนอาจเป็นการวิเคราะห์จากข้อมูลที่มีอยู่*"
        
        return ChatResponse(
            reply=reply, 
            sources=sources,
            response_type=response_type,
            confidence=confidence
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        return ChatResponse(
            reply=f"❌ เกิดข้อผิดพลาดในระบบ: {str(e)}\n\nกรุณาลองใหม่อีกครั้ง หรือติดต่อผู้ดูแลระบบ",
            sources=[],
            response_type="error",
            confidence=0.0
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
        question_class = QuestionClassifier.classify_question(query)
        hits = multi_query_retrieval(query, question_class, k=5)
        return {
            "query": query,
            "question_type": question_class["type"],
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

@app.post("/analyze-question")
async def analyze_question(req: ChatRequest):
    """Analyze question type and strategy"""
    try:
        question_class = QuestionClassifier.classify_question(req.message)
        
        return {
            "question": req.message,
            "classification": question_class,
            "strategy": {
                "response_approach": question_class["type"],
                "needs_opinion": question_class["needs_opinion"],
                "interview_mode": question_class["interview_mode"],
                "confidence_threshold": question_class["confidence_threshold"]
            }
        }
    except Exception as e:
        logger.error(f"Question analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate-interview")
async def simulate_interview():
    """Get common interview questions for testing"""
    try:
        await ensure_data_loaded()
        
        interview_questions = [
            "ทำไมถึงสนใจตำแหน่งนี้?",
            "จุดแข็งและจุดอ่อนของคุณคืออะไร?",
            "เป้าหมายในอนาคตคืออะไร?",
            "ท่านคาดหวังเงินเดือนเท่าไหร่?",
            "มีประสบการณ์ที่ท้าทายที่สุดคืออะไร?",
            "ทำไมถึงอยากเปลี่ยนงาน?",
            "คุณจะจัดการกับแรงกดดันในการทำงานอย่างไร?",
            "มีคำถามอะไรที่อยากถามเกี่ยวกับบริษัทบ้างไหม?"
        ]
        
        # Generate sample responses for a few questions
        sample_responses = {}
        for q in interview_questions[:3]:
            question_class = QuestionClassifier.classify_question(q)
            hits = multi_query_retrieval(q, question_class, k=3)
            contexts = [hit[1] for hit in hits]
            
            if contexts:
                reply, _ = generate_smart_response(q, contexts, question_class)
                sample_responses[q] = reply
        
        return {
            "interview_questions": interview_questions,
            "sample_responses": sample_responses,
            "note": "ใช้ /chat endpoint เพื่อทดสอบคำถามสัมภาษณ์"
        }
        
    except Exception as e:
        logger.error(f"Interview simulation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/capabilities")
async def get_capabilities():
    """Get chatbot capabilities and features"""
    return {
        "question_types": {
            "factual": {
                "description": "คำถามเกี่ยวกับข้อเท็จจริงในเรซูเม่",
                "examples": ["ชื่ออะไร?", "อีเมลคืออะไร?", "เรียนจบจากไหน?"],
                "response_style": "ตรงไปตรงมา อิงจากข้อมูลจริง"
            },
            "opinion": {
                "description": "คำถามที่ต้องการการวิเคราะห์และความเห็น",
                "examples": ["เหมาะกับตำแหน่ง Data Analyst ไหม?", "จุดแข็งคืออะไร?"],
                "response_style": "แยกข้อเท็จจริงและความเห็นชัดเจน"
            },
            "capability": {
                "description": "คำถามเกี่ยวกับทักษะและความสามารถ",
                "examples": ["มีทักษะอะไรบ้าง?", "สามารถทำงานด้าน AI ได้ไหม?"],
                "response_style": "อิงจากข้อมูลทักษะและประสบการณ์"
            },
            "interview": {
                "description": "จำลองการสัมภาษณ์งาน",
                "examples": ["ทำไมสนใจตำแหน่งนี้?", "เป้าหมายคืออะไร?"],
                "response_style": "ตอบในฐานะผู้สมัครงาน"
            }
        },
        "features": [
            "อัตโนมัติจำแนกประเภทคำถาม",
            "แยกข้อเท็จจริงและความเห็น",
            "จำลองการสัมภาษณ์งาน",
            "คำนวณค่าความเชื่อมั่นในคำตอบ",
            "ค้นหาบริบทที่เกี่ยวข้องอัจฉริยะ"
        ],
        "transparency": {
            "factual_responses": "อิงจากข้อมูลในเรซูเม่เท่านั้น",
            "opinion_responses": "ระบุชัดเจนส่วนที่เป็นการวิเคราะห์",
            "interview_simulation": "แจ้งเตือนว่าเป็นการจำลอง",
            "confidence_scoring": "ให้คะแนนความเชื่อมั่นในคำตอบ"
        }
    }

# ===================== Testing and Validation =====================
@app.post("/test-intelligence")
async def test_intelligence():
    """Test the intelligent response system"""
    try:
        await ensure_data_loaded()
        
        test_cases = [
            {
                "question": "ชื่ออะไร?",
                "expected_type": "factual",
                "expected_confidence": "high"
            },
            {
                "question": "เหมาะกับตำแหน่ง Data Scientist ไหม?",
                "expected_type": "opinion",
                "expected_confidence": "medium"
            },
            {
                "question": "ทำไมถึงสนใจตำแหน่งนี้?",
                "expected_type": "interview",
                "expected_confidence": "low-medium"
            },
            {
                "question": "มีทักษะด้าน Machine Learning ไหม?",
                "expected_type": "capability",
                "expected_confidence": "medium"
            }
        ]
        
        results = []
        for test in test_cases:
            question_class = QuestionClassifier.classify_question(test["question"])
            hits = multi_query_retrieval(test["question"], question_class, k=3)
            contexts = [hit[1] for hit in hits]
            confidence = calculate_response_confidence(contexts, test["question"], question_class)
            
            if contexts:
                reply, response_type = generate_smart_response(test["question"], contexts, question_class)
            else:
                reply, response_type = handle_no_context_response(test["question"], question_class)
            
            results.append({
                "question": test["question"],
                "expected_type": test["expected_type"],
                "actual_type": question_class["type"],
                "type_match": question_class["type"] == test["expected_type"],
                "confidence": confidence,
                "contexts_found": len(contexts),
                "response_preview": reply[:100] + "..." if len(reply) > 100 else reply
            })
        
        return {
            "test_results": results,
            "summary": {
                "total_tests": len(test_cases),
                "type_accuracy": sum(1 for r in results if r["type_match"]) / len(results),
                "avg_confidence": sum(r["confidence"] for r in results) / len(results)
            }
        }
        
    except Exception as e:
        logger.error(f"Intelligence test failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
