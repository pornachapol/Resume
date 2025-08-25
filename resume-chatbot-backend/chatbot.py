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
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import hashlib

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
app = FastAPI(title="Enhanced Resume Chatbot with Metadata", version="2.1.0")
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
    metadata: Dict = field(default_factory=dict)  # Added metadata field

# ===================== Enhanced Data Structures with Rich Metadata =====================
@dataclass
class ContentMetadata:
    """Rich metadata for content analysis"""
    # Content characteristics
    word_count: int = 0
    sentence_count: int = 0
    paragraph_count: int = 0
    content_hash: str = ""
    
    # Semantic information
    keywords: List[str] = field(default_factory=list)
    entities: List[Dict] = field(default_factory=list)  # names, dates, companies, etc.
    technical_terms: List[str] = field(default_factory=list)
    
    # Context relevance
    relevance_tags: List[str] = field(default_factory=list)  # skills, experience, education, etc.
    importance_score: float = 0.0  # 0-1 scale
    factual_density: float = 0.0  # ratio of factual vs descriptive content
    
    # Temporal information
    date_references: List[Dict] = field(default_factory=list)  # extracted dates and periods
    recency_score: float = 0.0  # how recent/current the information is
    
    # Question-answering metadata
    question_types: List[str] = field(default_factory=list)  # what types of questions this can answer
    answer_confidence: Dict[str, float] = field(default_factory=dict)  # confidence per question type

class ResumeChunk:
    def __init__(self, content: str, section: str = "general", metadata: Dict = None):
        self.content = content
        self.section = section
        self.base_metadata = metadata or {}
        self.rich_metadata = ContentMetadata()
        self.created_at = datetime.now()
        self.last_accessed = datetime.now()
        self.access_count = 0
        
        # Generate content metadata
        self._generate_content_metadata()
    
    def _generate_content_metadata(self):
        """Generate rich metadata for the content"""
        content = self.content.lower()
        
        # Basic statistics
        self.rich_metadata.word_count = len(self.content.split())
        self.rich_metadata.sentence_count = len([s for s in self.content.split('.') if s.strip()])
        self.rich_metadata.paragraph_count = len([p for p in self.content.split('\n\n') if p.strip()])
        self.rich_metadata.content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        
        # Extract keywords and technical terms
        self._extract_keywords_and_terms()
        
        # Extract entities
        self._extract_entities()
        
        # Calculate importance and factual density
        self._calculate_content_scores()
        
        # Extract temporal information
        self._extract_temporal_info()
        
        # Generate question-answer metadata
        self._generate_qa_metadata()
    
    def _extract_keywords_and_terms(self):
        """Extract keywords and technical terms"""
        content_lower = self.content.lower()
        
        # Technical skills and tools
        technical_patterns = [
            r'\b(?:python|sql|javascript|excel|power\s*bi|tableau|r|sas|spss)\b',
            r'\b(?:machine\s*learning|data\s*analytics|statistics|automation|rpa)\b',
            r'\b(?:lean\s*six\s*sigma|agile|scrum|kanban|process\s*improvement)\b',
            r'\b(?:etl|dashboard|visualization|reporting|analysis)\b',
        ]
        
        for pattern in technical_patterns:
            matches = re.findall(pattern, content_lower)
            self.rich_metadata.technical_terms.extend(matches)
        
        # Industry keywords
        industry_keywords = [
            'insurance', 'manufacturing', 'retail', 'supply chain', 'claims',
            'production', 'operations', 'quality', 'efficiency', 'productivity',
            'automation', 'optimization', 'leadership', 'management', 'analysis'
        ]
        
        for keyword in industry_keywords:
            if keyword in content_lower:
                self.rich_metadata.keywords.append(keyword)
        
        # Remove duplicates
        self.rich_metadata.technical_terms = list(set(self.rich_metadata.technical_terms))
        self.rich_metadata.keywords = list(set(self.rich_metadata.keywords))
    
    def _extract_entities(self):
        """Extract entities like companies, dates, locations"""
        # Company names
        company_patterns = [
            r'(?:Generali|NGG Enterprise|Shinning Gold|Siam Kubota|Thammasat University|NIDA)',
            r'(?:Corporation|Company|Enterprise|University|Institute|School)'
        ]
        
        # Date patterns
        date_patterns = [
            r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}',
            r'\d{4}\s*-\s*\d{4}',
            r'Present|Current'
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, self.content, re.IGNORECASE)
            for match in matches:
                self.rich_metadata.entities.append({
                    'type': 'organization',
                    'value': match,
                    'context': self.section
                })
        
        for pattern in date_patterns:
            matches = re.findall(pattern, self.content)
            for match in matches:
                self.rich_metadata.entities.append({
                    'type': 'date',
                    'value': match,
                    'context': self.section
                })
    
    def _calculate_content_scores(self):
        """Calculate importance and factual density scores"""
        content_lower = self.content.lower()
        
        # Importance based on section and content indicators
        importance_indicators = {
            'achievements': 0.9,
            'experience': 0.8,
            'skills': 0.8,
            'education': 0.7,
            'basic_info': 0.9,
            'summary': 0.6
        }
        
        base_importance = importance_indicators.get(self.section, 0.5)
        
        # Boost for quantitative achievements
        quantitative_boost = 0
        if re.search(r'\d+%|\d+\s*(?:THB|year|kg|time)', content_lower):
            quantitative_boost += 0.2
        
        if re.search(r'(?:led|managed|implemented|designed|developed|improved)', content_lower):
            quantitative_boost += 0.1
        
        self.rich_metadata.importance_score = min(base_importance + quantitative_boost, 1.0)
        
        # Factual density - ratio of factual vs descriptive content
        factual_indicators = len(re.findall(r'\d+|%|year|month|project|system|tool|skill', content_lower))
        total_words = len(self.content.split())
        self.rich_metadata.factual_density = min(factual_indicators / max(total_words, 1), 1.0)
    
    def _extract_temporal_info(self):
        """Extract and analyze temporal information"""
        # Extract date ranges and calculate recency
        current_year = datetime.now().year
        
        # Find years in content
        years = re.findall(r'\b(20\d{2})\b', self.content)
        years = [int(y) for y in years if y]
        
        if years:
            most_recent_year = max(years)
            years_ago = current_year - most_recent_year
            
            # Recency score: 1.0 for current year, decreasing over time
            self.rich_metadata.recency_score = max(0, 1.0 - (years_ago * 0.1))
            
            self.rich_metadata.date_references.append({
                'years_found': years,
                'most_recent': most_recent_year,
                'years_ago': years_ago
            })
        
        # Check for "Present" or "Current"
        if re.search(r'present|current', self.content, re.IGNORECASE):
            self.rich_metadata.recency_score = 1.0
    
    def _generate_qa_metadata(self):
        """Generate metadata for question-answering capabilities"""
        content_lower = self.content.lower()
        
        # Map content to question types it can answer
        qa_mapping = {
            'factual': {
                'patterns': [r'name|email|phone|location|education|degree|university'],
                'confidence': 0.9
            },
            'experience': {
                'patterns': [r'work|job|position|role|company|responsibility|project'],
                'confidence': 0.8
            },
            'skills': {
                'patterns': [r'skill|technology|tool|software|programming|analysis'],
                'confidence': 0.8
            },
            'achievements': {
                'patterns': [r'achievement|award|improvement|success|result|\d+%'],
                'confidence': 0.7
            },
            'temporal': {
                'patterns': [r'\d{4}|year|month|experience|duration|when'],
                'confidence': 0.8
            }
        }
        
        for q_type, config in qa_mapping.items():
            for pattern in config['patterns']:
                if re.search(pattern, content_lower):
                    self.rich_metadata.question_types.append(q_type)
                    self.rich_metadata.answer_confidence[q_type] = config['confidence']
                    break
        
        # Remove duplicates
        self.rich_metadata.question_types = list(set(self.rich_metadata.question_types))
        
        # Add relevance tags based on content
        if self.section == 'basic_info' or any(x in content_lower for x in ['name', 'email', 'phone']):
            self.rich_metadata.relevance_tags.append('contact_info')
        
        if any(x in content_lower for x in ['bachelor', 'master', 'degree', 'university']):
            self.rich_metadata.relevance_tags.append('education')
        
        if any(x in content_lower for x in ['manager', 'engineer', 'supervisor', 'led', 'managed']):
            self.rich_metadata.relevance_tags.append('leadership')
        
        if len(self.rich_metadata.technical_terms) > 0:
            self.rich_metadata.relevance_tags.append('technical_skills')
    
    def update_access_stats(self):
        """Update access statistics"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def get_full_metadata(self) -> Dict:
        """Get complete metadata dictionary"""
        return {
            'section': self.section,
            'base_metadata': self.base_metadata,
            'content_stats': {
                'word_count': self.rich_metadata.word_count,
                'sentence_count': self.rich_metadata.sentence_count,
                'content_hash': self.rich_metadata.content_hash,
                'importance_score': self.rich_metadata.importance_score,
                'factual_density': self.rich_metadata.factual_density,
                'recency_score': self.rich_metadata.recency_score
            },
            'semantic_info': {
                'keywords': self.rich_metadata.keywords,
                'technical_terms': self.rich_metadata.technical_terms,
                'entities': self.rich_metadata.entities,
                'relevance_tags': self.rich_metadata.relevance_tags
            },
            'qa_capabilities': {
                'question_types': self.rich_metadata.question_types,
                'answer_confidence': self.rich_metadata.answer_confidence
            },
            'access_stats': {
                'created_at': self.created_at.isoformat(),
                'last_accessed': self.last_accessed.isoformat(),
                'access_count': self.access_count
            },
            'temporal_info': self.rich_metadata.date_references
        }

@dataclass
class ProfileData:
    """Enhanced profile data with metadata"""
    # Basic information
    name_en: Optional[str] = None
    name_th: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    
    # Structured data
    skills: List[str] = field(default_factory=list)
    experience: List[Dict] = field(default_factory=list)
    education: List[Dict] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    achievements: List[str] = field(default_factory=list)
    
    # Metadata
    profile_completeness: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    data_quality_score: float = 0.0
    skill_categories: Dict[str, List[str]] = field(default_factory=dict)
    experience_summary: Dict = field(default_factory=dict)
    
    def calculate_completeness(self):
        """Calculate profile completeness percentage"""
        fields_to_check = [
            self.name_en or self.name_th,
            self.email,
            self.phone,
            self.location,
            len(self.skills) > 0,
            len(self.experience) > 0,
            len(self.education) > 0
        ]
        
        completed_fields = sum(1 for field in fields_to_check if field)
        self.profile_completeness = completed_fields / len(fields_to_check)
        return self.profile_completeness
    
    def categorize_skills(self):
        """Categorize skills into technical, soft skills, etc."""
        technical_skills = []
        soft_skills = []
        industry_skills = []
        
        technical_keywords = [
            'python', 'sql', 'javascript', 'excel', 'power bi', 'tableau',
            'machine learning', 'data analytics', 'rpa', 'automation',
            'etl', 'dashboard', 'visualization'
        ]
        
        soft_keywords = [
            'leadership', 'management', 'communication', 'teamwork',
            'problem solving', 'analytical thinking', 'project management'
        ]
        
        for skill in self.skills:
            skill_lower = skill.lower()
            if any(tech in skill_lower for tech in technical_keywords):
                technical_skills.append(skill)
            elif any(soft in skill_lower for soft in soft_keywords):
                soft_skills.append(skill)
            else:
                industry_skills.append(skill)
        
        self.skill_categories = {
            'technical': technical_skills,
            'soft_skills': soft_skills,
            'industry': industry_skills
        }

class QuestionClassifier:
    """Enhanced question classifier with metadata support"""
    
    @staticmethod
    def classify_question(question: str) -> Dict[str, Any]:
        q_lower = question.lower()
        
        # Enhanced patterns with metadata
        classification_patterns = {
            'factual': {
                'patterns': [
                    r'ชื่ออะไร|what.*name',
                    r'อีเมล|email',
                    r'เบอร์|โทร|phone',
                    r'ที่อยู่|location|address',
                    r'ประสบการณ์กี่ปี|years.*experience',
                    r'เรียนจบ|graduated|education',
                    r'ทำงานที่|work.*at|company'
                ],
                'confidence_threshold': 0.9,
                'response_strategy': 'direct_facts',
                'metadata_priority': ['basic_info', 'education', 'experience']
            },
            'opinion': {
                'patterns': [
                    r'เหมาะ|suitable|fit|match',
                    r'จุดแข็ง|จุดอ่อน|strength|weakness',
                    r'แนะนำ|recommend|suggest',
                    r'คิดว่า|think|opinion',
                    r'ประเมิน|evaluate|assess',
                    r'เปรียบเทียบ|compare',
                    r'ควร|should|would',
                    r'โอกาส|opportunity|potential'
                ],
                'confidence_threshold': 0.5,
                'response_strategy': 'analytical',
                'metadata_priority': ['achievements', 'experience', 'skills']
            },
            'capability': {
                'patterns': [
                    r'ทักษะ|skill|ability|competent',
                    r'สามารถ|can|able',
                    r'มีประสบการณ์.*ใน|experience.*in',
                    r'เคยทำ|have.*done|worked.*on'
                ],
                'confidence_threshold': 0.6,
                'response_strategy': 'capability_focused',
                'metadata_priority': ['skills', 'technical_skills', 'experience']
            },
            'interview': {
                'patterns': [
                    r'ทำไม.*สนใจ|why.*interested',
                    r'motivat|แรงจูงใจ',
                    r'goal|เป้าหมาย',
                    r'expect.*salary|เงินเดือน.*คาด',
                    r'weakness|จุดอ่อน.*คุณ',
                    r'challenge|ความท้าทาย'
                ],
                'confidence_threshold': 0.3,
                'response_strategy': 'interview_simulation',
                'metadata_priority': ['goals', 'strengths', 'summary', 'achievements']
            }
        }
        
        # Classify question
        question_type = "factual"  # default
        config = classification_patterns['factual']
        
        for q_type, q_config in classification_patterns.items():
            for pattern in q_config['patterns']:
                if re.search(pattern, q_lower):
                    question_type = q_type
                    config = q_config
                    break
            if question_type == q_type:
                break
        
        # Enhanced metadata extraction
        question_metadata = {
            'original_question': question,
            'question_length': len(question.split()),
            'contains_thai': bool(re.search(r'[\u0E00-\u0E7F]', question)),
            'contains_numbers': bool(re.search(r'\d', question)),
            'question_words': re.findall(r'\b(?:what|who|when|where|why|how|อะไร|ใคร|เมื่อไหร่|ที่ไหน|ทำไม|อย่างไร)\b', q_lower),
            'urgency_indicators': re.findall(r'\b(?:urgent|immediate|now|ด่วน|เดี๋ยวนี้)\b', q_lower),
            'complexity_score': len(question.split()) / 20.0  # Simple complexity measure
        }
        
        return {
            "type": question_type,
            "needs_opinion": question_type in ['opinion', 'capability', 'interview'],
            "interview_mode": question_type == 'interview',
            "confidence_threshold": config['confidence_threshold'],
            "response_strategy": config['response_strategy'],
            "metadata_priority": config['metadata_priority'],
            "question_metadata": question_metadata,
            "processing_hints": {
                "prefer_recent_data": question_type in ['capability', 'interview'],
                "include_quantitative": 'number' in q_lower or 'จำนวน' in q_lower,
                "focus_achievements": 'achieve' in q_lower or 'success' in q_lower,
                "need_examples": 'example' in q_lower or 'ตัวอย่าง' in q_lower
            }
        }

# ===================== Global Variables =====================
RESUME_CHUNKS: List[ResumeChunk] = []
PROFILE: ProfileData = ProfileData()
VECTORIZER = None
TFIDF_MATRIX = None
EMB_MATRIX = None
LAST_FETCH_AT = 0

# Analytics and performance tracking
QUERY_ANALYTICS = {
    'total_queries': 0,
    'question_types': {},
    'avg_response_time': 0.0,
    'chunk_usage_stats': {},
    'confidence_distribution': []
}

# ===================== Enhanced Parsing with Metadata =====================
def parse_structured_resume(text: str) -> Tuple[ProfileData, List[ResumeChunk]]:
    """Enhanced parsing with rich metadata generation"""
    profile = ProfileData()
    chunks = []
    
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    current_section = "general"
    section_content = []
    
    # Enhanced section patterns
    section_patterns = {
        'basic_info': r'(basic information|personal|contact|bangkok|thailand|email|phone)',
        'summary': r'(professional summary|summary|profile|process improvement|automation leader)',
        'skills': r'(area of expertise|skills|expertise|competencies|technical)',
        'experience': r'(professional experience|experience|work history|career)',
        'education': r'(education|academic|qualification|bachelor|master)',
        'achievements': r'(key achievements|achievements|accomplishments|awards)',
        'strengths': r'(strengths|weaknesses)',
        'goals': r'(goals|objectives|future|passion)',
        'additional': r'(additional information|languages|certifications|awards)'
    }
    
    try:
        section_metadata = {}  # Track metadata per section
        
        for line_idx, line in enumerate(lines):
            line_lower = line.lower()
            
            # Detect section changes with enhanced logic
            new_section = None
            confidence_scores = {}
            
            for section, pattern in section_patterns.items():
                if re.search(pattern, line_lower):
                    # Calculate confidence based on line position and content
                    position_bonus = 0.1 if line_idx < len(lines) * 0.3 else 0  # Early sections get bonus
                    content_match_score = len(re.findall(pattern, line_lower)) * 0.2
                    confidence_scores[section] = 0.7 + position_bonus + content_match_score
            
            if confidence_scores:
                new_section = max(confidence_scores.items(), key=lambda x: x[1])[0]
            
            if new_section and new_section != current_section:
                # Save previous section with metadata
                if section_content:
                    section_meta = {
                        'line_count': len(section_content),
                        'avg_line_length': sum(len(l) for l in section_content) / len(section_content),
                        'detection_confidence': confidence_scores.get(current_section, 0.5),
                        'content_indicators': _extract_section_indicators(section_content, current_section)
                    }
                    
                    chunk = ResumeChunk(
                        content='\n'.join(section_content),
                        section=current_section,
                        metadata=section_meta
                    )
                    chunks.append(chunk)
                    section_metadata[current_section] = section_meta
                
                current_section = new_section
                section_content = [line]
            else:
                section_content.append(line)
        
        # Save last section
        if section_content:
            section_meta = {
                'line_count': len(section_content),
                'avg_line_length': sum(len(l) for l in section_content) / len(section_content),
                'detection_confidence': 0.8,
                'content_indicators': _extract_section_indicators(section_content, current_section)
            }
            
            chunk = ResumeChunk(
                content='\n'.join(section_content),
                section=current_section,
                metadata=section_meta
            )
            chunks.append(chunk)
        
        # Enhanced profile data extraction with metadata
        profile = _extract_enhanced_profile_data(text, chunks)
        
        logger.info(f"Successfully parsed resume with rich metadata: {len(chunks)} chunks, "
                   f"profile completeness: {profile.calculate_completeness():.2f}")
        
        return profile, chunks
        
    except Exception as e:
        logger.error(f"Error parsing resume: {e}")
        return profile, chunks

def _extract_section_indicators(content_lines: List[str], section: str) -> Dict:
    """Extract indicators that help identify section content quality"""
    full_content = '\n'.join(content_lines).lower()
    
    indicators = {
        'has_dates': bool(re.search(r'\b\d{4}\b|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec', full_content)),
        'has_numbers': bool(re.search(r'\d+', full_content)),
        'has_percentages': bool(re.search(r'\d+%', full_content)),
        'has_technical_terms': len(re.findall(r'\b(?:sql|python|excel|power\s*bi|automation|rpa|lean|six\s*sigma)\b', full_content)),
        'has_action_verbs': len(re.findall(r'\b(?:led|managed|implemented|designed|developed|improved|coordinated|supervised)\b', full_content)),
        'bullet_points': len([l for l in content_lines if l.strip().startswith(('•', '-', '*'))]),
        'avg_sentence_length': len(full_content.split()) / max(len([s for s in full_content.split('.') if s.strip()]), 1)
    }
    
    return indicators

def _extract_enhanced_profile_data(text: str, chunks: List[ResumeChunk]) -> ProfileData:
    """Extract enhanced profile data with metadata"""
    profile = ProfileData()
    full_text = text.lower()
    
    # Basic information extraction (unchanged logic but with metadata tracking)
    name_match = re.search(r'([A-Z][A-Z\s]+[A-Z])\s*(?:\n|$)', text)
    if name_match:
        profile.name_en = name_match.group(1).strip()
    
    # Extract structured experience data
    experience_chunk = next((c for c in chunks if c.section == 'experience'), None)
    if experience_chunk:
        profile.experience = _extract_structured_experience(experience_chunk.content)
    
    # Extract structured education data
    education_chunk = next((c for c in chunks if c.section == 'education'), None)
    if education_chunk:
        profile.education = _extract_structured_education(education_chunk.content)
    
    # Enhanced skills extraction with categorization
    skills_chunk = next((c for c in chunks if c.section in ['skills', 'area_of_expertise']), None)
    if skills_chunk:
        profile.skills = _extract_enhanced_skills(skills_chunk.content)
        profile.categorize_skills()
    
    # Calculate profile metrics
    profile.calculate_completeness()
    profile.data_quality_score = _calculate_data_quality_score(profile, chunks)
    profile.last_updated = datetime.now()
    
    return profile

def _extract_structured_experience(content: str) -> List[Dict]:
    """Extract structured experience data with metadata"""
    experiences = []
    
    # Split by date ranges or job titles
    job_sections = re.split(r'\n(?=\w+\s+\d{4})', content)
    
    for section in job_sections:
        if not section.strip():
            continue
        
        exp_data = {
            'raw_content': section,
            'title': '',
            'company': '',
            'duration': '',
            'responsibilities': [],
            'achievements': [],
            'technologies': [],
            'metadata': {}
        }
        
        lines = [l.strip() for l in section.split('\n') if l.strip()]
        
        # Extract job title and dates from first line
        if lines:
            first_line = lines[0]
            date_match = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}.*?(?:Present|\d{4}))', first_line)
            if date_match:
                exp_data['duration'] = date_match.group(1)
                exp_data['title'] = first_line.replace(date_match.group(1), '').strip()
            else:
                exp_data['title'] = first_line
        
        # Extract company from second line if available
        if len(lines) > 1:
            exp_data['company'] = lines[1]
        
        # Process remaining lines for responsibilities and achievements
        for line in lines[2:]:
            if re.search(r'^\s*[•-]\s*', line) or line.startswith('    '):
                # This is a responsibility or achievement
                clean_line = re.sub(r'^\s*[•-]\s*', '', line).strip()
                
                # Classify as achievement if it contains quantitative data
                if re.search(r'\d+%|\d+\s*(?:THB|year|kg|time)|improvement|increase|reduce', clean_line.lower()):
                    exp_data['achievements'].append(clean_line)
                else:
                    exp_data['responsibilities'].append(clean_line)
                
                # Extract technologies mentioned
                tech_terms = re.findall(r'\b(?:Excel|Power BI|SQL|Python|JavaScript|AGV|RPA|ETL|Dashboard)\b', clean_line, re.IGNORECASE)
                exp_data['technologies'].extend(tech_terms)
        
        # Calculate experience metadata
        exp_data['metadata'] = {
            'has_quantitative_results': len(exp_data['achievements']) > 0,
            'technology_count': len(set(exp_data['technologies'])),
            'responsibility_count': len(exp_data['responsibilities']),
            'leadership_indicators': len(re.findall(r'\b(?:led|managed|supervised|coordinated)\b', section.lower())),
            'impact_score': len(re.findall(r'\d+%|\d+\s*(?:THB|improvement|increase|reduce)', section.lower())) / max(len(lines), 1)
        }
        
        experiences.append(exp_data)
    
    return experiences

def _extract_structured_education(content: str) -> List[Dict]:
    """Extract structured education data"""
    education = []
    
    # Split by degree programs
    edu_sections = re.split(r'\n(?=(?:Bachelor|Master|PhD))', content)
    
    for section in edu_sections:
        if not section.strip():
            continue
        
        edu_data = {
            'raw_content': section,
            'degree': '',
            'field': '',
            'institution': '',
            'duration': '',
            'gpa': None,
            'status': 'completed',
            'metadata': {}
        }
        
        lines = [l.strip() for l in section.split('\n') if l.strip()]
        
        for line in lines:
            # Extract degree and field
            degree_match = re.search(r'(Bachelor|Master|PhD).*?(?:in|of)\s+([^|]+)', line)
            if degree_match:
                edu_data['degree'] = degree_match.group(1)
                edu_data['field'] = degree_match.group(2).strip()
            
            # Extract institution
            if '|' in line and not edu_data['institution']:
                parts = line.split('|')
                for part in parts:
                    if any(word in part.lower() for word in ['university', 'institute', 'school', 'college']):
                        edu_data['institution'] = part.strip()
                        break
            
            # Extract GPA
            gpa_match = re.search(r'GPA:\s*(\d+\.?\d*)', line)
            if gpa_match:
                edu_data['gpa'] = float(gpa_match.group(1))
            
            # Extract dates
            date_match = re.search(r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}.*?(?:Present|\d{4}))', line)
            if date_match:
                edu_data['duration'] = date_match.group(1)
                if 'Expected' in line or 'Present' in line:
                    edu_data['status'] = 'in_progress'
        
        # Calculate education metadata
        current_year = datetime.now().year
        years_mentioned = re.findall(r'\b(20\d{2})\b', section)
        most_recent_year = max([int(y) for y in years_mentioned]) if years_mentioned else None
        
        edu_data['metadata'] = {
            'is_current': edu_data['status'] == 'in_progress',
            'recency_years': current_year - most_recent_year if most_recent_year else None,
            'has_gpa': edu_data['gpa'] is not None,
            'field_keywords': re.findall(r'\b(?:engineering|analytics|data|management|science|technology)\b', section.lower())
        }
        
        education.append(edu_data)
    
    return education

def _extract_enhanced_skills(content: str) -> List[str]:
    """Extract skills with enhanced categorization"""
    skills = []
    
    # Extract from bullet points and structured lists
    skill_lines = re.findall(r'[•·-]\s*([^\n]+)', content)
    for line in skill_lines:
        # Split by common separators
        line_skills = re.split(r'[,/|&]', line)
        for skill in line_skills:
            clean_skill = skill.strip()
            if clean_skill and len(clean_skill) > 2:
                skills.append(clean_skill)
    
    # Extract from regular text patterns
    content_lines = content.split('\n')
    for line in content_lines:
        if ':' in line and not line.strip().startswith(('•', '-')):
            # Category: skills format
            parts = line.split(':', 1)
            if len(parts) == 2:
                skill_text = parts[1].strip()
                line_skills = re.split(r'[,/|&]', skill_text)
                for skill in line_skills:
                    clean_skill = skill.strip()
                    if clean_skill and len(clean_skill) > 2:
                        skills.append(clean_skill)
    
    return list(set(skills))  # Remove duplicates

def _calculate_data_quality_score(profile: ProfileData, chunks: List[ResumeChunk]) -> float:
    """Calculate overall data quality score"""
    scores = []
    
    # Profile completeness (30%)
    completeness_score = profile.calculate_completeness()
    scores.append(('completeness', completeness_score, 0.3))
    
    # Content richness (25%)
    total_words = sum(chunk.rich_metadata.word_count for chunk in chunks)
    content_richness = min(total_words / 1000, 1.0)  # Normalize to 1000 words
    scores.append(('content_richness', content_richness, 0.25))
    
    # Factual density (20%)
    if chunks:
        avg_factual_density = sum(chunk.rich_metadata.factual_density for chunk in chunks) / len(chunks)
    else:
        avg_factual_density = 0
    scores.append(('factual_density', avg_factual_density, 0.2))
    
    # Technical content (15%)
    total_technical_terms = sum(len(chunk.rich_metadata.technical_terms) for chunk in chunks)
    tech_score = min(total_technical_terms / 20, 1.0)  # Normalize to 20 terms
    scores.append(('technical_content', tech_score, 0.15))
    
    # Recency (10%)
    if chunks:
        avg_recency = sum(chunk.rich_metadata.recency_score for chunk in chunks) / len(chunks)
    else:
        avg_recency = 0
    scores.append(('recency', avg_recency, 0.1))
    
    # Calculate weighted average
    weighted_score = sum(score * weight for _, score, weight in scores)
    return weighted_score

# ===================== Enhanced Query Processing =====================
def expand_query_for_context(query: str, question_class: Dict) -> List[str]:
    """Enhanced query expansion with metadata awareness"""
    queries = [query]
    query_lower = query.lower()
    processing_hints = question_class.get('processing_hints', {})
    
    # Base expansion based on question type
    if question_class["type"] == "opinion":
        if 'data' in query_lower:
            queries.extend([
                'data analysis experience dashboard visualization',
                'SQL Python analytics Power BI',
                'business intelligence reporting ETL',
                'statistical analysis process improvement'
            ])
        if 'management' in query_lower:
            queries.extend([
                'leadership team management supervision',
                'project management coordination',
                'stakeholder management cross-functional'
            ])
    
    # Add queries based on processing hints
    if processing_hints.get('focus_achievements'):
        queries.extend([
            'achievements results improvement percentage',
            'saved reduced increased optimized'
        ])
    
    if processing_hints.get('include_quantitative'):
        queries.extend([
            'numbers statistics metrics KPI',
            'percentage improvement efficiency'
        ])
    
    if processing_hints.get('prefer_recent_data'):
        queries.extend([
            'current present recent latest',
            '2024 2025 recent experience'
        ])
    
    return queries

def multi_query_retrieval_with_metadata(query: str, question_class: Dict, k: int = 5) -> List[Tuple[int, str, float, Dict]]:
    """Enhanced retrieval with metadata-aware scoring"""
    if not RESUME_CHUNKS or not VECTORIZER:
        logger.warning("No resume chunks or vectorizer available")
        return []
    
    all_results = {}  # chunk_idx -> (max_score, metadata)
    processing_hints = question_class.get('processing_hints', {})
    metadata_priority = question_class.get('metadata_priority', [])
    
    try:
        queries = expand_query_for_context(query, question_class)
        
        for q in queries:
            # Traditional TF-IDF and embedding retrieval
            qv = VECTORIZER.transform([q])
            tfidf_scores = cosine_similarity(qv, TFIDF_MATRIX).ravel()
            
            emb_scores = np.zeros_like(tfidf_scores)
            if EMB_MATRIX is not None and EMB_MATRIX.size > 0:
                emb_q = embed_texts([q])
                if emb_q is not None and emb_q.size > 0:
                    qn = emb_q / (np.linalg.norm(emb_q, axis=1, keepdims=True) + 1e-12)
                    emb_scores = (EMB_MATRIX @ qn.T).ravel()
            
            # Enhanced metadata-aware scoring
            for i, (tfidf_score, emb_score) in enumerate(zip(tfidf_scores, emb_scores)):
                chunk = RESUME_CHUNKS[i]
                chunk.update_access_stats()  # Track usage
                
                # Base hybrid score
                hybrid_score = 0.4 * tfidf_score + 0.6 * emb_score
                
                # Metadata-based boosting
                metadata_boost = 0
                
                # Section priority boost
                if chunk.section in metadata_priority:
                    priority_index = metadata_priority.index(chunk.section)
                    metadata_boost += (len(metadata_priority) - priority_index) * 0.1
                
                # Content quality boost
                metadata_boost += chunk.rich_metadata.importance_score * 0.2
                metadata_boost += chunk.rich_metadata.factual_density * 0.15
                
                # Recency boost if preferred
                if processing_hints.get('prefer_recent_data'):
                    metadata_boost += chunk.rich_metadata.recency_score * 0.2
                
                # Achievement boost if focused on achievements
                if processing_hints.get('focus_achievements'):
                    achievement_indicators = len([tag for tag in chunk.rich_metadata.relevance_tags if 'achievement' in tag])
                    metadata_boost += achievement_indicators * 0.1
                
                # Technical content boost for capability questions
                if question_class["type"] == "capability":
                    tech_boost = len(chunk.rich_metadata.technical_terms) * 0.05
                    metadata_boost += min(tech_boost, 0.3)  # Cap at 0.3
                
                # Question-type specific confidence boost
                qa_confidence = chunk.rich_metadata.answer_confidence.get(question_class["type"], 0)
                metadata_boost += qa_confidence * 0.1
                
                final_score = hybrid_score + metadata_boost
                
                # Store results with metadata
                if i not in all_results or final_score > all_results[i][0]:
                    all_results[i] = (
                        final_score, 
                        {
                            'base_hybrid_score': hybrid_score,
                            'metadata_boost': metadata_boost,
                            'section': chunk.section,
                            'importance_score': chunk.rich_metadata.importance_score,
                            'factual_density': chunk.rich_metadata.factual_density,
                            'recency_score': chunk.rich_metadata.recency_score,
                            'technical_terms_count': len(chunk.rich_metadata.technical_terms),
                            'qa_confidence': qa_confidence,
                            'access_count': chunk.access_count
                        }
                    )
        
        # Sort and return top k with metadata
        sorted_results = sorted(all_results.items(), key=lambda x: x[1][0], reverse=True)
        
        results = []
        for idx, (score, metadata) in sorted_results[:k]:
            if score > question_class["confidence_threshold"] * 0.1:
                results.append((idx, RESUME_CHUNKS[idx].content, score, metadata))
        
        # Update analytics
        QUERY_ANALYTICS['total_queries'] += 1
        q_type = question_class["type"]
        QUERY_ANALYTICS['question_types'][q_type] = QUERY_ANALYTICS['question_types'].get(q_type, 0) + 1
        
        logger.info(f"Retrieved {len(results)} relevant chunks with metadata for {q_type} question")
        return results
        
    except Exception as e:
        logger.error(f"Error in metadata-aware retrieval: {e}")
        return []

# ===================== Enhanced Response Generation =====================
def generate_smart_response_with_metadata(question: str, contexts: List[str], context_metadata: List[Dict], question_class: Dict) -> Tuple[str, str, Dict]:
    """Generate response with metadata insights"""
    if not contexts:
        return handle_no_context_response(question, question_class) + ({},)
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        
        # Analyze context metadata for insights
        metadata_insights = analyze_context_metadata(context_metadata, question_class)
        
        # Enhanced prompts with metadata awareness
        if question_class["type"] == "factual":
            confidence_indicators = ", ".join([
                f"Section: {meta['section']}"
                f" (Factual density: {meta['factual_density']:.1f})"
                for meta in context_metadata[:2]
            ])
            
            prompt = f"""
คุณเป็น AI Assistant ที่ตอบคำถามจากเรซูเม่อย่างแม่นยำ

**หลักการตอบ:**
- ตอบจากข้อเท็จจริงในเรซูเม่เป็นหลัก
- หากไม่มีข้อมูลตรงตัว ให้บอกชัดเจนว่า "ไม่ระบุในเรซูเม่"
- ตอบกระชับ ตรงประเด็น

**คุณภาพข้อมูล:** {confidence_indicators}

**ข้อมูลจากเรซูเม่:**
{chr(10).join(contexts)}

**คำถาม:** {question}

กรุณาตอบเป็นภาษาไทยที่ชัดเจน
"""
            response_type = "factual"
            
        elif question_class["type"] in ["opinion", "capability"]:
            technical_insight = f"Technical terms found: {metadata_insights['total_technical_terms']}" if metadata_insights['total_technical_terms'] > 0 else "Limited technical detail available"
            achievement_insight = f"Achievement indicators: {metadata_insights['achievement_indicators']}" if metadata_insights['achievement_indicators'] > 0 else "Few quantitative achievements found"
            
            prompt = f"""
คุณเป็น AI Recruiter ที่วิเคราะห์เรซูเม่อย่างเชี่ยวชาญ

**การตอบ:**
1. แยกชัดเจนระหว่างข้อเท็จจริงและความเห็น
2. ให้ความเห็นที่สมเหตุสมผลตามข้อมูลจริง
3. ประเมินจุดแข็งและโอกาสพัฒนา

**ข้อมูลเชิงลึก:**
- {technical_insight}
- {achievement_insight}
- ความทันสมัยของข้อมูล: {metadata_insights['avg_recency']:.1f}/1.0

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
            personality_hints = "มีประสบการณ์ด้านการปรับปรุงกระบวนการและการจัดการข้อมูล" if metadata_insights['total_technical_terms'] > 5 else "เน้นประสบการณ์การทำงานและการเรียนรู้"
            
            prompt = f"""
คุณกำลังจำลองการตอบคำถามสัมภาษณ์งานในนาม Nachapol

**หลักการ:**
- ตอบในบุคคลที่ 1 (ผม/ดิฉัน)
- อ้างอิงข้อมูลจริงจากเรซูเม่
- แสดงบุคลิกที่เหมาะสมกับตำแหน่งงาน ({personality_hints})
- เพิ่มความน่าเชื่อถือด้วยตัวอย่างจริง

**ข้อมูลจากเรซูเม่:**
{chr(10).join(contexts)}

**คำถามสัมภาษณ์:** {question}

*หมายเหตุ: นี่คือการจำลองคำตอบตามข้อมูลในเรซูเม่*

ตอบในลักษณะผู้สมัครงานที่มั่นใจและมืออาชีพ
"""
            response_type = "interview_simulation"
        
        else:
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
            temperature=0.2 if question_class["type"] == "factual" else 0.6,
            top_p=0.8,
            top_k=40,
            max_output_tokens=2048,
        )
        
        response = model.generate_content(prompt, generation_config=generation_config)
        answer = (getattr(response, "text", "") or "").strip()
        
        if not answer:
            return "❌ ไม่สามารถประมวลผลคำตอบได้ กรุณาลองใหม่", response_type, metadata_insights
        
        return answer, response_type, metadata_insights
        
    except Exception as e:
        logger.error(f"Error in generate_smart_response_with_metadata: {e}")
        return "❌ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่", "error", {}

def analyze_context_metadata(context_metadata: List[Dict], question_class: Dict) -> Dict:
    """Analyze context metadata to provide insights"""
    if not context_metadata:
        return {}
    
    insights = {
        'total_technical_terms': sum(meta.get('technical_terms_count', 0) for meta in context_metadata),
        'avg_importance': sum(meta.get('importance_score', 0) for meta in context_metadata) / len(context_metadata),
        'avg_factual_density': sum(meta.get('factual_density', 0) for meta in context_metadata) / len(context_metadata),
        'avg_recency': sum(meta.get('recency_score', 0) for meta in context_metadata) / len(context_metadata),
        'sections_covered': list(set(meta.get('section', 'unknown') for meta in context_metadata)),
        'achievement_indicators': sum(1 for meta in context_metadata if meta.get('importance_score', 0) > 0.7),
        'confidence_assessment': sum(meta.get('qa_confidence', 0) for meta in context_metadata) / len(context_metadata),
        'access_patterns': [meta.get('access_count', 0) for meta in context_metadata]
    }
    
    return insights

# ===================== Utility Functions (Enhanced) =====================
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

def calculate_response_confidence_with_metadata(contexts: List[str], context_metadata: List[Dict], question: str, question_class: Dict) -> float:
    """Enhanced confidence calculation using metadata"""
    if not contexts:
        return 0.0
    
    # Base confidence from context quality
    context_length = sum(len(c.split()) for c in contexts)
    base_confidence = min(context_length / 150, 1.0)  # Adjusted for better scaling
    
    # Metadata-based confidence adjustments
    if context_metadata:
        avg_importance = sum(meta.get('importance_score', 0) for meta in context_metadata) / len(context_metadata)
        avg_factual_density = sum(meta.get('factual_density', 0) for meta in context_metadata) / len(context_metadata)
        avg_qa_confidence = sum(meta.get('qa_confidence', 0) for meta in context_metadata) / len(context_metadata)
        
        metadata_confidence = (avg_importance * 0.4 + avg_factual_density * 0.3 + avg_qa_confidence * 0.3)
        
        # Combine base and metadata confidence
        combined_confidence = (base_confidence * 0.6 + metadata_confidence * 0.4)
    else:
        combined_confidence = base_confidence
    
    # Adjust by question type
    type_multipliers = {
        "factual": 1.0,
        "opinion": 0.7,
        "capability": 0.8,
        "interview": 0.6
    }
    
    multiplier = type_multipliers.get(question_class["type"], 0.7)
    return combined_confidence * multiplier

# ===================== Enhanced API Routes =====================
@app.get("/")
def home():
    return {
        "service": "enhanced-resume-chatbot-with-metadata", 
        "status": "ready",
        "version": "2.1.0",
        "features": [
            "rich_content_metadata",
            "intelligent_classification", 
            "interview_simulation", 
            "confidence_scoring",
            "usage_analytics",
            "quality_assessment"
        ]
    }

@app.get("/health")
async def health():
    try:
        await ensure_data_loaded()
        
        # Enhanced health check with metadata
        chunk_stats = {}
        if RESUME_CHUNKS:
            chunk_stats = {
                'total_chunks': len(RESUME_CHUNKS),
                'sections': list(set(chunk.section for chunk in RESUME_CHUNKS)),
                'avg_importance': sum(chunk.rich_metadata.importance_score for chunk in RESUME_CHUNKS) / len(RESUME_CHUNKS),
                'avg_factual_density': sum(chunk.rich_metadata.factual_density for chunk in RESUME_CHUNKS) / len(RESUME_CHUNKS),
                'total_technical_terms': sum(len(chunk.rich_metadata.technical_terms) for chunk in RESUME_CHUNKS),
                'total_keywords': sum(len(chunk.rich_metadata.keywords) for chunk in RESUME_CHUNKS)
            }
        
        return {
            "ok": True, 
            "profile_completeness": PROFILE.profile_completeness,
            "data_quality_score": PROFILE.data_quality_score,
            "chunk_statistics": chunk_stats,
            "vectorizer_ready": VECTORIZER is not None,
            "embeddings_ready": EMB_MATRIX is not None,
            "ai_ready": bool(GOOGLE_API_KEY),
            "analytics": QUERY_ANALYTICS
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        start_time = time.time()
        await ensure_data_loaded()
        
        question = (req.message or "").strip()
        if not question:
            return ChatResponse(
                reply="👋 สวัสดีครับ! ผมคือ AI Assistant ที่จะช่วยตอบคำถามเกี่ยวกับเรซูเม่ของคุณ Nachapol\n\n🔍 **ตัวอย่างคำถาม:**\n• ข้อมูลพื้นฐาน: ชื่ออะไร? อีเมลคืออะไร?\n• ทักษะและความสามารถ: มีทักษะอะไรบ้าง?\n• การประเมิน: เหมาะกับตำแหน่ง Data Analyst ไหม?\n• สัมภาษณ์งาน: ทำไมสนใจตำแหน่งนี้?\n\n💡 ระบบใช้ metadata เพื่อให้คำตอบที่แม่นยำและเชื่อถือได้มากขึ้น",
                sources=[],
                response_type="greeting",
                confidence=1.0,
                metadata={
                    'system_version': '2.1.0',
                    'features_available': ['metadata_analysis', 'intelligent_classification', 'confidence_scoring']
                }
            )
        
        # Classify the question with enhanced metadata
        question_class = QuestionClassifier.classify_question(question)
        
        # Try quick answer first for factual questions
        if question_class["type"] == "factual":
            quick = get_quick_answer(question)
            if quick:
                return ChatResponse(
                    reply=quick[0], 
                    sources=[],
                    response_type=quick[1],
                    confidence=0.95,
                    metadata={
                        'source': 'quick_answer',
                        'question_classification': question_class,
                        'processing_time': time.time() - start_time
                    }
                )
        
        # Enhanced retrieval with metadata
        hits = multi_query_retrieval_with_metadata(question, question_class, k=6)
        contexts = [hit[1] for hit in hits[:4]]
        context_metadata = [hit[3] for hit in hits[:4]]
        sources = [(hit[0], hit[1][:200] + "..." if len(hit[1]) > 200 else hit[1]) for hit in hits[:3]]
        
        # Calculate enhanced confidence
        confidence = calculate_response_confidence_with_metadata(contexts, context_metadata, question, question_class)
        
        # Generate intelligent response with metadata insights
        reply, response_type, metadata_insights = generate_smart_response_with_metadata(question, contexts, context_metadata, question_class)
        
        # Prepare response metadata
        response_metadata = {
            'question_classification': question_class,
            'context_metadata': metadata_insights,
            'processing_time': time.time() - start_time,
            'sources_used': len(contexts),
            'confidence_factors': {
                'base_confidence': confidence,
                'metadata_boost': metadata_insights.get('confidence_assessment', 0),
                'question_type_adjustment': question_class.get('confidence_threshold', 0.5)
            }
        }
        
        # Add transparency note if needed
        if response_type in ["mixed", "interview_simulation"]:
            if not any(indicator in reply.lower() for indicator in ["📋", "💡", "หมายเหตุ"]):
                if response_type == "interview_simulation":
                    reply += "\n\n*หมายเหตุ: คำตอบนี้เป็นการจำลองการสัมภาษณ์ตามข้อมูลในเรซูเม่*"
                elif confidence < 0.6:
                    reply += "\n\n*หมายเหตุ: คำตอบบางส่วนอาจเป็นการวิเคราะห์จากข้อมูลที่มีอยู่*"
        
        # Update analytics
        processing_time = time.time() - start_time
        QUERY_ANALYTICS['avg_response_time'] = (
            (QUERY_ANALYTICS['avg_response_time'] * (QUERY_ANALYTICS['total_queries'] - 1) + processing_time) / 
            QUERY_ANALYTICS['total_queries']
        )
        QUERY_ANALYTICS['confidence_distribution'].append(confidence)
        
        # Track chunk usage for analytics
        for hit in hits[:4]:
            chunk_idx = hit[0]
            QUERY_ANALYTICS['chunk_usage_stats'][chunk_idx] = QUERY_ANALYTICS['chunk_usage_stats'].get(chunk_idx, 0) + 1
        
        return ChatResponse(
            reply=reply, 
            sources=sources,
            response_type=response_type,
            confidence=confidence,
            metadata=response_metadata
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint failed: {e}")
        return ChatResponse(
            reply=f"เกิดข้อผิดพลาดในระบบ: {str(e)}\n\nกรุณาลองใหม่อีกครั้ง หรือติดต่อผู้ดูแลระบบ",
            sources=[],
            response_type="error",
            confidence=0.0,
            metadata={'error': str(e), 'timestamp': datetime.now().isoformat()}
        )

# ===================== Additional Enhanced API Endpoints =====================
@app.get("/profile")
async def get_profile():
    """Get structured profile data with rich metadata"""
    try:
        await ensure_data_loaded()
        return {
            "basic_info": {
                "name_en": PROFILE.name_en,
                "name_th": PROFILE.name_th,
                "email": PROFILE.email,
                "phone": PROFILE.phone,
                "location": PROFILE.location
            },
            "skills": {
                "all_skills": PROFILE.skills[:25],
                "categories": PROFILE.skill_categories,
                "total_count": len(PROFILE.skills)
            },
            "experience": {
                "structured_data": PROFILE.experience[:5],  # Top 5 most recent
                "total_positions": len(PROFILE.experience),
                "summary": PROFILE.experience_summary
            },
            "education": {
                "structured_data": PROFILE.education,
                "total_degrees": len(PROFILE.education)
            },
            "quality_metrics": {
                "profile_completeness": PROFILE.profile_completeness,
                "data_quality_score": PROFILE.data_quality_score,
                "last_updated": PROFILE.last_updated.isoformat()
            },
            "content_analysis": {
                "total_chunks": len(RESUME_CHUNKS),
                "sections": {
                    section: {
                        "count": len([c for c in RESUME_CHUNKS if c.section == section]),
                        "avg_importance": sum(c.rich_metadata.importance_score for c in RESUME_CHUNKS if c.section == section) / 
                                        max(len([c for c in RESUME_CHUNKS if c.section == section]), 1),
                        "technical_terms": sum(len(c.rich_metadata.technical_terms) for c in RESUME_CHUNKS if c.section == section)
                    }
                    for section in set(chunk.section for chunk in RESUME_CHUNKS)
                }
            }
        }
    except Exception as e:
        logger.error(f"Profile endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metadata/chunks")
async def get_chunks_metadata():
    """Get detailed metadata for all chunks"""
    try:
        await ensure_data_loaded()
        
        chunks_metadata = []
        for i, chunk in enumerate(RESUME_CHUNKS):
            metadata = chunk.get_full_metadata()
            metadata['chunk_id'] = i
            metadata['content_preview'] = chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            chunks_metadata.append(metadata)
        
        return {
            "total_chunks": len(chunks_metadata),
            "chunks": chunks_metadata,
            "summary": {
                "avg_word_count": sum(c['content_stats']['word_count'] for c in chunks_metadata) / len(chunks_metadata) if chunks_metadata else 0,
                "avg_importance": sum(c['content_stats']['importance_score'] for c in chunks_metadata) / len(chunks_metadata) if chunks_metadata else 0,
                "avg_factual_density": sum(c['content_stats']['factual_density'] for c in chunks_metadata) / len(chunks_metadata) if chunks_metadata else 0,
                "total_technical_terms": sum(len(c['semantic_info']['technical_terms']) for c in chunks_metadata),
                "total_entities": sum(len(c['semantic_info']['entities']) for c in chunks_metadata)
            }
        }
    except Exception as e:
        logger.error(f"Chunks metadata endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics")
async def get_analytics():
    """Get system analytics and usage statistics"""
    try:
        await ensure_data_loaded()
        
        # Calculate additional analytics
        chunk_usage_analysis = {}
        if QUERY_ANALYTICS['chunk_usage_stats']:
            most_used_chunks = sorted(QUERY_ANALYTICS['chunk_usage_stats'].items(), 
                                    key=lambda x: x[1], reverse=True)[:5]
            chunk_usage_analysis = {
                'most_accessed_chunks': [
                    {
                        'chunk_id': chunk_id,
                        'access_count': count,
                        'section': RESUME_CHUNKS[chunk_id].section if chunk_id < len(RESUME_CHUNKS) else 'unknown',
                        'importance_score': RESUME_CHUNKS[chunk_id].rich_metadata.importance_score if chunk_id < len(RESUME_CHUNKS) else 0
                    }
                    for chunk_id, count in most_used_chunks
                ]
            }
        
        confidence_stats = {}
        if QUERY_ANALYTICS['confidence_distribution']:
            confidence_stats = {
                'avg_confidence': sum(QUERY_ANALYTICS['confidence_distribution']) / len(QUERY_ANALYTICS['confidence_distribution']),
                'min_confidence': min(QUERY_ANALYTICS['confidence_distribution']),
                'max_confidence': max(QUERY_ANALYTICS['confidence_distribution']),
                'high_confidence_queries': sum(1 for c in QUERY_ANALYTICS['confidence_distribution'] if c > 0.8)
            }
        
        return {
            "query_analytics": QUERY_ANALYTICS,
            "chunk_usage": chunk_usage_analysis,
            "confidence_statistics": confidence_stats,
            "system_performance": {
                "avg_response_time": QUERY_ANALYTICS['avg_response_time'],
                "total_processed_queries": QUERY_ANALYTICS['total_queries']
            },
            "content_quality": {
                "profile_completeness": PROFILE.profile_completeness,
                "data_quality_score": PROFILE.data_quality_score,
                "chunks_with_high_importance": len([c for c in RESUME_CHUNKS if c.rich_metadata.importance_score > 0.7]),
                "chunks_with_recent_data": len([c for c in RESUME_CHUNKS if c.rich_metadata.recency_score > 0.8])
            }
        }
    except Exception as e:
        logger.error(f"Analytics endpoint failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-content-quality")
async def analyze_content_quality():
    """Analyze and report on content quality metrics"""
    try:
        await ensure_data_loaded()
        
        quality_report = {
            "overall_scores": {
                "profile_completeness": PROFILE.profile_completeness,
                "data_quality_score": PROFILE.data_quality_score,
                "content_coverage": len(set(chunk.section for chunk in RESUME_CHUNKS)) / 8  # Expected 8 sections
            },
            "section_analysis": {},
            "improvement_suggestions": [],
            "strengths": [],
            "metadata_insights": {}
        }
        
        # Analyze each section
        for section in set(chunk.section for chunk in RESUME_CHUNKS):
            section_chunks = [c for c in RESUME_CHUNKS if c.section == section]
            
            section_analysis = {
                "chunk_count": len(section_chunks),
                "total_words": sum(c.rich_metadata.word_count for c in section_chunks),
                "avg_importance": sum(c.rich_metadata.importance_score for c in section_chunks) / len(section_chunks),
                "avg_factual_density": sum(c.rich_metadata.factual_density for c in section_chunks) / len(section_chunks),
                "avg_recency": sum(c.rich_metadata.recency_score for c in section_chunks) / len(section_chunks),
                "technical_terms": sum(len(c.rich_metadata.technical_terms) for c in section_chunks),
                "entities": sum(len(c.rich_metadata.entities) for c in section_chunks),
                "question_types_supported": list(set().union(*[c.rich_metadata.question_types for c in section_chunks]))
            }
            
            quality_report["section_analysis"][section] = section_analysis
            
            # Generate suggestions based on analysis
            if section_analysis["avg_factual_density"] < 0.3:
                quality_report["improvement_suggestions"].append(
                    f"Section '{section}' could benefit from more specific, factual details"
                )
            
            if section_analysis["technical_terms"] == 0 and section in ['skills', 'experience']:
                quality_report["improvement_suggestions"].append(
                    f"Section '{section}' lacks technical terminology that could enhance searchability"
                )
            
            # Identify strengths
            if section_analysis["avg_importance"] > 0.8:
                quality_report["strengths"].append(
                    f"Section '{section}' contains high-value content with strong importance scoring"
                )
            
            if section_analysis["avg_recency"] > 0.8:
                quality_report["strengths"].append(
                    f"Section '{section}' contains current and relevant information"
                )
        
        # Overall metadata insights
        all_technical_terms = []
        all_keywords = []
        all_entities = []
        
        for chunk in RESUME_CHUNKS:
            all_technical_terms.extend(chunk.rich_metadata.technical_terms)
            all_keywords.extend(chunk.rich_metadata.keywords)
            all_entities.extend(chunk.rich_metadata.entities)
        
        quality_report["metadata_insights"] = {
            "unique_technical_terms": len(set(all_technical_terms)),
            "unique_keywords": len(set(all_keywords)),
            "total_entities": len(all_entities),
            "entity_types": list(set(entity.get('type', 'unknown') for entity in all_entities)),
            "most_common_technical_terms": [
                (term, all_technical_terms.count(term)) 
                for term in set(all_technical_terms)
            ][:10],
            "question_answering_coverage": len(set().union(*[c.rich_metadata.question_types for c in RESUME_CHUNKS]))
        }
        
        return quality_report
        
    except Exception as e:
        logger.error(f"Content quality analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/optimize-retrieval")
async def optimize_retrieval_system():
    """Analyze and optimize the retrieval system based on usage patterns"""
    try:
        await ensure_data_loaded()
        
        optimization_report = {
            "current_performance": {},
            "recommendations": [],
            "configuration_suggestions": {},
            "chunk_reranking": []
        }
        
        # Analyze current performance
        if QUERY_ANALYTICS['chunk_usage_stats']:
            usage_distribution = list(QUERY_ANALYTICS['chunk_usage_stats'].values())
            optimization_report["current_performance"] = {
                "total_chunk_accesses": sum(usage_distribution),
                "avg_accesses_per_chunk": sum(usage_distribution) / len(usage_distribution),
                "usage_variance": np.var(usage_distribution) if len(usage_distribution) > 1 else 0,
                "underutilized_chunks": len([c for c in RESUME_CHUNKS if c.access_count == 0]),
                "overutilized_chunks": len([usage for usage in usage_distribution if usage > np.mean(usage_distribution) * 2])
            }
        
        # Generate recommendations
        underutilized_chunks = [i for i, chunk in enumerate(RESUME_CHUNKS) if chunk.access_count == 0]
        if len(underutilized_chunks) > len(RESUME_CHUNKS) * 0.3:  # More than 30% underutilized
            optimization_report["recommendations"].append(
                "Consider reviewing content chunking strategy - many chunks are never accessed"
            )
        
        # Check for chunks with high importance but low usage
        high_importance_low_usage = [
            (i, chunk) for i, chunk in enumerate(RESUME_CHUNKS)
            if chunk.rich_metadata.importance_score > 0.8 and chunk.access_count < 2
        ]
        
        if high_importance_low_usage:
            optimization_report["recommendations"].append(
                f"Found {len(high_importance_low_usage)} high-importance chunks with low usage - consider boosting their retrieval scores"
            )
            
            optimization_report["chunk_reranking"] = [
                {
                    "chunk_id": chunk_id,
                    "section": chunk.section,
                    "importance_score": chunk.rich_metadata.importance_score,
                    "current_access_count": chunk.access_count,
                    "suggested_boost": 0.2
                }
                for chunk_id, chunk in high_importance_low_usage[:5]
            ]
        
        # Configuration suggestions based on query patterns
        question_type_distribution = QUERY_ANALYTICS.get('question_types', {})
        if question_type_distribution:
            most_common_type = max(question_type_distribution.items(), key=lambda x: x[1])[0]
            optimization_report["configuration_suggestions"] = {
                "most_common_question_type": most_common_type,
                "suggested_default_boost": f"Boost {most_common_type}-related chunks by 0.1",
                "retrieval_k_recommendation": min(8, max(4, len(RESUME_CHUNKS) // 3))
            }
        
        return optimization_report
        
    except Exception as e:
        logger.error(f"Retrieval optimization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ===================== Utility Functions (Continued) =====================
async def fetch_resume_text() -> str:
    """Fetch resume content from URL with better error handling"""
    if not RESUME_URL:
        logger.warning("No RESUME_URL provided")
        return ""
        
    headers = {"User-Agent": "resume-bot/2.1.0"}
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
        
        # Enhanced TF-IDF with metadata-aware features
        VECTORIZER = TfidfVectorizer(
            min_df=1, 
            ngram_range=(1,2), 
            max_features=1200,  # Increased for better coverage
            stop_words=None,  # Keep all words for multilingual support
            analyzer='word',
            lowercase=True
        )
        TFIDF_MATRIX = VECTORIZER.fit_transform(contents)
        
        # Embeddings with enhanced error handling
        EMB_MATRIX = embed_texts(contents)
        if EMB_MATRIX is not None and EMB_MATRIX.size > 0:
            norms = np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True) + 1e-12
            EMB_MATRIX = EMB_MATRIX / norms
        
        logger.info(f"Built enhanced search index with {len(contents)} chunks, "
                   f"TF-IDF shape: {TFIDF_MATRIX.shape}, Embedding shape: {EMB_MATRIX.shape}")
        
    except Exception as e:
        logger.error(f"Error building search index: {e}")

async def ensure_data_loaded(force: bool = False):
    """Ensure resume data is loaded and indexed with comprehensive error handling"""
    global PROFILE, RESUME_CHUNKS, LAST_FETCH_AT
    
    if RESUME_CHUNKS and not force and (time.time() - LAST_FETCH_AT < 3600):
        return
    
    try:
        text = await fetch_resume_text()
        if text:
            PROFILE, RESUME_CHUNKS = parse_structured_resume(text)
            build_search_index()
            LAST_FETCH_AT = time.time()
            
            # Generate summary statistics
            total_words = sum(chunk.rich_metadata.word_count for chunk in RESUME_CHUNKS)
            avg_importance = sum(chunk.rich_metadata.importance_score for chunk in RESUME_CHUNKS) / len(RESUME_CHUNKS) if RESUME_CHUNKS else 0
            
            logger.info(f"Data loaded successfully: {len(RESUME_CHUNKS)} chunks, "
                       f"{total_words} total words, avg importance: {avg_importance:.2f}, "
                       f"profile completeness: {PROFILE.profile_completeness:.2f}")
        else:
            logger.warning("No resume text fetched")
    except Exception as e:
        logger.error(f"Error ensuring data loaded: {e}")

def get_quick_answer(question: str) -> Optional[Tuple[str, str]]:
    """Enhanced quick answers with metadata support"""
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
        
        # Contact info with enhanced responses
        if "email" in q_lower or "อีเมล" in q_lower:
            if PROFILE.email:
                return f"อีเมล: {PROFILE.email}", "factual"
        
        if "phone" in q_lower or "โทร" in q_lower or "เบอร์" in q_lower:
            if PROFILE.phone:
                return f"เบอร์โทร: {PROFILE.phone}", "factual"
        
        if "location" in q_lower or "ที่อยู่" in q_lower or "address" in q_lower:
            if PROFILE.location:
                return f"ที่ตั้ง: {PROFILE.location}", "factual"
        
        # Skills summary
        if "skill" in q_lower and "summary" in q_lower:
            if PROFILE.skills:
                skill_count = len(PROFILE.skills)
                tech_skills = len(PROFILE.skill_categories.get('technical', []))
                return f"มีทักษะทั้งหมด {skill_count} ทักษะ รวมทักษะด้านเทคนิค {tech_skills} ทักษะ", "factual"
        
        return None
        
    except Exception as e:
        logger.error(f"Error in get_quick_answer: {e}")
        return None

def handle_no_context_response(question: str, question_class: Dict) -> Tuple[str, str]:
    """Enhanced no-context response handling"""
    
    if question_class["type"] == "factual":
        return "ขออภัย ไม่พบข้อมูลที่เกี่ยวข้องในเรซูเม่ กรุณาลองถามในรูปแบบอื่น", "no_context"
    
    elif question_class["type"] == "interview":
        # Enhanced general interview response
        model = genai.GenerativeModel(MODEL_NAME)
        prompt = f"""
คุณกำลังตอบคำถามสัมภาษณ์ในฐานะผู้สมัครงานด้านการปรับปรุงกระบวนการและการวิเคราะห์ข้อมูล

ให้ตอบเป็นคำตอบทั่วไปที่เหมาะสมสำหรับผู้สมัครงาน พร้อมระบุว่าเป็นคำตอบทั่วไป

**คำถาม:** {question}

*หมายเหตุ: คำตอบนี้เป็นการจำลองทั่วไป ไม่ใช่ข้อมูลเฉพาะจากเรซูเม่*

ตอบในลักษณะผู้สมัครงานที่มีประสบการณ์ด้าน process improvement
"""
        try:
            response = model.generate_content(prompt)
            answer = getattr(response, "text", "").strip()
            return answer or "ขออภัย ไม่สามารถตอบคำถามนี้ได้", "general_interview"
        except:
            return "ขออภัย คำถามนี้ต้องการข้อมูลเฉพาะที่ไม่มีในเรซูเม่", "no_context"
    
    else:
        return "ไม่พบข้อมูลที่เกี่ยวข้องในเรซูเม่ กรุณาลองถามในรูปแบบอื่น หรือถามคำถามที่เฉพาะเจาะจงมากขึ้น", "no_context"

# ===================== Final API Routes =====================
@app.get("/debug")
async def debug():
    try:
        await ensure_data_loaded()
        return {
            "chunks_count": len(RESUME_CHUNKS),
            "sections": [(i, chunk.section, chunk.rich_metadata.importance_score) for i, chunk in enumerate(RESUME_CHUNKS)],
            "profile": {
                "name_en": PROFILE.name_en,
                "name_th": PROFILE.name_th,
                "email": PROFILE.email,
                "skills_count": len(PROFILE.skills),
                "completeness": PROFILE.profile_completeness,
                "quality_score": PROFILE.data_quality_score
            },
            "system_status": {
                "last_fetch": LAST_FETCH_AT,
                "vectorizer_features": VECTORIZER.get_feature_names_out()[:20].tolist() if VECTORIZER else [],
                "embedding_dimensions": EMB_MATRIX.shape[1] if EMB_MATRIX is not None else 0
            },
            "metadata_features": {
                "intelligent_classification": True,
                "rich_content_metadata": True,
                "usage_analytics": True,
                "quality_assessment": True
            }
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
            "profile_quality": PROFILE.data_quality_score,
            "message": "Data refreshed successfully with enhanced metadata"
        }
    except Exception as e:
        logger.error(f"Refresh failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
