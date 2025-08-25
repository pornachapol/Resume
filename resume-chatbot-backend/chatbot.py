import re, datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import json

@dataclass
class TimelineEvent:
    start_date: Optional[datetime.date] = None
    end_date: Optional[datetime.date] = None
    duration_months: int = 0
    is_current: bool = False
    date_precision: str = "month"  # year, month, day
    
    def __post_init__(self):
        if self.start_date and self.end_date:
            self.duration_months = (self.end_date.year - self.start_date.year) * 12 + \
                                 (self.end_date.month - self.start_date.month)
        elif self.start_date and self.is_current:
            today = datetime.date.today()
            self.duration_months = (today.year - self.start_date.year) * 12 + \
                                 (today.month - self.start_date.month)

@dataclass
class SkillEntity:
    name: str
    category: str  # technical, soft, language, certification, tool
    proficiency: str = "mentioned"  # mentioned, basic, intermediate, advanced, expert
    context: List[str] = field(default_factory=list)
    years_experience: Optional[int] = None
    
@dataclass
class Achievement:
    title: str
    metric: Optional[str] = None
    value: Optional[str] = None
    context: str = ""
    timeline: Optional[TimelineEvent] = None

@dataclass 
class EnhancedResumeChunk:
    content: str
    section: str
    subsection: str = ""
    timeline: Optional[TimelineEvent] = None
    entities: List[SkillEntity] = field(default_factory=list)
    achievements: List[Achievement] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Auto-extract keywords
        self.keywords = self._extract_keywords()
        
    def _extract_keywords(self) -> List[str]:
        """Extract important keywords from content"""
        # Industry terms, tools, methodologies
        keyword_patterns = [
            r'\b(?:Excel|SQL|Python|JavaScript|Power BI|ETL|RPA|AGV|SLA|UAT|BRD)\b',
            r'\b(?:Lean|Six Sigma|Automation|Analytics|Dashboard|Process)\b',
            r'\b(?:Project Management|Team Leadership|Business Analysis)\b',
            r'\b(?:Manufacturing|Insurance|Retail|Supply Chain)\b'
        ]
        
        keywords = []
        for pattern in keyword_patterns:
            matches = re.findall(pattern, self.content, re.IGNORECASE)
            keywords.extend([m.lower() for m in matches])
        
        return list(set(keywords))

@dataclass
class EnhancedProfileData:
    # Basic info
    name_en: str = ""
    name_th: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    
    # Skills taxonomy
    skills: Dict[str, List[SkillEntity]] = field(default_factory=lambda: defaultdict(list))
    
    # Timeline-based data
    experience: List[Dict[str, Any]] = field(default_factory=list)
    education: List[Dict[str, Any]] = field(default_factory=list)
    
    # Achievements and metrics
    achievements: List[Achievement] = field(default_factory=list)
    
    # Career progression
    career_timeline: List[TimelineEvent] = field(default_factory=list)
    total_experience_years: float = 0
    
    # Domain expertise
    industry_experience: Dict[str, int] = field(default_factory=dict)  # industry -> months
    role_progression: List[str] = field(default_factory=list)

class EnhancedResumeParser:
    def __init__(self):
        # Skill taxonomy
        self.skill_categories = {
            'technical': [
                'python', 'sql', 'javascript', 'excel', 'power bi', 'etl', 
                'rpa', 'automation', 'agv', 'lean six sigma', 'analytics'
            ],
            'tools': [
                'excel', 'power bi', 'javascript', 'macro', 'dashboard',
                'etl', 'agv', 'rpa'
            ],
            'methodologies': [
                'lean', 'six sigma', 'process improvement', 'project management',
                'business analysis', 'uât', 'brd'
            ],
            'soft': [
                'leadership', 'team management', 'stakeholder management',
                'cross-functional', 'collaboration'
            ],
            'languages': ['thai', 'english'],
            'certifications': ['lean six sigma green belt']
        }
        
        # Industry mapping
        self.industry_keywords = {
            'insurance': ['claim', 'reimbursement', 'generali', 'sla'],
            'manufacturing': ['production', 'kubota', 'agv', 'supply chain'],
            'retail': ['vending', 'commission', 'shinning gold'],
            'consulting': ['ngg enterprise', 'business transformation']
        }
        
        # Date patterns
        self.date_patterns = [
            r'(?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(?P<year>\d{4})',
            r'(?P<month>\d{1,2})/(?P<year>\d{4})',
            r'(?P<year>\d{4})',
        ]
    
    def parse_date(self, date_str: str) -> Optional[datetime.date]:
        """Parse various date formats"""
        date_str = date_str.strip().replace(',', '')
        
        month_map = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        for pattern in self.date_patterns:
            match = re.search(pattern, date_str, re.IGNORECASE)
            if match:
                groups = match.groupdict()
                year = int(groups['year'])
                
                if 'month' in groups and groups['month']:
                    if groups['month'].isdigit():
                        month = int(groups['month'])
                    else:
                        month = month_map.get(groups['month'].lower()[:3], 1)
                else:
                    month = 1
                    
                try:
                    return datetime.date(year, month, 1)
                except ValueError:
                    continue
        
        return None
    
    def extract_timeline(self, text: str) -> Optional[TimelineEvent]:
        """Extract timeline information from text"""
        # Look for date ranges
        date_range_pattern = r'(?P<start>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2}/)\s*\d{4})\s*[-–]\s*(?P<end>(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|\d{1,2}/)\s*\d{4}|Present)'
        
        match = re.search(date_range_pattern, text, re.IGNORECASE)
        if match:
            start_str = match.group('start')
            end_str = match.group('end')
            
            start_date = self.parse_date(start_str)
            
            if end_str.lower() in ['present', 'current']:
                end_date = None
                is_current = True
            else:
                end_date = self.parse_date(end_str)
                is_current = False
            
            return TimelineEvent(
                start_date=start_date,
                end_date=end_date,
                is_current=is_current
            )
        
        return None
    
    def categorize_skill(self, skill: str) -> str:
        """Categorize a skill"""
        skill_lower = skill.lower()
        
        for category, keywords in self.skill_categories.items():
            if any(keyword in skill_lower for keyword in keywords):
                return category
        
        return 'other'
    
    def extract_achievements(self, text: str) -> List[Achievement]:
        """Extract quantified achievements"""
        achievements = []
        
        # Patterns for achievements with metrics
        achievement_patterns = [
            r'(\d+)%\+?\s*(improvement|increase|reduction|saved)',
            r'(\d+(?:,\d+)*)\s*(THB|USD|baht)/year\s*saved',
            r'(\d+)\s*(kg|tons?|hours?|days?)\s*(reduction|saved|improved)',
            r'(led|implemented|achieved|reduced|increased)\s+([^.]+?)(?:\.|$)'
        ]
        
        for pattern in achievement_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                if match.group(1).isdigit():
                    # Quantified achievement
                    value = match.group(1)
                    metric = match.group(2) if len(match.groups()) > 1 else ""
                    achievement = Achievement(
                        title=match.group(0).strip(),
                        metric=metric,
                        value=value,
                        context=text[:100]
                    )
                else:
                    # Qualitative achievement
                    achievement = Achievement(
                        title=match.group(0).strip(),
                        context=text[:100]
                    )
                
                achievements.append(achievement)
        
        return achievements
    
    def parse_structured_resume(self, text: str) -> Tuple[EnhancedProfileData, List[EnhancedResumeChunk]]:
        """Enhanced parsing with rich metadata"""
        profile = EnhancedProfileData()
        chunks = []
        
        lines = [l.strip() for l in text.split('\n') if l.strip()]
        
        # Better section detection based on actual resume structure
        section_patterns = {
            'header': r'^[A-Z\s]{10,}$|contact.*info|name|email|phone',
            'summary': r'process improvement.*leader|professional.*summary',
            'expertise': r'area.*expertise|key.*skills|expertise',
            'experience': r'professional.*experience|experience|work.*history',
            'education': r'education|academic|degree|university',
            'achievements': r'key.*achievements?|accomplishments?',
            'additional': r'additional.*information|languages?|certifications?'
        }
        
        current_section = "general"
        section_content = []
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Detect section changes
            new_section = None
            for section, pattern in section_patterns.items():
                if re.search(pattern, line_lower):
                    new_section = section
                    break
            
            # Also detect by formatting (all caps, etc.)
            if line.isupper() and len(line) > 10:
                if any(keyword in line_lower for keyword in ['experience', 'education', 'expertise']):
                    for section in section_patterns.keys():
                        if section in line_lower:
                            new_section = section
                            break
            
            if new_section and new_section != current_section:
                # Save previous section
                if section_content:
                    content_text = '\n'.join(section_content)
                    
                    # Extract timeline for this section
                    timeline = self.extract_timeline(content_text)
                    
                    # Extract achievements
                    achievements = self.extract_achievements(content_text)
                    
                    # Extract skills
                    skills = []
                    for category, keywords in self.skill_categories.items():
                        for keyword in keywords:
                            if keyword.lower() in content_text.lower():
                                skill = SkillEntity(
                                    name=keyword,
                                    category=category,
                                    context=[current_section]
                                )
                                skills.append(skill)
                    
                    chunk = EnhancedResumeChunk(
                        content=content_text,
                        section=current_section,
                        timeline=timeline,
                        entities=skills,
                        achievements=achievements,
                        metadata={
                            'line_count': len(section_content),
                            'char_count': len(content_text),
                            'has_timeline': timeline is not None,
                            'achievement_count': len(achievements)
                        }
                    )
                    chunks.append(chunk)
                
                current_section = new_section
                section_content = [line]
            else:
                section_content.append(line)
        
        # Save last section
        if section_content:
            content_text = '\n'.join(section_content)
            timeline = self.extract_timeline(content_text)
            achievements = self.extract_achievements(content_text)
            
            chunk = EnhancedResumeChunk(
                content=content_text,
                section=current_section,
                timeline=timeline,
                achievements=achievements,
                metadata={'line_count': len(section_content)}
            )
            chunks.append(chunk)
        
        # Extract profile data with enhanced metadata
        self._extract_profile_data(profile, chunks, text)
        
        return profile, chunks
    
    def _extract_profile_data(self, profile: EnhancedProfileData, chunks: List[EnhancedResumeChunk], full_text: str):
        """Extract enhanced profile data"""
        # Basic contact info
        email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', full_text)
        if email_match:
            profile.email = email_match.group(1)
        
        phone_match = re.search(r'(\d{3}-\d{3}-\d{4})', full_text)
        if phone_match:
            profile.phone = phone_match.group(1)
        
        # Extract name from first meaningful line
        first_line = full_text.split('\n')[0].strip()
        if not any(char in first_line for char in ['@', '|', 'Thailand']):
            profile.name_en = first_line
        
        # Build skill taxonomy
        for chunk in chunks:
            for entity in chunk.entities:
                profile.skills[entity.category].append(entity)
        
        # Extract experience timeline
        experience_chunks = [c for c in chunks if c.section == 'experience']
        career_events = []
        
        for chunk in experience_chunks:
            if chunk.timeline:
                # Extract company and role
                lines = chunk.content.split('\n')
                for line in lines:
                    if any(company in line.lower() for company in ['kubota', 'shinning', 'ngg', 'generali']):
                        # Parse role and company
                        role_match = re.search(r'([^,]+),\s*([^,\n]+)', line)
                        if role_match:
                            role = role_match.group(1).strip()
                            company = role_match.group(2).strip()
                            
                            exp_record = {
                                'role': role,
                                'company': company,
                                'timeline': chunk.timeline,
                                'achievements': chunk.achievements,
                                'skills': [e.name for e in chunk.entities]
                            }
                            profile.experience.append(exp_record)
                            career_events.append(chunk.timeline)
        
        # Calculate total experience
        total_months = sum(event.duration_months for event in career_events if event.duration_months)
        profile.total_experience_years = round(total_months / 12, 1)
        
        # Industry experience mapping
        for exp in profile.experience:
            company = exp['company'].lower()
            months = exp['timeline'].duration_months if exp['timeline'] else 0
            
            for industry, keywords in self.industry_keywords.items():
                if any(keyword in company for keyword in keywords):
                    profile.industry_experience[industry] = profile.industry_experience.get(industry, 0) + months
        
        # Role progression
        profile.role_progression = [exp['role'] for exp in sorted(profile.experience, 
                                   key=lambda x: x['timeline'].start_date if x['timeline'] and x['timeline'].start_date else datetime.date.min)]
        
        # Collect all achievements
        for chunk in chunks:
            profile.achievements.extend(chunk.achievements)

# Usage example and integration helper
class EnhancedResumeManager:
    def __init__(self):
        self.parser = EnhancedResumeParser()
        self.profile: Optional[EnhancedProfileData] = None
        self.chunks: List[EnhancedResumeChunk] = []
        self.metadata_index = {}
        
    def load_resume(self, text: str):
        """Load and parse resume with enhanced features"""
        self.profile, self.chunks = self.parser.parse_structured_resume(text)
        self._build_metadata_index()
    
    def _build_metadata_index(self):
        """Build searchable metadata index"""
        self.metadata_index = {
            'skills_by_category': defaultdict(list),
            'timeline_events': [],
            'achievements_by_metric': defaultdict(list),
            'keywords_by_section': defaultdict(set),
            'industry_context': defaultdict(list)
        }
        
        for chunk in self.chunks:
            # Index skills by category
            for entity in chunk.entities:
                self.metadata_index['skills_by_category'][entity.category].append(entity.name)
            
            # Index timeline events
            if chunk.timeline:
                self.metadata_index['timeline_events'].append({
                    'section': chunk.section,
                    'timeline': chunk.timeline,
                    'content': chunk.content[:200]
                })
            
            # Index achievements
            for achievement in chunk.achievements:
                if achievement.metric:
                    self.metadata_index['achievements_by_metric'][achievement.metric].append(achievement)
            
            # Index keywords
            self.metadata_index['keywords_by_section'][chunk.section].update(chunk.keywords)
    
    def get_contextual_information(self, query_type: str, specific_topic: str = None) -> Dict[str, Any]:
        """Get rich contextual information for query processing"""
        context = {
            'profile_summary': {
                'name': self.profile.name_en,
                'total_experience': self.profile.total_experience_years,
                'industries': list(self.profile.industry_experience.keys()),
                'current_role': self.profile.role_progression[-1] if self.profile.role_progression else None
            },
            'timeline_context': self.metadata_index['timeline_events'],
            'skill_taxonomy': dict(self.metadata_index['skills_by_category']),
            'quantified_achievements': [
                a for achievements in self.metadata_index['achievements_by_metric'].values() 
                for a in achievements
            ]
        }
        
        if specific_topic:
            # Add topic-specific context
            topic_lower = specific_topic.lower()
            relevant_chunks = []
            
            for chunk in self.chunks:
                if (topic_lower in chunk.content.lower() or 
                    any(topic_lower in keyword for keyword in chunk.keywords)):
                    relevant_chunks.append({
                        'section': chunk.section,
                        'content': chunk.content,
                        'relevance_score': self._calculate_relevance(chunk, topic_lower)
                    })
            
            context['topic_specific'] = sorted(relevant_chunks, 
                                             key=lambda x: x['relevance_score'], 
                                             reverse=True)[:3]
        
        return context
    
    def _calculate_relevance(self, chunk: EnhancedResumeChunk, topic: str) -> float:
        """Calculate relevance score for topic"""
        content_lower = chunk.content.lower()
        score = 0
        
        # Direct mention
        if topic in content_lower:
            score += 2
        
        # Keyword match
        if any(topic in keyword for keyword in chunk.keywords):
            score += 1.5
        
        # Section relevance
        if chunk.section in ['experience', 'expertise'] and topic in content_lower:
            score += 1
        
        # Timeline bonus (recent experience is more relevant)
        if chunk.timeline and chunk.timeline.is_current:
            score += 0.5
        
        return score
    
    def export_metadata(self) -> Dict[str, Any]:
        """Export all metadata for debugging/analysis"""
        return {
            'profile': {
                'basic_info': {
                    'name': self.profile.name_en,
                    'email': self.profile.email,
                    'phone': self.profile.phone,
                    'total_experience_years': self.profile.total_experience_years
                },
                'career_progression': self.profile.role_progression,
                'industry_experience': self.profile.industry_experience,
                'skills_by_category': {k: [s.name for s in v] for k, v in self.profile.skills.items()}
            },
            'chunks_metadata': [
                {
                    'section': chunk.section,
                    'has_timeline': chunk.timeline is not None,
                    'timeline_summary': {
                        'duration_months': chunk.timeline.duration_months,
                        'is_current': chunk.timeline.is_current
                    } if chunk.timeline else None,
                    'keywords': chunk.keywords,
                    'achievements_count': len(chunk.achievements),
                    'entities_count': len(chunk.entities)
                }
                for chunk in self.chunks
            ],
            'search_index': dict(self.metadata_index)
        }

# Integration with existing code
def integrate_enhanced_parser():
    """Helper function showing how to integrate with existing chatbot"""
    enhanced_manager = EnhancedResumeManager()
    
    # Replace the existing parse_structured_resume function
    def enhanced_parse_structured_resume(text: str):
        enhanced_manager.load_resume(text)
        
        # Convert to original format for compatibility
        original_chunks = []
        for enhanced_chunk in enhanced_manager.chunks:
            # Create ResumeChunk with enhanced metadata
            original_chunk = type('ResumeChunk', (), {
                'content': enhanced_chunk.content,
                'section': enhanced_chunk.section,
                'metadata': {
                    **enhanced_chunk.metadata,
                    'timeline': enhanced_chunk.timeline,
                    'keywords': enhanced_chunk.keywords,
                    'achievements': enhanced_chunk.achievements,
                    'entities': enhanced_chunk.entities
                }
            })()
            original_chunks.append(original_chunk)
        
        # Create enhanced ProfileData
        enhanced_profile = type('ProfileData', (), {
            'name_en': enhanced_manager.profile.name_en,
            'name_th': enhanced_manager.profile.name_th,
            'email': enhanced_manager.profile.email,
            'phone': enhanced_manager.profile.phone,
            'location': enhanced_manager.profile.location,
            'skills': [skill.name for skills in enhanced_manager.profile.skills.values() for skill in skills],
            'total_experience_years': enhanced_manager.profile.total_experience_years,
            'industry_experience': enhanced_manager.profile.industry_experience,
            'enhanced_metadata': enhanced_manager.export_metadata()
        })()
        
        return enhanced_profile, original_chunks
    
    return enhanced_parse_structured_resume, enhanced_manager
