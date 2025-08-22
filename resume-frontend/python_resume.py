import os
import requests
import streamlit as st
from PIL import Image
import time

# Page configuration
st.set_page_config(
    page_title="Nachapol Roc-anusorn | Resume",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling - consolidated and cleaned up
st.markdown(
    """
    <style>
    .stApp {
        /* Theme variables */
        --color-bg: #202020;
        --color-panel: #303030;
        --color-text: #ffffff;
        --color-muted: #cccccc;
        --color-accent: #4cb5f9;
        --color-accent-2: #78d88f;
        --color-border: #505050;
        --color-chip: #404040;
        background-color: var(--color-bg);
        color: var(--color-text);
    }
    
    /* Main sections styling */
    .main-header {
        padding: 2rem 0;
        background-color: #303030;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
    }
    
    .bot-bubble ul, .bot-bubble ol {
      margin: 0;             /* ตัด margin เริ่มต้น */
      padding-left: 18px;    /* ระยะ indent */
    }
    
    .bot-bubble li {
      margin: 2px 0;         /* ลดช่องว่างระหว่างแต่ละ bullet */
      line-height: 1.35;     /* ความสูงบรรทัดกระชับ */
    }

    .content-section {
        background-color: var(--color-panel);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
        margin-bottom: 1.5rem;
        color: var(--color-text);
    }
    
    .section-title {
        color: var(--color-text);
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid var(--color-border);
    }
    
    .job-title {
        color: var(--color-text);
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    
    .job-period {
        color: var(--color-muted);
        font-size: 0.9rem;
        font-style: italic;
        margin-bottom: 0.8rem;
    }
    
    .download-button {
        background-color: #ffffff;
        color: #202020 !important;
        padding: 0.6rem 1.2rem;
        text-decoration: none;
        border-radius: 6px;
        font-weight: 500;
        display: inline-block;
        text-align: center;
        margin: 1rem 0;
        transition: background-color 0.3s;
    }
    
    .download-button:hover {
        background-color: #e0e0e0;
    }
    
    .profile-image {
        border: 3px solid var(--color-chip);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        border-radius: 12px;
        width: 200px;
        max-width: 100%;
        height: auto;
    }
    
    .skill-tag {
        background-color: var(--color-chip);
        color: var(--color-text);
        border-radius: 20px;
        padding: 5px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        display: inline-block;
        font-size: 0.85rem;
    }
    
    .project-card {
        background-color: var(--color-chip);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid var(--color-accent);
    }
    
    .highlight {
        color: var(--color-accent);
        font-weight: 500;
    }
    
    .award-badge {
        background-color: var(--color-accent);
        color: #202020;
        border-radius: 4px;
        padding: 3px 8px;
        margin-right: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    .output-badge {
        background-color: var(--color-accent-2);
        color: #202020;
        border-radius: 4px;
        padding: 3px 8px;
        margin-right: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Base text and link styles */
    .stApp, .stApp p, .stApp li, .stApp div, .stApp span {
        color: var(--color-text);
    }
    a {
        color: var(--color-accent) !important;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Streamlit markdown/text blocks */
    .stMarkdown, .stText { color: var(--color-text); }
    
    /* Remove default Streamlit background color from elements */
    div[data-testid="stVerticalBlock"] {
        background-color: transparent;
    }
    
    /* Chat styling - consolidated */
    .chat-container {
        max-width: 600px;
        margin: auto;
    }

    .chat-row {
        display: flex;
        gap: 10px;
        align-items: flex-end;
        margin: 10px 0;
    }
    
    .chat-row.user {
        justify-content: flex-end;
    }
    
    .chat-row.bot {
        justify-content: flex-start;
    }

    .chat-row .avatar {
        width: 36px;
        height: 36px;
        border-radius: 50%;
        background: #444;
        flex: 0 0 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #fff;
        font-weight: 700;
        font-size: 14px;
        box-shadow: 0 2px 6px rgba(0,0,0,.25);
        overflow: hidden;
    }
    
    .chat-row .avatar img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    
    .chat-row.user .avatar {
        order: 2;
    }
    
    .chat-row.user .bubble {
        order: 1;
    }
    
    .chat-row.bot .avatar {
        order: 1;
    }
    
    .chat-row.bot .bubble {
        order: 2;
    }

    .bubble {
        max-width: 70vw;
        padding: 12px 14px;
        border-radius: 16px;
        line-height: 1.45;
        font-size: 0.95rem;
        box-shadow: 0 2px 8px rgba(0,0,0,.25);
        word-wrap: break-word;
        white-space: pre-wrap;
    }
    
    .user-bubble {
        background: linear-gradient(180deg,#0057ff,#1a73e8);
        color: #ffffff !important;
        border-bottom-right-radius: 6px;
    }
    
    .bot-bubble {
        background: #404040;
        color: #ffffff !important;
        border-bottom-left-radius: 6px;
    }
    
    /* Force text color in chat bubbles */
    .user-bubble * {
        color: #ffffff !important;
    }
    
    .bot-bubble * {
        color: #ffffff !important;
    }
    
    /* Clear button styling */
    div[data-testid="stButton"].clear-chat-btn > button {
      background-color: #ffffff;
      color: #202020 !important;
      padding: 0.6rem 1.2rem;
      border-radius: 6px;
      font-weight: 500;
      text-align: center;
      margin: 1rem 0;
      transition: background-color 0.3s ease;
      border: none;
      box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    }
    div[data-testid="stButton"].clear-chat-btn > button:hover {
      background-color: #e0e0e0;
    }
    .clear-btn .stButton > button:hover {
      background-color: #e0e0e0;
    }
    
    .stButton > button:hover {
        color: #ffffff !important;
        background-color: #ff4444 !important;
        border: 1px solid #ff4444 !important;
    }
    
    /* Sticky chat input */
    div[data-testid="stChatInput"] {
        position: fixed;
        left: 5%;
        right: 5%;
        bottom: 24px;
        z-index: 999;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
        margin-bottom: 40px !important;
    }
    
    div[data-testid="stChatInput"] textarea {
        min-height: 36px !important;
        font-size: 0.95rem !important;
        padding: 6px 12px !important;
        border-radius: 18px !important;
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    
    div[data-testid="stChatInput"] textarea::placeholder {
        color: #666666 !important;
    }
    
    main .block-container {
        padding-bottom: 120px;
    }
    
    .chat-row:last-child {
        margin-bottom: 30px !important;
    }
    
    /* Hide source chips */
    .src-chip {
        display: none !important;
    }
    
    /* Responsive design fixes */
    @media (max-width: 768px) {
        .profile-image {
            width: 150px;
        }
        
        .bubble {
            max-width: 85vw;
            font-size: 0.9rem;
        }
        
        div[data-testid="stChatInput"] {
            left: 2%;
            right: 2%;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Helper functions
def get_backend_url():
    """Get backend URL from environment or secrets with fallback"""
    try:
        # Try environment variable first
        backend_url = os.getenv("BACKEND_URL")
        if backend_url:
            return backend_url
        
        # Try Streamlit secrets
        if hasattr(st, 'secrets') and "BACKEND_URL" in st.secrets:
            return st.secrets["BACKEND_URL"]
        
        # Fallback URL (you should replace this with your actual backend URL)
        return "https://your-backend-url.com"
    except Exception:
        return "https://your-backend-url.com"

def avatar_html(url=None, text_fallback="U"):
    """Generate avatar HTML with error handling"""
    if url and url.strip():
        return f'<div class="avatar"><img src="{url}" alt="Avatar" onerror="this.style.display=\'none\'"/></div>'
    return f'<div class="avatar">{text_fallback}</div>'

def safe_request(url, data, timeout=30):
    """Make HTTP request with proper error handling"""
    try:
        response = requests.post(url, json=data, timeout=timeout)
        response.raise_for_status()  # Raises HTTPError for bad status codes
        return response.json()
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to server. Please check your connection."}
    except requests.exceptions.HTTPError as e:
        return {"error": f"Server error: {e.response.status_code}"}
    except requests.exceptions.RequestException:
        return {"error": "An error occurred while processing your request."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

# Header Section
with st.container():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        try:
            st.markdown(
                """
                <div style='display: flex; justify-content: center; padding: 1rem;'>
                    <img src='https://raw.githubusercontent.com/pornachapol/Resume/main/assets/profile_picture.jpeg' 
                         class='profile-image' alt='Profile Picture' 
                         onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjAwIiBoZWlnaHQ9IjIwMCIgdmlld0JveD0iMCAwIDIwMCAyMDAiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGNpcmNsZSBjeD0iMTAwIiBjeT0iMTAwIiByPSIxMDAiIGZpbGw9IiM0MDQwNDAiLz48dGV4dCB4PSIxMDAiIHk9IjEwNSIgZm9udC1mYW1pbHk9IkFyaWFsIiBmb250LXNpemU9IjQ4IiBmaWxsPSIjZmZmIiB0ZXh0LWFuY2hvcj0ibWlkZGxlIj5OPC90ZXh0Pjwvc3ZnPg=='; this.onerror=null;"/>
                </div>
                """,
                unsafe_allow_html=True
            )
        except Exception:
            # Fallback if image fails
            st.markdown(
                """
                <div style='display: flex; justify-content: center; padding: 1rem;'>
                    <div style='width: 200px; height: 200px; background: #404040; border-radius: 12px; 
                                display: flex; align-items: center; justify-content: center; font-size: 48px; color: #fff;'>
                        N
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    with col2:
        st.markdown("<h1 style='font-size: 2.5rem; margin-bottom: 0.5rem; color: #ffffff;'>Nachapol Roc-anusorn</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='font-size: 1.3rem; color: #cccccc; margin-top: 0;'>Process Improvement | Project Management | Operations Team Leadership</h2>", unsafe_allow_html=True)
        
        col_contact1, col_contact2 = st.columns(2)
        
        with col_contact1:
            st.markdown(
                """
                <div style='margin-top: 1rem;'>
                    <p>📍 Bangkok, Thailand</p>
                    <p>📧 <a href="mailto:r.nachapol@gmail.com">r.nachapol@gmail.com</a></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_contact2:
            st.markdown(
                """
                <div style='margin-top: 1rem;'>
                    <p>📞 064-687-7333</p>
                    <p>
                        🔗 <a href="https://www.linkedin.com/in/r-nachapol" target="_blank" rel="noopener noreferrer">LinkedIn</a> | 
                        💻 <a href="https://github.com/pornachapol" target="_blank" rel="noopener noreferrer">GitHub</a>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Resume Download
        resume_url = "https://github.com/pornachapol/Resume/raw/main/assets/Nachapol_Resume_2025.pdf"
        st.markdown(
            f"""
            <a href="{resume_url}" class="download-button" target="_blank" rel="noopener noreferrer">
                📥 Download Resume (PDF)
            </a>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Summary Section
with st.container():
    st.markdown('<div class="content-section" id="summary">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Professional Summary</h2>', unsafe_allow_html=True)
    st.markdown(
        """
        Operations and transformation leader with a proven track record in process optimization, automation, and cross-functional 
        project management. Experienced across insurance, manufacturing, and retail industries. Adept at streamlining operations 
        using Lean methods and analytics tools to improve business performance, service levels, and system efficiency.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Skills Section
with st.container():
    st.markdown('<div class="content-section" id="skills">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Skills & Expertise</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Technical & Automation</h3>", unsafe_allow_html=True)
        
        technical_skills = [
            "UiPath", "Excel VBA / Macro", "Power BI / SQL", "Power Query",
            "ETL Development", "Power Automate", "JavaScript (basic)",
            "Python (basic)", "Jira"
        ]
        
        html_skills = "".join(f'<span class="skill-tag">{skill}</span>' for skill in technical_skills)
        st.markdown(html_skills, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Business & Process</h3>", unsafe_allow_html=True)
        
        business_skills = [
            "Lean Six Sigma", "SOP Standardization", "Project Management", "UAT Coordination",
            "Supply Chain Analysis", "Inventory Management", "BRD Documentation",
            "Process Optimization", "Data Visualization"
        ]
        
        html_skills = "".join(f'<span class="skill-tag">{skill}</span>' for skill in business_skills)
        st.markdown(html_skills, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Leadership & Strategy</h3>", unsafe_allow_html=True)
        
        leadership_skills = [
            "Team Management", "Change Management", "Performance Coaching", "Cross-functional Collaboration",
            "Stakeholder Management", "Problem-Solving", "Communication",
            "Time Management", "Decision-Making"
        ]
        
        html_skills = "".join(f'<span class="skill-tag">{skill}</span>' for skill in leadership_skills)
        st.markdown(html_skills, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Experience Section
with st.container():
    st.markdown('<div class="content-section" id="experience">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Professional Experience</h2>', unsafe_allow_html=True)
    
    # Job 1
    st.markdown('<h3 class="job-title">Claim Registration and Data Service Manager</h3>', unsafe_allow_html=True)
    st.markdown('<p class="job-period">Generali Life Assurance (Thailand) | Dec 2024 – Present | Bangkok</p>', unsafe_allow_html=True)
    st.markdown(
        """
        • Manage operations for <span class="highlight">claim registration and data services</span> across credit and reimbursement claims<br>
        • Coordinate workload allocation, <span class="highlight">SLA monitoring</span>, and team performance management<br>
        • Collaborate with IT and business units to enhance system functionality and integration<br>
        • <span class="highlight">Lead UAT preparation and execution</span> for e-Claim system initiatives<br>
        • Conduct requirement gathering and document business needs into BRDs
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Job 2
    st.markdown('<h3 class="job-title">Business Transformation Manager</h3>', unsafe_allow_html=True)
    st.markdown('<p class="job-period">NGG Enterprise Co., Ltd | Apr 2022 – Nov 2024 | Bangkok</p>', unsafe_allow_html=True)
    st.markdown(
        """
        • Lead <span class="highlight">digital transformation and business improvement projects</span> across departments<br>
        • Design <span class="highlight">dashboards and analytics pipelines</span> using Power BI and ETL tools<br>
        • Oversee end-to-end project delivery including feasibility, planning, and execution<br>
        • Optimize internal processes through collaboration with functional teams
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Job 3
    st.markdown('<h3 class="job-title">Supervisor | Process Improvement Engineer</h3>', unsafe_allow_html=True)
    st.markdown('<p class="job-period">Shinning Gold | Jul 2019 – Apr 2022 | Pathum Thani</p>', unsafe_allow_html=True)
    st.markdown(
        """
        • Supervise production teams and enforce <span class="highlight">standardized operating procedures</span><br>
        • Develop <span class="highlight">automation tools</span> using Excel Macro and JavaScript for planning and reporting<br>
        • Lead Lean-based improvement projects to reduce waste and improve efficiency<br>
        • Align production capacity planning with business forecasts and operational targets
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Job 4
    st.markdown('<h3 class="job-title">Improvement Engineer</h3>', unsafe_allow_html=True)
    st.markdown('<p class="job-period">Siam Kubota Corporation | Jun 2017 – Jun 2019 | Chonburi</p>', unsafe_allow_html=True)
    st.markdown(
        """
        • Implement <span class="highlight">automation solutions</span> including AGV for production line optimization<br>
        • Lead supply chain improvement initiatives including Set Box delivery system<br>
        • Conduct process analysis and layout redesign to support labor efficiency<br>
        • Participate in continuous improvement and quality control circle programs
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Achievements Section
with st.container():
    st.markdown('<div class="content-section" id="achievements">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Key Achievements</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div class="project-card">
                <h4 style="margin-top: 0;">Operational Excellence at Generali</h4>
                <p>Reduced registration backlog by <span class="highlight">70%</span> and improved SLA from <span class="highlight">75% to 95%</span>.
                Successfully led UAT and deployment of e-Claim Data Integration system.</p>
                <p><span class="award-badge">AWARD</span> Exceptional Performance (Innovation), Generali Thailand, 2025</p>
                <p><span class="output-badge">OUTPUT</span> Reduced registration backlog by 70% and improved SLA from 75% to 95%.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col2:
        st.markdown(
            """
            <div class="project-card">
                <h4 style="margin-top: 0;">Jewelry Vending Machine Project at NGG Enterprise</h4>
                <p>Successfully launched jewelry vending machine from feasibility study through 
                deployment, creating a new low-cost retail channel.</p>
                <p><span class="output-badge">OUTPUT</span> Delivered full feasibility report, business model, and ready-to-launch kiosk design for executive consideration.</p>
            </div>

            <div class="project-card">
                <h4 style="margin-top: 0;">End-to-End Sales Dashboard Implementation at NGG Enterprise</h4>
                <p>Designed and implemented a real-time Sales Dashboard for executives using Power BI, AWS Cloud Storage, and Excel. Managed full-cycle data pipeline from cleansing to visualization using storytelling methodology and effective visual design principles.</p>
                <p><span class="output-badge">OUTPUT</span> Reduced daily reporting by 1 hour and enabled real-time sales visibility.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div class="project-card">
                <h4 style="margin-top: 0;">Inventory Optimization at Shinning Gold</h4>
                <p>Reduced inventory redundancy by <span class="highlight">5%</span> (~20kg gold ≈ ฿6M) through 
                comprehensive stock analysis and process redesign.</p>
            </div>
            
            <div class="project-card">
                <h4 style="margin-top: 0;">Manufacturing Performance at Shinning Gold</h4>
                <p>Improved OEE by <span class="highlight">30%</span> and lead time by <span class="highlight">20%</span>; doubled daily output
                through process standardization and workflow optimization.</p>
                <p><span class="award-badge">AWARD</span> Team Efficiency Award, Shinning Gold, 2021</p>
            </div>
            
            <div class="project-card">
                <h4 style="margin-top: 0;">Automation & Cost Reduction at Shinning Gold</h4>
                <p>Reduced manual workload by <span class="highlight">2 FTEs</span>, saving approximately 
                ฿540,000/year through strategic automation of reporting and data processing.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    with col4:
        st.markdown(
            """
            <div class="project-card">
                <h4 style="margin-top: 0;">Process Innovation at Siam Kubota</h4>
                <p>Implemented AGV and Set Box projects for production flow enhancement at Siam Kubota.</p>
                <p><span class="award-badge">AWARD</span> Best QCC Award, Siam Kubota, 2018</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Education & Certifications Section
with st.container():
    st.markdown('<div class="content-section" id="education">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Education & Certifications</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div style="margin-bottom: 1.5rem;">
                <h4 style="margin-bottom: 0.3rem; color: #ffffff;">Master of Science in Management Analytics and Data Technologies</h4>
                <p style="color: #cccccc; font-style: italic; margin-top: 0;">
                    School of Applied Statistics, National Institute of Development Administration (NIDA)<br>
                    Expected Completion: 2025
                </p>
                <p>Focus: Data Analytics, Process Improvement, and Business Strategy</p>
            </div>
            
            <div>
                <h4 style="margin-bottom: 0.3rem; color: #ffffff;">Bachelor of Engineering in Industrial Engineering</h4>
                <p style="color: #cccccc; font-style: italic; margin-top: 0;">
                    Thammasat University<br>
                    2013 – 2017
                </p>
                <p>GPA: 3.15</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style="margin-bottom: 1.5rem;">
                <h4 style="margin-bottom: 0.3rem; color: #ffffff;">Certifications</h4>
                <ul>
                    <li>Lean Six Sigma – Green Belt</li>
                </ul>
            </div>
            
            <div>
                <h4 style="margin-bottom: 0.3rem; color: #ffffff;">Languages</h4>
                <ul>
                    <li><strong>Thai</strong> – Native</li>
                    <li><strong>English</strong> – Strong reading/writing, conversational speaking</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Contact Footer
with st.container():
    st.markdown('<div class="content-section" style="text-align: center; padding: 2rem;">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title" style="border-bottom: none;">Contact Information</h2>', unsafe_allow_html=True)
    
    st.markdown(
        """
        <div style="display: flex; justify-content: center; gap: 2rem; flex-wrap: wrap;">
            <div>
                <p style="margin-bottom: 0.5rem; color: #ffffff !important;"><strong>📍 Location</strong></p>
                <p style="color: #ffffff !important;">Bangkok, Thailand</p>
            </div>
            <div>
                <p style="margin-bottom: 0.5rem; color: #ffffff !important;"><strong>📧 Email</strong></p>
                <p style="color: #ffffff !important;"><a href="mailto:r.nachapol@gmail.com">r.nachapol@gmail.com</a></p>
            </div>
            <div>
                <p style="margin-bottom: 0.5rem; color: #ffffff !important;"><strong>📞 Phone</strong></p>
                <p style="color: #ffffff !important;">064-687-7333</p>
            </div>
            <div>
                <p style="margin-bottom: 0.5rem; color: #ffffff !important;"><strong>🔗 Social</strong></p>
                <p style="color: #ffffff !important;">
                    <a href="https://www.linkedin.com/in/r-nachapol" target="_blank" rel="noopener noreferrer">LinkedIn</a> | 
                    <a href="https://github.com/pornachapol" target="_blank" rel="noopener noreferrer">GitHub</a>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# ========================= Chatbot Section =========================
st.divider()

# Create subheader with forced white color
st.markdown('<h3 style="color: #ffffff !important; margin-bottom: 1rem;">💬 Chat with my Profile</h3>', unsafe_allow_html=True)

# Initialize session state
if "chat" not in st.session_state:
    st.session_state.chat = []

if "backend_url" not in st.session_state:
    st.session_state.backend_url = get_backend_url()

# Avatar configuration
AVATAR_USER = "https://i.imgur.com/1XK7Q9U.png"
AVATAR_BOT = "https://i.imgur.com/3G4cK6X.png"

# Display chat history
import markdown

for role, msg in st.session_state.chat:
    if role == "user":
        st.markdown(
            f"""
            <div class='chat-row user'>
                {avatar_html(AVATAR_USER, 'U')}
                <div class='bubble user-bubble'>{msg}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        # แปลง Markdown → HTML
        html_msg = markdown.markdown(msg)

        st.markdown(
            f"""
            <div class='chat-row bot'>
                {avatar_html(AVATAR_BOT, 'B')}
                <div class='bubble bot-bubble'>{html_msg}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Chat input
user_input = st.chat_input("พิมพ์ข้อความเหมือนคุยใน Messenger เลย...")

if user_input and user_input.strip():
    # Add user message to chat history
    st.session_state.chat.append(("user", user_input))
    
    # Display user message immediately
    st.markdown(
        f"<div class='chat-row user'>{avatar_html(AVATAR_USER,'U')}<div class='bubble user-bubble'>{user_input}</div></div>",
        unsafe_allow_html=True
    )

    # Show loading indicator
    with st.spinner("กำลังตอบกลับ..."):
        # Make request to backend
        response_data = safe_request(
            f"{st.session_state.backend_url}/chat",
            {"message": user_input},
            timeout=60
        )
        
        # Process response
        if "error" in response_data:
            bot_response = f"❌ {response_data['error']}"
        else:
            bot_response = response_data.get("reply", "ขออภัย ระบบไม่สามารถตอบกลับได้ในขณะนี้")
    
    # Add bot response to chat history
    st.session_state.chat.append(("assistant", bot_response))
    
    # Display bot response
    st.markdown(
        f"<div class='chat-row bot'>{avatar_html(AVATAR_BOT,'B')}<div class='bubble bot-bubble'>{bot_response}</div></div>",
        unsafe_allow_html=True
    )
    
    # Rerun to update the display
    st.rerun()

# Add a clear chat button
if st.session_state.get("chat"):
    # ใช้ key และ class เพื่อให้สไตล์เจาะจงปุ่มนี้
    if st.button("🗑️ Clear Chat History", key="clear_chat", help="Clear all chat messages"):
        st.session_state.chat = []
        st.rerun()
    
# Footer with additional info
st.markdown(
    """
    <div style="text-align: center; padding: 2rem; margin-top: 2rem; border-top: 1px solid #404040; color: #ffffff !important;">
        <p style="color: #cccccc !important; font-size: 0.9rem; margin: 0;">
            💡 This chatbot can answer questions about Nachapol's experience, skills, and projects.
        </p>
        <p style="color: #cccccc !important; font-size: 0.9rem; margin: 0.5rem 0 0 0;">
            Feel free to ask about specific achievements, technical expertise, or career background!
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
