import os
import requests
import streamlit as st
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Nachapol Roc-anusorn | Resume",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling
st.markdown(
    """
    <style>
    .stApp {
        background-color: #202020;
        color: #ffffff;
    }
    
    .main-header {
        padding: 2rem 0;
        background-color: #303030;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        margin-bottom: 2rem;
    }
    
    .content-section {
        background-color: #303030;
        padding: 1.5rem 2rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.15);
        margin-bottom: 1.5rem;
        color: #ffffff;
    }
    
    .section-title {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 600;
        margin-bottom: 1.2rem;
        padding-bottom: 0.6rem;
        border-bottom: 2px solid #505050;
    }
    
    .job-title {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 600;
        margin-bottom: 0.2rem;
    }
    
    .job-period {
        color: #cccccc;
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
        border: 3px solid #404040;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        border-radius: 12px;
        width: 200px;
    }
    
    .skill-tag {
        background-color: #404040;
        color: #ffffff;
        border-radius: 20px;
        padding: 5px 12px;
        margin-right: 8px;
        margin-bottom: 8px;
        display: inline-block;
        font-size: 0.85rem;
    }
    
    .project-card {
        background-color: #404040;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 1rem;
        border-left: 4px solid #4cb5f9;
    }
    
    .highlight {
        color: #4cb5f9;
        font-weight: 500;
    }
    
    .award-badge {
        background-color: #4cb5f9;
        color: #202020;
        border-radius: 4px;
        padding: 3px 8px;
        margin-right: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .output-badge {
        background-color: #78d88f;
        color: #202020;
        border-radius: 4px;
        padding: 3px 8px;
        margin-right: 8px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Fix for Streamlit's default text color */
    p, li, div {
        color: #ffffff;
    }
    
    /* Fix for links color */
    a {
        color: #4cb5f9 !important;
        text-decoration: none;
    }
    
    a:hover {
        text-decoration: underline;
    }
    
    /* Fix for Streamlit's components */
    .stMarkdown, .stText {
        color: #ffffff;
    }
    
    /* Remove default Streamlit background color from elements */
    div[data-testid="stVerticalBlock"] {
        background-color: transparent;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# Messenger-like CSS (NEW)
st.markdown("""
<style>
/* ===== Messenger-like Chat ===== */
.chat-wrap{
  max-width: 900px; margin: 0 auto; padding: 8px 8px 0 8px;
}
.msg-row{ display:flex; margin: 10px 0; align-items:flex-end; }
.msg-row.user{ justify-content:flex-end; }
.msg-row.bot{ justify-content:flex-start; }
.avatar{
  width: 36px; height: 36px; border-radius: 50%; overflow:hidden; flex-shrink:0;
  box-shadow: 0 2px 6px rgba(0,0,0,.2);
}
.avatar img{ width:100%; height:100%; object-fit:cover; }
.bubble{
  max-width: 70%;
  padding: 10px 14px; border-radius: 18px; word-wrap: break-word; line-height: 1.45;
  box-shadow: 0 1px 4px rgba(0,0,0,.2);
}
.msg-row.user .bubble{
  background: #0084ff; color:#fff; border-bottom-right-radius: 6px; margin-left: 10px;
}
.msg-row.bot .bubble{
  background: #f1f0f0; color:#111; border-bottom-left-radius: 6px; margin-right: 10px;
}
.meta{
  font-size: 12px; opacity:.7; margin-top: 4px;
  display: flex; gap: 8px; justify-content: flex-end;
}
.msg-row.bot .meta{ justify-content:flex-start; }
.src-chip{
  display:inline-block; font-size:11px; padding:2px 6px; border-radius:12px;
  background:#e8eefc; color:#345; margin-right:6px; border:1px solid #d8e2ff;
}
</style>
""", unsafe_allow_html=True)

# Add a navigation menu
menu = st.container()
with menu:
    st.markdown(
        """
        <div style="background-color: #202020; padding: 0.8rem 1rem; border-bottom: 1px solid #404040; margin-bottom: 2rem; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">Nachapol Resume</span>
            </div>
            <div>
                <a href="#summary" style="color: #ffffff; margin-right: 1.5rem; text-decoration: none;">Summary</a>
                <a href="#skills" style="color: #ffffff; margin-right: 1.5rem; text-decoration: none;">Skills</a>
                <a href="#experience" style="color: #ffffff; margin-right: 1.5rem; text-decoration: none;">Experience</a>
                <a href="#achievements" style="color: #ffffff; margin-right: 1.5rem; text-decoration: none;">Achievements</a>
                <a href="#education" style="color: #ffffff; text-decoration: none;">Education</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# Header Section
with st.container():
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown(
            """
            <div style='display: flex; justify-content: center; padding: 1rem;'>
                <img src='https://raw.githubusercontent.com/pornachapol/Resume/main/assets/profile_picture.jpeg' 
                     class='profile-image'/>
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
                    <p><i class="fas fa-map-marker-alt"></i> 📍 Bangkok, Thailand</p>
                    <p><i class="fas fa-envelope"></i> 📧 <a href="mailto:r.nachapol@gmail.com">r.nachapol@gmail.com</a></p>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        with col_contact2:
            st.markdown(
                """
                <div style='margin-top: 1rem;'>
                    <p><i class="fas fa-phone"></i> 📞 064-687-7333</p>
                    <p>
                        <i class="fab fa-linkedin"></i> 🔗 <a href="https://www.linkedin.com/in/r-nachapol" target="_blank">LinkedIn</a> | 
                        <i class="fab fa-github"></i> 💻 <a href="https://github.com/pornachapol" target="_blank">GitHub</a>
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Resume Download
        resume_url = "https://github.com/pornachapol/Resume/raw/main/assets/Nachapol_Resume_2025.pdf"
        st.markdown(
            f"""
            <a href="{resume_url}" class="download-button" target="_blank">
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
            "UiPath", 
            "Excel VBA / Macro", 
            "Power BI / SQL", 
            "Power Query",
            "ETL Development", 
            "Power Automate", 
            "JavaScript (basic)",
            "Python (basic)",
            "Jira"
        ]
        
        # Display skills as tags
        html_skills = ""
        for skill in technical_skills:
            html_skills += f'<span class="skill-tag">{skill}</span>'
        
        st.markdown(html_skills, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Business & Process</h3>", unsafe_allow_html=True)
        
        business_skills = [
            "Lean Six Sigma", 
            "SOP Standardization", 
            "Project Management", 
            "UAT Coordination",
            "Supply Chain Analysis", 
            "Inventory Management", 
            "BRD Documentation",
            "Process Optimization",
            "Data Visualization"
        ]
        
        # Display skills as tags
        html_skills = ""
        for skill in business_skills:
            html_skills += f'<span class="skill-tag">{skill}</span>'
        
        st.markdown(html_skills, unsafe_allow_html=True)
    
    with col3:
        st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Leadership & Strategy</h3>", unsafe_allow_html=True)
        
        leadership_skills = [
            "Team Management", 
            "Change Management", 
            "Performance Coaching", 
            "Cross-functional Collaboration",
            "Stakeholder Management", 
            "Problem-Solving", 
            "Communication",
            "Time Management",
            "Decision-Making"
        ]
        
        # Display skills as tags
        html_skills = ""
        for skill in leadership_skills:
            html_skills += f'<span class="skill-tag">{skill}</span>'
        
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
        • Manage operations for <span class="highlight">claim registration and data services</span> across credit and reimbursement claims\n
        • Coordinate workload allocation, <span class="highlight">SLA monitoring</span>, and team performance management\n
        • Collaborate with IT and business units to enhance system functionality and integration\n
        • <span class="highlight">Lead UAT preparation and execution</span> for e-Claim system initiatives\n
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
        • Lead <span class="highlight">digital transformation and business improvement projects</span> across departments\n
        • Design <span class="highlight">dashboards and analytics pipelines</span> using Power BI and ETL tools\n
        • Oversee end-to-end project delivery including feasibility, planning, and execution\n
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
        • Supervise production teams and enforce <span class="highlight">standardized operating procedures</span>\n
        • Develop <span class="highlight">automation tools</span> using Excel Macro and JavaScript for planning and reporting\n
        • Lead Lean-based improvement projects to reduce waste and improve efficiency\n
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
        • Implement <span class="highlight">automation solutions</span> including AGV for production line optimization\n
        • Lead supply chain improvement initiatives including Set Box delivery system\n
        • Conduct process analysis and layout redesign to support labor efficiency\n
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
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
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
                <p style="margin-bottom: 0.5rem;"><strong>📍 Location</strong></p>
                <p>Bangkok, Thailand</p>
            </div>
            <div>
                <p style="margin-bottom: 0.5rem;"><strong>📧 Email</strong></p>
                <p><a href="mailto:r.nachapol@gmail.com">r.nachapol@gmail.com</a></p>
            </div>
            <div>
                <p style="margin-bottom: 0.5rem;"><strong>📞 Phone</strong></p>
                <p>064-687-7333</p>
            </div>
            <div>
                <p style="margin-bottom: 0.5rem;"><strong>🔗 Social</strong></p>
                <p>
                    <a href="https://www.linkedin.com/in/r-nachapol" target="_blank">LinkedIn</a> | 
                    <a href="https://github.com/pornachapol" target="_blank">GitHub</a>
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Chatbot Section
    import requests

    from datetime import datetime

    USER_AVATAR = "https://i.imgur.com/1XK7Q9U.png"  # เปลี่ยนเป็นรูปคุณก็ได้
    BOT_AVATAR  = "https://i.imgur.com/8Km9tLL.png"  # หรือไอคอนบอท
    
    def render_msg(role: str, text: str, sources=None, ts=None):
        """
        role: "user" | "assistant"
        text: เนื้อหาข้อความ
        sources: list[str] (ออปชัน) เพื่อแสดงแหล่งอ้างอิงเป็นชิปเล็ก ๆ
        ts: timestamp string (ออปชัน)
        """
        cls = "user" if role=="user" else "bot"
        ava = USER_AVATAR if role=="user" else BOT_AVATAR
        if ts is None:
            ts = datetime.now().strftime("%H:%M")
        src_html = ""
        if sources:
            chips = "".join([f'<span class="src-chip">{s}</span>' for s in sources][:5])
            src_html = f'<div style="margin-top:6px;">{chips}</div>'
        st.markdown(f"""
        <div class="chat-wrap">
          <div class="msg-row {cls}">
            {'<div class="avatar"><img src="'+ava+'" /></div>' if cls=='bot' else ''}
            <div class="bubble">
              {text}
              {src_html}
              <div class="meta">{ts}</div>
            </div>
            {'<div class="avatar"><img src="'+ava+'" /></div>' if cls=='user' else ''}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.subheader("💬 Chat with my Resume")
    
    BACKEND_URL = os.getenv("BACKEND_URL") or st.secrets.get("BACKEND_URL")
    
    if "chat" not in st.session_state:
        # เก็บ (role, text, ts, sources)
        st.session_state.chat = []
    
    # แสดงประวัติ
    for role, msg, ts, srcs in st.session_state.chat:
        render_msg(role, msg, sources=srcs, ts=ts)
    
    q = st.chat_input("พิมพ์ข้อความเหมือนคุยใน Messenger เลย…")
    if q:
        # ผู้ใช้ส่งข้อความ
        now = datetime.now().strftime("%H:%M")
        st.session_state.chat.append(("user", q, now, None))
        render_msg("user", q, ts=now)
    
        # เรียก backend
        ans_text = "❌ เชื่อมต่อ Backend ไม่ได้"
        sources = None
        try:
            r = requests.post(f"{BACKEND_URL}/chat", json={"message": q}, timeout=60)
            r.raise_for_status()
            data = r.json()
            ans_text = data.get("reply", "ขออภัย ระบบไม่ตอบกลับ")
            # แปลง sources จาก backend [(idx, preview), ...] → ["#0 ...", ...]
            raw_src = data.get("sources") or []
            sources = [f"#{i} {pre}" for i, pre in raw_src][:4]
        except Exception as e:
            ans_text = f"❌ เชื่อมต่อ Backend ไม่ได้\n\n`{e}`"
    
        now2 = datetime.now().strftime("%H:%M")
        st.session_state.chat.append(("assistant", ans_text, now2, sources))
        render_msg("assistant", ans_text, sources=sources, ts=now2)

# Footer
st.markdown(
    """
    <footer style="text-align: center; padding: 1rem; color: #cccccc; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #404040;">
        &copy; 2025 Nachapol Roc-anusorn | Last Updated: May 2025
    </footer>
    """,
    unsafe_allow_html=True
)
