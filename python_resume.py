import streamlit as st

# Page config
st.set_page_config(
    page_title="Nachapol Roc-anusorn | Resume",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- CSS Styling ----------
st.markdown("""
<style>
/* Sticky Navigation */
.fixed-top-menu {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: #202020;
    z-index: 999;
    min-height: 60px;
    padding: 0.8rem 1rem;
    border-bottom: 1px solid #404040;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Push content below navbar */
.stApp {
    padding-top: 100px !important;
    background-color: #202020;
    color: #ffffff;
}

/* Section styling */
.main-header, .content-section {
    background-color: #303030;
    padding: 2rem;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    margin-bottom: 2rem;
}

.section-title {
    color: #ffffff;
    font-size: 1.8rem;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 2px solid #505050;
    padding-bottom: 0.5rem;
}

.job-title {
    font-size: 1.2rem;
    font-weight: 600;
    color: #ffffff;
}

.job-period {
    font-style: italic;
    color: #cccccc;
    margin-bottom: 0.8rem;
}

.download-button {
    background-color: #ffffff;
    color: #202020 !important;
    padding: 0.6rem 1.2rem;
    border-radius: 6px;
    font-weight: 500;
    display: inline-block;
    margin-top: 1rem;
    text-decoration: none;
}

.download-button:hover {
    background-color: #dddddd;
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
    margin: 4px 8px 8px 0;
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

a {
    color: #4cb5f9 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Navigation Bar (Fix visibility with full HTML render) ---
st.components.v1.html("""
<div style="
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    background-color: #202020;
    z-index: 999;
    min-height: 60px;
    padding: 0.8rem 1rem;
    border-bottom: 1px solid #404040;
    display: flex;
    justify-content: space-between;
    align-items: center;
">
    <div style="color: #ffffff; font-size: 1.2rem; font-weight: 600;">
        Nachapol Resume
    </div>
    <div style="color: #ffffff;">
        <a href="#summary" style="margin-right: 1.5rem; color: #ffffff; text-decoration: none;">Summary</a>
        <a href="#skills" style="margin-right: 1.5rem; color: #ffffff; text-decoration: none;">Skills</a>
        <a href="#experience" style="margin-right: 1.5rem; color: #ffffff; text-decoration: none;">Experience</a>
        <a href="#projects" style="margin-right: 1.5rem; color: #ffffff; text-decoration: none;">Projects</a>
        <a href="#education" style="color: #ffffff; text-decoration: none;">Education</a>
    </div>
</div>
""", height=80)

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
        st.markdown("<h2 style='font-size: 1.3rem; color: #cccccc; margin-top: 0;'>Business Analyst | Process Improvement | Project Leader</h2>", unsafe_allow_html=True)
        
        col_contact1, col_contact2 = st.columns(2)
        
        with col_contact1:
            st.markdown(
                """
                <div style='margin-top: 1rem;'>
                    <p><i class="fas fa-map-marker-alt"></i> 📍 Thonburi, Bangkok</p>
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
        Results-driven professional with strong experience in process optimization, automation, and cross-functional team leadership. 
        Skilled in delivering operational improvements through Lean methodology, RPA (UiPath), and data analytics tools such as Power BI and Excel VBA. 
        Proven ability to lead end-to-end projects across manufacturing, retail, and insurance industries, driving measurable results in productivity, 
        SLA compliance, and stock efficiency.
        """
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Skills Section
with st.container():
    st.markdown('<div class="content-section" id="skills">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Skills & Expertise</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Technical Skills</h3>", unsafe_allow_html=True)
        
        technical_skills = [
            "Process Optimization", 
            "Automation (UiPath / VBA)", 
            "Power BI / SQL", 
            "Python (Basic)",
            "Data Visualization", 
            "Business Analysis", 
            "Supply Chain Management",
            "Jira",
            "Power Automate",
            "Lean Methodology"
        ]
        
        # Display skills as tags
        html_skills = ""
        for skill in technical_skills:
            html_skills += f'<span class="skill-tag">{skill}</span>'
        
        st.markdown(html_skills, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h3 style='font-size: 1.2rem; margin-bottom: 1rem;'>Soft Skills</h3>", unsafe_allow_html=True)
        
        soft_skills = [
            "Team Leadership", 
            "Problem-Solving", 
            "Communication", 
            "Collaboration",
            "Decision-Making", 
            "Adaptability", 
            "Time Management",
            "Change Management",
            "Presentation Skills",
            "Stakeholder Management"
        ]
        
        # Display skills as tags
        html_skills = ""
        for skill in soft_skills:
            html_skills += f'<span class="skill-tag">{skill}</span>'
        
        st.markdown(html_skills, unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Experience Section
with st.container():
    st.markdown('<div class="content-section" id="experience">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Professional Experience</h2>', unsafe_allow_html=True)
    
    # Job 1
    st.markdown('<h3 class="job-title">Claim Registration Manager</h3>', unsafe_allow_html=True)
    st.markdown('<p class="job-period">Generali Life Assurance (Thailand) | Dec 2024 – Present | Bangkok</p>', unsafe_allow_html=True)
    st.markdown(
        """
        • <span class="highlight">Reduced claim registration backlog</span> from 20,000+ to 6,000 transactions within 2 months
        • <span class="highlight">Improved SLA compliance</span> from 75% to 95% through workflow optimization
        • Led UAT and business analysis for E-Claim integration system
        • Streamlined internal processes using Lean methodology, enhancing team efficiency
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Job 2
    st.markdown('<h3 class="job-title">Transformation & Project Management Manager</h3>', unsafe_allow_html=True)
    st.markdown('<p class="job-period">NGG Enterprise | Apr 2022 – Dec 2024 | Bangkok</p>', unsafe_allow_html=True)
    st.markdown(
        """
        • Managed cross-functional projects including E-Cert System, Vending Machine, and Price Optimization
        • Built <span class="highlight">data-driven dashboards</span> for Sales and Supply Chain using Excel, Power BI, and SQL
        • <span class="highlight">Automated reporting workflows</span> using VBA and UiPath, significantly reducing manual effort
        • Designed commission scheme and performed EBITDA analysis to clarify financial state
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Job 3
    st.markdown('<h3 class="job-title">Supervisor & Process Improvement</h3>', unsafe_allow_html=True)
    st.markdown('<p class="job-period">Shinning Gold | Jul 2019 – Apr 2022 | Pathum Thani</p>', unsafe_allow_html=True)
    st.markdown(
        """
        • <span class="highlight">Increased productivity</span> in wiring team by 25% and doubled overall output
        • Reduced lead time by 20% and improved OEE by 30% through standardization
        • Implemented automation with Excel Macro and JavaScript for planning processes
        • Consolidated inventory to <span class="highlight">reduce duplicate stock</span> by 5% (~20kg of gold)
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("<hr style='margin: 1.5rem 0; opacity: 0.3;'>", unsafe_allow_html=True)
    
    # Job 4
    st.markdown('<h3 class="job-title">Improvement Engineer</h3>', unsafe_allow_html=True)
    st.markdown('<p class="job-period">Siam Kubota Corporation | Jun 2017 – Jul 2019 | Chonburi</p>', unsafe_allow_html=True)
    st.markdown(
        """
        • Implemented AGV and Set Box projects to improve production flow
        • Enhanced operational efficiency and reduced manpower through process redesign
        """,
        unsafe_allow_html=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Projects Section
with st.container():
    st.markdown('<div class="content-section" id="projects">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Key Projects</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div class="project-card">
                <h4 style="margin-top: 0;">E-Certification System for Jewelry</h4>
                <p>Led the end-to-end development of an electronic certification system for jewelry products, 
                improving operational efficiency and document traceability.</p>
            </div>
            
            <div class="project-card">
                <h4 style="margin-top: 0;">Jewelry Vending Machine Project</h4>
                <p>Managed feasibility, design, and implementation of an automated vending machine project 
                to expand retail channels with lower operational cost.</p>
            </div>
            
            <div class="project-card">
                <h4 style="margin-top: 0;">Retail Price Optimization</h4>
                <p>Analyzed market price gaps and optimized the retail pricing and tag-changing process, 
                resulting in improved competitiveness and agility.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div class="project-card">
                <h4 style="margin-top: 0;">Sales & Supply Chain Dashboard Development</h4>
                <p>Built dashboards using Excel, Power BI, and SQL to visualize sales and inventory performance, 
                streamlining decision-making for business units.</p>
            </div>
            
            <div class="project-card">
                <h4 style="margin-top: 0;">Automation for Report Generation</h4>
                <p>Created automation tools using Excel Macro and JavaScript for data extraction and dashboard updates, 
                reducing report time from days to hours.</p>
            </div>
            
            <div class="project-card">
                <h4 style="margin-top: 0;">Stock Optimization Project</h4>
                <p>Used data-driven decision-making to reduce 5% of duplicate stock (~20 kg gold), 
                saving over 6 million baht in material costs.</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Education Section
with st.container():
    st.markdown('<div class="content-section" id="education">', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Education</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div style="margin-bottom: 1.5rem;">
                <h4 style="margin-bottom: 0.3rem; color: #ffffff;">Master of Science in Management Analytics and Data Technologies (MADT)</h4>
                <p style="color: #cccccc; font-style: italic; margin-top: 0;">
                    National Institute of Development Administration (NIDA)<br>
                    2024 – 2025 (Expected)
                </p>
                <p>Focus: Data Analytics, Process Improvement, and Business Strategy</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
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
                <p>Thonburi, Bangkok</p>
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

# Footer
st.markdown(
    """
    <footer style="text-align: center; padding: 1rem; color: #cccccc; font-size: 0.8rem; margin-top: 2rem; border-top: 1px solid #404040;">
        &copy; 2025 Nachapol Roc-anusorn | Last Updated: May 2025
    </footer>
    """,
    unsafe_allow_html=True
)
