import streamlit as st
from PIL import Image

st.set_page_config(page_title="My Resume", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .stApp {
        background-color: #D3D3D3 !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Load assets
img = Image.open("assets/profile_picture.jpeg")  # โหลดรูปโปรไฟล์


# --- Header
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown(
        """
        <div style='display: flex; justify-content: center;'>
            <img src='https://raw.githubusercontent.com/pornachapol/Resume/main/assets/profile_picture.jpeg' 
                 style='
                    border: 2px solid black;
                    box-shadow: 4px 4px 10px rgba(0, 0, 0, 0.3);
                    width: 150px;
                '/>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.title("Nachapol Roc-anusorn")
    st.subheader("Business Analyst | Process Improvement | Project Leader")
    st.markdown(
        """
        📍 Thonburi, Bangkok  
        📧 [r.nachapol@gmail.com](mailto:r.nachapol@gmail.com) | 📞 064-687-7333  
        🔗 [LinkedIn](https://www.linkedin.com/in/r-nachapol) | 💻 [GitHub](https://github.com/pornachapol)
        """
    )

# --- Resume Download Section
st.write("---")
st.subheader("📄 Resume for Download")

resume_url = "https://github.com/pornachapol/Resume/raw/main/assets/Nachapol_Resume_2025.pdf"  # ลิงก์ GitHub ที่เก็บ PDF
st.markdown(f"[📥 Click here to download my resume (PDF)]({resume_url})", unsafe_allow_html=True)

# --- Summary
st.write("---")
st.subheader("Summary")
st.write("""
Results-driven professional with strong experience in process optimization, automation, and cross-functional team leadership. 
Skilled in delivering operational improvements through Lean methodology, RPA (UiPath), and data analytics tools such as Power BI and Excel VBA. 
Proven ability to lead end-to-end projects across manufacturing, retail, and insurance industries, driving measurable results in productivity, SLA compliance, and stock efficiency.
""")


# --- Skills
st.write("---")
st.subheader("Skills")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Hard Skills**")
    st.markdown("- Process Optimization")
    st.markdown("- Automation (UiPath / VBA)")
    st.markdown("- Power BI / SQL / Python (Basic)")
    st.markdown("- Data Visualization")
    st.markdown("- Project & Business Analysis")
    st.markdown("- Supply Chain Management")
    st.markdown("- Jira, Power Automate")

with col2:
    st.markdown("**Soft Skills**")
    st.markdown("- Team Leadership")
    st.markdown("- Problem-Solving")
    st.markdown("- Communication & Collaboration")
    st.markdown("- Decision-Making")
    st.markdown("- Adaptability")
    st.markdown("- Time & Change Management")

# --- Experience
st.write("---")
st.subheader("Experience")

st.markdown("### Claim Registration Manager – Generali Life Assurance (Thailand)")
st.markdown("_Dec 2024 – Present | Bangkok_")
st.markdown("""
- Reduced claim registration backlog from 20,000+ to 6,000 transactions within 2 months.  
- Improved SLA compliance from 75% to 95% through workflow optimization.  
- Led UAT and business analysis for E-Claim integration system.  
- Streamlined internal processes using Lean, enhancing team efficiency.
""")

st.markdown("### Transformation & Project Management Manager – NGG Enterprise")
st.markdown("_Apr 2022 – Dec 2024 | Bangkok_")
st.markdown("""
- Managed cross-functional projects such as E-Cert System, Vending Machine, and Price Optimization.  
- Built dashboards for Sales and Supply Chain using Excel, Power BI, and SQL.  
- Automated reporting workflows using VBA and UiPath, reducing manual effort.  
- Designed commission scheme and clarified financial state using EBITDA analysis.
""")

st.markdown("### Supervisor & Process Improvement – Shinning Gold")
st.markdown("_Jul 2019 – Apr 2022 | Pathum Thani_")
st.markdown("""
- Increased productivity in wiring team by 25% and doubled overall output.  
- Reduced lead time by 20% and improved OEE by 30% through standardization.  
- Implemented automation with Excel Macro and JavaScript for planning processes.  
- Consolidated inventory to reduce duplicate stock by 5% (~20kg of gold).
""")

st.markdown("### Improvement Engineer – Siam Kubota Corporation")
st.markdown("_Jun 2017 – Jul 2019 | Chonburi_")
st.markdown("""
- Implemented AGV and Set Box projects to improve production flow.  
- Enhanced operational efficiency and reduced manpower through process redesign.
""")


# --- Projects
st.write("---")
st.subheader("Projects")

st.markdown("**E-Certification System for Jewelry**")
st.markdown("""
Led the end-to-end development of an electronic certification system for jewelry products, improving operational efficiency and document traceability.
""")

st.markdown("**Jewelry Vending Machine Project**")
st.markdown("""
Managed feasibility, design, and implementation of an automated vending machine project to expand retail channels with lower operational cost.
""")

st.markdown("**Retail Price Optimization & Tag Process Improvement**")
st.markdown("""
Analyzed market price gaps and optimized the retail pricing and tag-changing process, resulting in improved competitiveness and agility.
""")

st.markdown("**Sales & Supply Chain Dashboard Development**")
st.markdown("""
Built dashboards using Excel, Power BI, and SQL to visualize sales and inventory performance, streamlining decision-making for business units.
""")

st.markdown("**Automation for Report Generation**")
st.markdown("""
Created automation tools using Excel Macro and JavaScript for data extraction and dashboard updates, reducing report time from days to hours.
""")

st.markdown("**Stock Optimization Project**")
st.markdown("""
Used data-driven decision-making to reduce 5% of duplicate stock (~20 kg gold), saving over 6 million baht in material costs.
""")

st.markdown("**Wiring Productivity Improvement**")
st.markdown("""
Standardized processes and motivated team members, doubling productivity and reducing lead time by 20%, while increasing OEE by 30%.
""")


# --- Education
st.write("---")
st.subheader("Education")

st.markdown("**Master of Science in Management Analytics and Data Technologies (MADT)**")
st.markdown("_National Institute of Development Administration (NIDA), 2024 – 2025 (Expected)_")
st.markdown("Focus: Data Analytics, Process Improvement, and Business Strategy")

st.markdown("**Bachelor of Engineering in Industrial Engineering**")
st.markdown("_Thammasat University, 2013 – 2017_")
st.markdown("GPA: 3.15")


# --- Contact
st.write("---")
st.subheader("Contact")

st.markdown("- 📍 Thonburi, Bangkok")
st.markdown("- 📧 r.nachapol@gmail.com")
st.markdown("- 📞 064-687-7333")
st.markdown("- 🔗 [LinkedIn](https://www.linkedin.com/in/r-nachapol)")
st.markdown("- 💻 [GitHub](https://github.com/pornachapol)")
