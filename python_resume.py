import streamlit as st
from PIL import Image

st.set_page_config(page_title="My Resume", page_icon="📄", layout="wide")

# --- Load assets
img = Image.open("assets/profile_picture.jpeg")  # โหลดรูปโปรไฟล์

# --- Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image(img, width=150)
with col2:
    st.title("Nachapol Roc-anusorn")
    st.subheader("Business Analyst | Data Strategy | Project Manager")
    st.markdown("📍 Bangkok, Thailand | 📧 email@example.com | 🔗 [LinkedIn](https://...)")

# --- Resume Download Section
st.write("---")
st.subheader("📄 Resume for Download")

resume_url = "https://github.com/pornachapol/Resume/raw/main/assets/Nachapol_Resume_2025.pdf"  # ลิงก์ GitHub ที่เก็บ PDF
st.markdown(f"[📥 Click here to download my resume (PDF)]({resume_url})", unsafe_allow_html=True)

# --- Summary
st.write("---")
st.subheader("Summary")
st.write("Experienced Business/Data Analyst with 5+ years in process improvement and data-driven decision-making. Passionate about turning data into insights and strategies.")

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
st.markdown("**Claim Registration Manager – Generali Insurance** (2023 – Present)")
st.markdown("- Improved SLA by 30% using automation and dashboarding.")
st.markdown("- Led e-claim data integration, reducing backlog by 40%.")

# --- Projects
st.write("---")
st.subheader("Projects")
st.markdown("- [AirSniff IoT Fire Monitoring](https://github.com/...)")
st.markdown("- [Jewelry Vending Machine ROI Analysis](https://github.com/...)")

# --- Education
st.write("---")
st.subheader("Education")
st.markdown("**M.S. Applied Statistics (MADT)** – NIDA (2025)")

# --- Contact
st.write("---")
st.subheader("Contact")
st.markdown("- 📧 email@example.com")
st.markdown("- 🔗 [GitHub](https://github.com/...)")
