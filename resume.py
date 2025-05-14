import streamlit as st
from PIL import Image

st.set_page_config(page_title="My Resume", page_icon="📄", layout="wide")

# --- Load assets
img = Image.open("/mount/src/resume/Profile_Picture.jpeg")

# --- Header
col1, col2 = st.columns([1, 3])
with col1:
    st.image(img, width=150)
with col2:
    st.title("Nachapol Roc-anusorn")
    st.subheader("Business Analyst | Data Strategy | Project Manager")
    st.markdown("📍 Bangkok, Thailand | 📧 email@example.com | 🔗 [LinkedIn](https://...)")

# --- Summary
st.write("---")
st.subheader("Summary")
st.write("Experienced Business/Data Analyst with 5+ years in process improvement and data-driven decision-making. Passionate about turning data into insights and strategies.")

# --- Skills
st.write("---")
st.subheader("Skills")
col1, col2 = st.columns(2)
with col1:
    st.markdown("- Python / SQL / Power BI")
    st.markdown("- Excel VBA / RPA UiPath")
with col2:
    st.markdown("- Leadership & Collaboration")
    st.markdown("- Data Strategy & Governance")

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

