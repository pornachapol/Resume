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
        background-color: #202020;
        color: #ffffff;
    }
    
    /* Main sections styling */
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
        max-width: 100%;
        height: auto;
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
        color: #fff;
    }
    
    .user-bubble {
        background: linear-gradient(180deg,#0057ff,#1a73e8);
        color: #fff;
        border-bottom-right-radius: 6px;
    }
    
    .bot-bubble {
        background: #E4E6EB;
        color: #111;
        border-bottom-left-radius: 6px;
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
    """Get backend URL from environment or secrets; return None if not configured"""
    try:
        # Try environment variable first
        env_url = (os.getenv("BACKEND_URL") or "").strip()
        if env_url:
            return env_url
        
        # Try Streamlit secrets
        if hasattr(st, 'secrets') and "BACKEND_URL" in st.secrets:
            secret_url = str(st.secrets["BACKEND_URL"]).strip()
            if secret_url:
                return secret_url
        
        # Not configured
        return None
    except Exception:
        return None

def is_backend_configured(url):
    """Heuristic to determine if a usable backend URL is configured"""
    if not url:
        return False
    if "your-backend-url.com" in url:
        return False
    return True

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
        st.markdown(
            """
            <h1 style="font-size: 2.5rem; margin-bottom: 0.5
