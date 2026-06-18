import streamlit as st
import os
import sys
import re
import socket
import threading

# Add parent directory to sys.path to ensure absolute imports work
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Configure Streamlit page layout
st.set_page_config(
    page_title="SocialSenseAI - Analisis Tingkat Kecanduan Media Sosial",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Helper function to check if a port is in use
def is_port_in_use(port):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex(('127.0.0.1', port)) == 0
    except Exception:
        return False

# Start FastAPI server in a background thread if it is not already running
def start_api_server():
    if not is_port_in_use(8000):
        try:
            import uvicorn
            from app.api import app
            api_thread = threading.Thread(
                target=lambda: uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning"),
                daemon=True
            )
            api_thread.start()
        except Exception as e:
            # Fallback in case of any start issues
            pass

# Initialize the API server
if "api_started" not in st.session_state:
    start_api_server()
    st.session_state["api_started"] = True

# Helper to read and inline HTML, CSS, and JS
def get_custom_html(api_url):
    frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
    
    # Read files with UTF-8 encoding
    with open(os.path.join(frontend_dir, "index.html"), "r", encoding="utf-8") as f:
        html = f.read()
        
    with open(os.path.join(frontend_dir, "style.css"), "r", encoding="utf-8") as f:
        css = f.read()
        
    with open(os.path.join(frontend_dir, "app.js"), "r", encoding="utf-8") as f:
        js = f.read()

    # Inline the CSS
    css_tag = f"<style>\n{css}\n</style>"
    html = html.replace('<link rel="stylesheet" href="style.css">', css_tag)
    
    # Replace API_URL dynamically based on configuration
    js_override = f'const API_URL = "{api_url}";'
    js_modified = re.sub(r'const\s+API_URL\s*=\s*["\'][^"\']*["\'];?', js_override, js)
    
    # Inline the modified JS
    js_tag = f"<script>\n{js_modified}\n</script>"
    html = html.replace('<script src="app.js"></script>', js_tag)
    
    return html

# Get the API URL from Streamlit query params, environment variable, or fallback to localhost
api_url = "http://127.0.0.1:8000"
if hasattr(st, "query_params"):
    api_url = st.query_params.get("api_url", os.getenv("API_URL", "http://127.0.0.1:8000"))
elif hasattr(st, "experimental_get_query_params"):
    params = st.experimental_get_query_params()
    api_url = params.get("api_url", [os.getenv("API_URL", "http://127.0.0.1:8000")])[0]

# Inject custom CSS to make Streamlit's iframe occupy the entire viewport
st.markdown("""
    <style>
    /* Hide Streamlit header, footer, and margins */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .block-container {
        padding: 0rem !important;
        max-width: 100% !important;
    }
    iframe {
        width: 100vw !important;
        height: 100vh !important;
        border: none !important;
        display: block !important;
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        z-index: 999999 !important;
    }
    body {
        overflow: hidden !important;
    }
    </style>
""", unsafe_allow_html=True)

# Render the self-contained frontend in an iframe with scrolling
html_content = get_custom_html(api_url)
st.iframe(html_content, height=800)
