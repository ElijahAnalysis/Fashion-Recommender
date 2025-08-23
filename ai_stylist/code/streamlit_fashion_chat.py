import streamlit as st
import requests
import json
from PIL import Image
import os
from io import BytesIO
import base64

# Configure the page
st.set_page_config(
    page_title="CLIP + Gemma Fashion Engine",
    page_icon="■",
    layout="wide"
)

# Minimalist CSS
st.markdown("""
<style>
    .main-header {
        font-family: 'SF Mono', 'Monaco', 'Inconsolata', 'Roboto Mono', monospace;
        font-weight: 300;
        color: #1a1a1a;
        text-align: center;
        margin-bottom: 40px;
        font-size: 28px;
    }
    
    .user-message {
        background-color: #000;
        color: white;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
        text-align: right;
        max-width: 60%;
        margin-left: auto;
        font-family: monospace;
        font-size: 14px;
    }
    
    .bot-message {
        background-color: #f8f9fa;
        color: #2c3e50;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 8px 0;
        max-width: 60%;
        border-left: 3px solid #000;
        font-family: monospace;
        font-size: 14px;
    }
    
    .fashion-items {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        padding: 20px;
        margin: 16px 0;
        font-family: monospace;
    }
    
    .tech-info {
        font-family: monospace;
        font-size: 12px;
        color: #666;
        text-align: center;
        margin: 40px 0;
    }
    
    .stApp > header {
        background-color: transparent;
    }
    
    .sidebar-header {
        font-family: monospace;
        font-weight: 600;
        color: #2c3e50;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:4321"

def call_fashion_api(text, api_url):
    """Call the fashion API with the user's text"""
    try:
        response = requests.post(
            f"{api_url}/make_look",
            json={"text": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status code {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to connect to API: {str(e)}"}

def display_image_safely(image_path):
    """Display image with error handling"""
    try:
        if os.path.exists(image_path):
            image = Image.open(image_path)
            return image
        else:
            placeholder = Image.new('RGB', (200, 200), color='#f0f0f0')
            return placeholder
    except Exception as e:
        placeholder = Image.new('RGB', (200, 200), color='#e0e0e0')
        return placeholder

def display_chat_message(role, content, items=None, image_paths=None):
    """Display a chat message with minimal styling"""
    if role == "user":
        st.markdown(f'<div class="user-message">> {content}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-message">SYSTEM: OUTFIT GENERATED</div>', unsafe_allow_html=True)
        
        if items:
            st.markdown('<div class="fashion-items">', unsafe_allow_html=True)
            st.markdown("**OUTPUT:**")
            for i, item in enumerate(items, 1):
                st.markdown(f"{i:02d}. {item}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if image_paths:
            st.markdown("**VISUAL REFERENCES:**")
            cols = st.columns(min(len(image_paths), 4))
            
            for idx, image_path in enumerate(image_paths):
                with cols[idx % 4]:
                    image = display_image_safely(image_path)
                    st.image(image, caption=f"REF_{idx+1:02d}", use_column_width=True)

# Main UI
st.markdown('<h1 class="main-header">CLIP + GEMMA FASHION ENGINE</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="sidebar-header">CONFIG</div>', unsafe_allow_html=True)
    api_url = st.text_input("API_ENDPOINT", value=st.session_state.api_url)
    st.session_state.api_url = api_url
    
    st.markdown("---")
    st.markdown('<div class="sidebar-header">PRESETS</div>', unsafe_allow_html=True)
    
    presets = [
        "total black winter",
        "minimal summer casual", 
        "monochrome gym wear",
        "dark business formal",
        "grey streetwear",
        "all white minimal"
    ]
    
    for preset in presets:
        if st.button(preset, key=f"preset_{preset}"):
            st.session_state.chat_history.append({
                "role": "user", 
                "content": preset
            })
            
            with st.spinner("PROCESSING..."):
                result = call_fashion_api(preset, st.session_state.api_url)
            
            if "error" not in result:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "OUTFIT GENERATED",
                    "items": result.get("items", []),
                    "image_paths": result.get("image_paths", [])
                })
            
            st.rerun()

    st.markdown("---")
    if st.button("CLEAR_HISTORY"):
        st.session_state.chat_history = []
        st.rerun()

# Chat interface
st.markdown("### FASHION_INPUT")

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        display_chat_message(
            message["role"], 
            message["content"],
            message.get("items"),
            message.get("image_paths")
        )

# Input form
with st.form("fashion_input", clear_on_submit=True):
    user_input = st.text_input(
        "",
        placeholder="describe target aesthetic..."
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        send_button = st.form_submit_button("GENERATE", type="primary", use_container_width=True)

# Handle input
if send_button and user_input.strip():
    st.session_state.chat_history.append({
        "role": "user",
        "content": user_input
    })
    
    with st.spinner("PROCESSING..."):
        result = call_fashion_api(user_input, st.session_state.api_url)
    
    if "error" in result:
        st.error(f"ERROR: {result['error']}")
        st.info("VERIFY API CONNECTION")
    else:
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": "OUTFIT GENERATED",
            "items": result.get("items", []),
            "image_paths": result.get("image_paths", [])
        })
    
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div class="tech-info">
        CLIP VISUAL ENCODER | GEMMA LLM | STREAMLIT INTERFACE<br>
        NEURAL FASHION SYNTHESIS v1.0
    </div>
    """, 
    unsafe_allow_html=True
)

# System info for first-time users
if len(st.session_state.chat_history) == 0:
    st.info("""
    **SYSTEM_STATUS: READY**
    
    Input fashion requirements for AI-generated outfit recommendations.
    
    SUPPORTED_QUERIES:
    - Color themes: "total black", "monochrome grey"
    - Seasonal: "winter minimal", "summer technical"
    - Context: "gym functional", "office formal"
    
    Select presets from sidebar or input custom parameters.
    """, icon="ℹ️")