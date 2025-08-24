import streamlit as st
import requests
import json
from PIL import Image
import os
from io import BytesIO
import base64

# Configure the page
st.set_page_config(
    page_title="OpenAI CLIP + Gemma Fashion Engine",
    page_icon="🔷",
    layout="wide"
)

# Simple CSS
st.markdown("""
<style>
    .main-title {
        font-size: 32px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 30px;
        color: #1f1f1f;
    }
    
    .openai-text {
        color: white;
        background-color: #000000;
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    .gemma-text {
        color: #6366f1;
        background-color: #000000;
        padding: 4px 8px;
        border-radius: 4px;
    }
    
    .plus-sign {
        color: #10b981;
        font-weight: 700;
    }
    
    .chat-message {
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        max-width: 80%;
    }
    
    .user-msg {
        background-color: #e3f2fd;
        margin-left: auto;
        text-align: right;
        color: #1565c0;
    }
    
    .assistant-msg {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        margin-right: auto;
    }
    
    .outfit-items {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    
    .stButton > button {
        width: 100%;
        margin: 2px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:4321"

def call_fashion_api(text, api_url):
    """Call the fashion API"""
    try:
        response = requests.post(
            f"{api_url}/make_look",
            json={"text": text},
            timeout=30
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection failed: {str(e)}"}

def display_image_safely(image_path):
    """Display image safely"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            return Image.new('RGB', (200, 200), color='#cccccc')
    except:
        return Image.new('RGB', (200, 200), color='#cccccc')

# Header
st.markdown('''
<h1 class="main-title">
    <span class="openai-text">OpenAI CLIP</span> 
    <span class="plus-sign">+</span> 
    <span class="gemma-text">Gemma</span> 
    Fashion Engine
</h1>
''', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 3])

# Sidebar
with col1:
    st.header("Settings")
    
    # API URL
    api_url = st.text_input("API URL", value=st.session_state.api_url)
    st.session_state.api_url = api_url
    
    st.header("Quick Presets")
    
    presets = [
        "minimal black outfit",
        "business casual",
        "summer casual",
        "winter layers",
        "gym wear",
        "date night",
        "office formal",
        "weekend comfort"
    ]
    
    for preset in presets:
        if st.button(preset):
            st.session_state.chat_history.append({
                "role": "user",
                "content": preset
            })
            
            with st.spinner("Generating..."):
                result = call_fashion_api(preset, st.session_state.api_url)
            
            if "error" not in result:
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": "Generated outfit",
                    "items": result.get("items", []),
                    "image_paths": result.get("image_paths", [])
                })
            else:
                st.error(result["error"])
            
            st.rerun()
    
    st.divider()
    
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat area
with col2:
    st.header("Chat")
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        if len(st.session_state.chat_history) == 0:
            st.info("Start by typing a message or selecting a preset from the sidebar.")
        else:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f'<div class="chat-message user-msg">{message["content"]}</div>', 
                              unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="chat-message assistant-msg">Generated outfit successfully</div>', 
                              unsafe_allow_html=True)
                    
                    # Show images only
                    if message.get("image_paths"):
                        st.write("**Reference Images:**")
                        
                        # Create columns for images
                        num_images = len(message["image_paths"])
                        if num_images > 0:
                            cols = st.columns(min(num_images, 4))
                            
                            for idx, image_path in enumerate(message["image_paths"]):
                                with cols[idx % 4]:
                                    image = display_image_safely(image_path)
                                    st.image(image, caption=f"Ref {idx+1}", use_column_width=True)
    
    st.divider()
    
    # Input area
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Your message:",
            placeholder="Describe the style you want...",
            height=100
        )
        
        submitted = st.form_submit_button("Generate Outfit")
    
    if submitted and user_input.strip():
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip()
        })
        
        # Call API
        with st.spinner("Generating outfit..."):
            result = call_fashion_api(user_input.strip(), st.session_state.api_url)
        
        if "error" in result:
            st.error(result["error"])
        else:
            # Add assistant response
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": "Generated outfit",
                "items": result.get("items", []),
                "image_paths": result.get("image_paths", [])
            })
        
        st.rerun()

# Footer
st.divider()
st.write("**System:** OpenAI CLIP + Gemma LLM + Streamlit Interface")