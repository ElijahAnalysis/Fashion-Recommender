import streamlit as st
import requests
import json
from PIL import Image
import os
from io import BytesIO
import base64

# Configure the page
st.set_page_config(
    page_title="OpenAI CLIP + Gemma - Your AI Stylist",
    page_icon="✨",
    layout="wide"
)

# Minimalistic CSS with grey-white theme
st.markdown("""
<style>
    /* Clean grey-white background */
    .stApp {
        background-color: #f8f9fa;
    }
    
    .main .block-container {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        margin-top: 1rem;
    }
    
    /* Clean title styling */
    .main-title {
        font-size: 36px;
        font-weight: 700;
        text-align: center;
        margin-bottom: 8px;
        font-family: 'Arial', sans-serif;
    }
    
    .openai-text {
        color: #2c3e50;
        background-color: #ecf0f1;
        padding: 6px 12px;
        border-radius: 6px;
        margin-right: 8px;
        border: 1px solid #bdc3c7;
    }
    
    .gemma-text {
        color: #6366f1;
        font-weight: 700;
    }
    
    .plus-sign {
        color: #2c3e50;
        margin: 0 8px;
        font-weight: 600;
    }
    
    .subtitle {
        font-size: 18px;
        font-weight: 400;
        text-align: center;
        margin-bottom: 40px;
        color: #7f8c8d;
        font-style: italic;
    }
    
    /* Clean chat styling */
    .chat-message {
        padding: 16px 20px;
        margin: 12px 0;
        border-radius: 12px;
        max-width: 85%;
        line-height: 1.5;
    }
    
    .user-msg {
        background-color: #e8f4fd;
        margin-left: auto;
        text-align: right;
        color: #2c3e50;
        border-left: 4px solid #3498db;
        font-size: 32px;
        font-weight: bold;
    }
    
    .assistant-msg {
        background-color: #f1f2f6;
        margin-right: auto;
        color: #2c3e50;
        border-left: 4px solid #95a5a6;
        font-size: 15px;
    }
    
    /* Clean button styling */
    .stButton > button {
        width: 100%;
        margin: 4px 0;
        background-color: #ecf0f1;
        color: #2c3e50;
        border: 1px solid #bdc3c7;
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background-color: #d5dbdb;
        border-color: #95a5a6;
    }
    
    /* Clean form styling */
    .gender-selector {
        background-color: #f8f9fa;
        padding: 18px;
        border-radius: 10px;
        margin: 20px 0;
        border: 1px solid #e9ecef;
    }
    
    /* Input styling */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #ced6e0;
        font-size: 14px;
    }
    
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #ced6e0;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2c3e50 !important;
        font-weight: 600;
    }
    
    /* Info box styling */
    .stInfo {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
    }
    
    /* Divider styling */
    hr {
        border-color: #e9ecef;
        margin: 24px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://127.0.0.1:4321"

if 'selected_gender' not in st.session_state:
    st.session_state.selected_gender = "men"

if 'num_items' not in st.session_state:
    st.session_state.num_items = 3

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

def format_user_prompt(text, gender, num_items):
    """Format user prompt with gender and item count specification"""
    return f"{text} ({gender}) ({num_items} items)"

# Header
st.markdown('''
<h1 class="main-title">
    <span class="openai-text">OpenAI CLIP</span><span class="plus-sign">+</span><span class="gemma-text">Gemma</span>
</h1>
<p class="subtitle">Your AI Stylist</p>
''', unsafe_allow_html=True)

# Layout
col1, col2 = st.columns([1, 3])

# Sidebar
with col1:
    # Gender Selection
    st.subheader("Gender Selection")
    gender_option = st.selectbox(
        "Select gender for outfit recommendations:",
        ["men", "women"],
        index=0 if st.session_state.selected_gender == "men" else 1
    )
    st.session_state.selected_gender = gender_option
    
    # Number of Items Selection
    st.subheader("Number of Items")
    num_items_option = st.selectbox(
        "Select number of outfit items:",
        [3, 4, 5, 6],
        index=[3, 4, 5, 6].index(st.session_state.num_items)
    )
    st.session_state.num_items = num_items_option
    
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
            # Format the preset with selected gender and number of items
            formatted_prompt = format_user_prompt(preset, st.session_state.selected_gender, st.session_state.num_items)
            
            st.session_state.chat_history.append({
                "role": "user",
                "content": preset,
                "formatted_prompt": formatted_prompt
            })
            
            with st.spinner("Generating..."):
                result = call_fashion_api(formatted_prompt, "http://127.0.0.1:4321")
            
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
    # Chat container - no header
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                # Show only the exact user input
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
        # Format the prompt with selected gender and number of items
        formatted_prompt = format_user_prompt(user_input.strip(), st.session_state.selected_gender, st.session_state.num_items)
        
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input.strip(),
            "formatted_prompt": formatted_prompt
        })
        
        # Call API with formatted prompt
        with st.spinner("Generating outfit..."):
            result = call_fashion_api(formatted_prompt, "http://127.0.0.1:4321")
        
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