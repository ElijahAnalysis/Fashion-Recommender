from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast

from transformers import AutoTokenizer, BitsAndBytesConfig, Gemma3ForCausalLM

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn

app = FastAPI()

# Force CPU usage
torch.set_default_tensor_type('torch.FloatTensor')

#### GEMMA SETUP
# Load model on CPU only (no quantization)
gemma_1b = Gemma3ForCausalLM.from_pretrained(
    "google/gemma-3-1b-it",
    device_map="cpu"   # ensures CPU only
).eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token



#### CLIP SETUP - CPU only
clip_model_id = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(clip_model_id, torch_dtype=torch.float32)
clip_model.to('cpu')
clip_processor = CLIPProcessor.from_pretrained(clip_model_id, use_fast=True)

#### LOAD DATA
clip_image_data = torch.load(r"C:\Users\User\Downloads\clip_image_embeddings.pt", map_location='cpu')
clip_image_embeddings = clip_image_data['image_embeddings'].to('cpu')
image_paths = clip_image_data['image_paths']


class UserLookRequest(BaseModel):

    text : str



#### FUNCTIONS
def process_gemma_output(outputs):
    raw_text = outputs[0]

    # Step 1: get the model response section
    start = raw_text.find("<start_of_turn>model") + len("<start_of_turn>model\n")
    end = raw_text.find("<end_of_turn>", start)
    list_str = raw_text[start:end].strip()

    # Step 2: convert string -> Python list
    fashion_items_text = ast.literal_eval(list_str)

    ### Embedding for text 
    processed_fashion_items = clip_processor.tokenizer(
        fashion_items_text,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to('cpu')

    with torch.no_grad():
        item_text_embeddings = clip_model.get_text_features(**processed_fashion_items)

    item_text_embeddings_normalized = F.normalize(item_text_embeddings, p=2, dim=-1)

    #### Compute cosine sim for images 
    text_image_cosine_sim_matrix = torch.matmul(
        item_text_embeddings_normalized.to('cpu'), 
        clip_image_embeddings.to('cpu').T
    )

    #### Find best match 
    best_image_indices = text_image_cosine_sim_matrix.argmax(dim=1)  
    top_image_paths = [image_paths[i] for i in best_image_indices.tolist()]

    return top_image_paths



@app.post("/make_look")
async def make_look(request: UserLookRequest):
    
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict fashion assistant specializing in **men's and women's clothing**. "
                "You must ALWAYS respect the user's requested gender and colors/themes. "
                "Your only output must be **exactly one valid Python list**, with no text, no formatting, no explanations. "
                ""
                "STRICT RULES: "
                "1. The list must contain **minimum 3 and maximum 6 items**. If you cannot meet this, output nothing. "
                "2. Each item must be a **single clothing or accessory description**. "
                "3. Each description must include **specific details** (colors, fabrics, styles). "
                "4. Each item must belong to **different categories** (bottoms, tops, outer layers, shoes, optional accessories). "
                "   - Only ONE item per category. "
                "5. If the user specifies a gender, you MUST return clothing appropriate for that gender. "
                "6. You must include the gender in every item description, in the format: 'men ...' or 'women ...'. "
                "7. If the user specifies a color or theme, ALL items must strictly follow it. NO EXCEPTIONS. "
                "   - 'total black' means EVERY item must be black. "
                "   - 'all white' means EVERY item must be white. "
                "   - 'monochrome blue' means EVERY item must be blue. "
                "8. You must NEVER output fewer than 3 or more than 6 items. "
                "9. Use only plain strings inside the Python list. "
                "10. NEVER repeat categories - one bottom, one top, one outer layer, one shoe maximum. "
                ""
                "ALLOWED CATEGORIES: "
                "- BOTTOMS: pants, trousers, jeans, chinos, shorts, joggers, skirts, dresses "
                "- TOPS: t-shirts, polo shirts, blouses, crop tops, shirts "
                "- OUTER LAYERS: hoodies, sweatshirts, zip-hoodies, cardigans, jackets, coats "
                "- SHOES: sneakers, loafers, boots, dress shoes, canvas shoes, running shoes, casual shoes, heels, sandals "
                "- ACCESSORIES (optional, max 2): hats, caps, glasses, watches, belts, bags, jewelry, scarves "
                ""
                "SEASONAL RULES: "
                "- WARM weather: lighter fabrics, breathable pieces, skirts, dresses, crop tops, sandals. "
                "- COLD weather: thick fabrics, warm outer layers, boots, scarves, cozy accessories. "
                ""
                "CRITICAL COLOR ENFORCEMENT: "
                "When user specifies a color theme, you must check each item contains that exact color. "
                "Examples of CORRECT responses: "
                "- 'total black for men' → ['men black cotton t-shirt', 'men black denim jeans', 'men black leather boots', 'men black bomber jacket'] "
                "- 'all white for women' → ['women white cotton blouse', 'women white linen pants', 'women white canvas sneakers'] "
                ""
                "Examples of WRONG responses (DO NOT DO): "
                "- 'total black for men' → ['men black shirt', 'men beige chinos', 'men white sneakers'] (violates color rule) "
                "- Any response with repeated categories like jeans AND chinos (both bottoms) "
                ""
                "OUTPUT FORMAT: "
                "Return ONLY a valid Python list (e.g., ['men black linen shirt', 'men black chinos', 'men black sneakers']). "
                "NO explanations, NO text before or after, NO code blocks, NO markdown formatting. "
            )
        },
        {
            "role": "user",
            "content": f"{request.text}"
        }
    ]
    



    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to('cpu')

    with torch.inference_mode():
        outputs = gemma_1b.generate(**inputs, max_new_tokens=64)

    outputs = tokenizer.batch_decode(outputs)

    print(">>> USER MESSAGE:", request.text)
    print(">>> GEMMA RAW OUTPUT:", outputs[0])
    
    top_image_paths = process_gemma_output(outputs)
    
    return {
        "items": ast.literal_eval(outputs[0].split("<start_of_turn>model\n")[1].split("<end_of_turn>")[0].strip()),
        "image_paths": top_image_paths
    }
    
    




    


