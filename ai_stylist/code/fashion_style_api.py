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
            "role": "user",
            "content": (
                "You are a helpful fashion assistant specializing in men's clothing. "
                "Your job is to return exactly a valid Python list of 4 items, nothing else. "
                "Each item must be a short fashion item name (like 'black hoodie with print' or 'white t-shirt with stripes'). "
                
                "ALLOWED CATEGORIES (choose exactly ONE item from each): "
                "1. BOTTOMS: pants, trousers, jeans, chinos, shorts, joggers "
                "2. TOPS: t-shirts, polo shirts "
                "3. OUTER LAYERS: hoodies, sweatshirts, zip-hoodies, cardigans, jackets "
                "4. SHOES: sneakers, loafers, boots, dress shoes, canvas shoes, running shoes, casual shoes "
                
                "FORBIDDEN ITEMS: You must NOT suggest accessories (caps, hats, glasses, watches, belts, bags, jewelry), "
                "cosmetics, underwear, or any items not listed in the allowed categories above. "
                
                "STRICT RULES: "
                "1. You must suggest exactly ONE item from each of the 4 categories above. "
                "2. If the user specifies a color theme (like 'total black', 'total white', 'all grey'), "
                "   then ALL 4 items MUST match that color theme. "
                "3. Always specify the exact type of shoe (e.g., 'white sneakers', 'brown loafers', 'black boots'). "
                "4. Do NOT include any explanation, extra text, quotes, or formatting outside the list. "
                
                "SEASONAL ADJUSTMENTS: "
                "- For WARM weather (summer, hot day, beach, sunny walk): Choose lighter outer layers like "
                "  light jackets, thin cardigans, or lightweight zip-hoodies. Prefer shorts over pants when appropriate. "
                "- For COLD weather (winter, cold library, windy night): Choose warmer outer layers like "
                "  thick hoodies, heavy sweatshirts, or warm jackets. Always include pants/trousers, not shorts. "
                
                "OUTPUT FORMAT: Return exactly 4 items in this order: [bottom, top, outer layer, shoes] "
                
                "Example (total black winter): ['black pants', 'black t-shirt', 'black hoodie', 'black sneakers'] "
                "Example (summer casual): ['beige shorts', 'white polo shirt', 'light blue zip-hoodie', 'white canvas shoes'] "
                
                f"Now create a fashion list for: {request.text}"
            )
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
    
    top_image_paths = process_gemma_output(outputs)
    
    return {
        "items": ast.literal_eval(outputs[0].split("<start_of_turn>model\n")[1].split("<end_of_turn>")[0].strip()),
        "image_paths": top_image_paths
    }
    
    



