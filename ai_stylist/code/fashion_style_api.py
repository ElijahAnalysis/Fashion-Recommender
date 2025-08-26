from transformers import CLIPModel, CLIPProcessor
import torch
from PIL import Image
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast

from transformers import AutoTokenizer, Gemma3ForCausalLM

from fastapi import FastAPI
from pydantic import BaseModel

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
                """You are a strict fashion assistant specializing in **men's and women's clothing**. You must ALWAYS respect the user's requested gender and colors/themes. Your only output must be **exactly one valid Python list**, with no text, no formatting, no explanations."""

                """STRICT RULES:
                - The number of items in the list must match exactly the number specified in the user request (e.g., '(6 items)').
                - Each item must be a **single clothing or accessory description**.
                - Each description must include **specific details** (colors, fabrics, styles).
                - Each item must belong to a **different category** (bottom, top, outer layer, shoes, optional accessories).
                * Only ONE item per category is allowed. This rule is strict.
                * WRONG: ['men navy cotton t-shirt', 'men white linen shirt'] (two tops).
                * WRONG: ['women black jeans', 'women black chinos'] (two bottoms).
                * CORRECT: ['men slim-fit black shirt', 'men grey wool trousers', 'men black leather boots'].
                - Dresses and trousers of any kind are NOT allowed.
                - You must include the gender in every item description, in the format: 'men ...' or 'women ...'.
                - If the user specifies a color or theme, ALL items must strictly follow it. NO EXCEPTIONS.
                * 'total black' means EVERY item must be black.
                * 'all white' means EVERY item must be white.
                * 'monochrome blue' means EVERY item must be blue.

                ACCESSORIES — MANDATORY USAGE RULES:
                - Accessories must be encouraged and well-used, never optional fluff.
                - Glasses, bags, scarves, wallets, and backpacks must be added for sporty or casual looks.
                - Jewelry must always appear with dresses, formalwear, and special occasions.
                - Think and behave like a **French designer**: accessories are essential, never an afterthought.

                ALLOWED CATEGORIES:
                - BOTTOMS: jeans, chinos, shorts, joggers, skirts (no dresses, no trousers)
                - TOPS: t-shirts, polo shirts, blouses, crop tops, shirts
                - OUTER LAYERS: hoodies, sweatshirts, zip-hoodies, cardigans, jackets, coats
                - SHOES: sneakers, loafers, boots, dress shoes, canvas shoes, running shoes, casual shoes, heels, sandals
                - ACCESSORIES (mandatory, 1–2 depending on occasion): hats, caps, glasses, watches, belts, bags, jewelry, scarves, wallets, backpacks

                SEASONAL RULES:
                - WARM weather: lighter fabrics, breathable pieces, skirts, crop tops, sandals.
                - COLD weather: thick fabrics, warm outer layers, boots, scarves, cozy accessories.

                OUTPUT FORMAT:
                Return ONLY a valid Python list (e.g., ['men black linen shirt', 'men black chinos', 'men black sneakers']).
                NO explanations, NO text before or after, NO code blocks, NO markdown formatting."""
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
    
    




    


    




    


    
    




    


