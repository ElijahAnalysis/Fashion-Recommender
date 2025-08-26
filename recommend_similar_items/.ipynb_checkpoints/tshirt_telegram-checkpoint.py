import os
import logging
import numpy as np
import pandas as pd
import random
import gzip
import joblib
from io import BytesIO
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from PIL import Image
import tensorflow as tf

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global variables
EMBEDDINGS_PATH = r"C:\Users\User\Desktop\tshirt_embeddings_df.csv.gz"
MODEL_PATH = r"C:\Users\User\Desktop\mobilenetv3s_tshirt.keras"
KMEANS_PATH = r"C:\Users\User\Desktop\tshirt_kmeans.joblib"

# Load embeddings dataframe
def load_embeddings():
    print("Loading embeddings dataframe...")
    embeddings_df = pd.read_csv(EMBEDDINGS_PATH, compression='gzip')
    print(f"Loaded dataframe with shape: {embeddings_df.shape}")
    print(f"Columns: {embeddings_df.columns}")
    
    # Verify 'image_path' and 'cluster' columns exist
    if 'image_path' not in embeddings_df.columns:
        # Check if there's a similar column that might contain the image paths
        path_candidates = [col for col in embeddings_df.columns if 'path' in col.lower() or 'file' in col.lower() or 'image' in col.lower()]
        if path_candidates:
            print(f"'image_path' column not found. Renaming '{path_candidates[0]}' to 'image_path'")
            embeddings_df.rename(columns={path_candidates[0]: 'image_path'}, inplace=True)
        else:
            print("Warning: No 'image_path' column found!")
    
    if 'cluster' not in embeddings_df.columns:
        # Check if there's a similar column that might contain cluster information
        cluster_candidates = [col for col in embeddings_df.columns if 'cluster' in col.lower() or 'class' in col.lower() or 'category' in col.lower()]
        if cluster_candidates:
            print(f"'cluster' column not found. Renaming '{cluster_candidates[0]}' to 'cluster'")
            embeddings_df.rename(columns={cluster_candidates[0]: 'cluster'}, inplace=True)
        else:
            print("Warning: No 'cluster' column found!")
    
    return embeddings_df

# Load models
def load_models():
    print("Loading models...")
    # Load the image embedding model
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Load the KMeans clustering model
    kmeans = joblib.load(KMEANS_PATH)
    
    print("Models loaded successfully!")
    return model, kmeans

# Preprocess image for model
def preprocess_image(image_bytes):
    img = Image.open(BytesIO(image_bytes))
    img = img.resize((224, 224))  # MobileNet input size
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array = tf.keras.applications.mobilenet_v3.preprocess_input(img_array)
    return img_array

# Get image embedding
def get_embedding(model, image_array):
    embedding = model.predict(image_array, verbose=0)
    return embedding[0]  # Return the embedding without batch dimension

# Predict cluster
def predict_cluster(kmeans, embedding):
    # Convert embedding to double/float64 to match KMeans expectations
    embedding_double = np.array(embedding, dtype=np.float64)
    cluster = kmeans.predict([embedding_double])[0]
    return cluster

# Find similar images
def find_similar_images(embeddings_df, cluster, num_images=3):
    try:
        print(f"Finding images in cluster {cluster}")
        print(f"DataFrame has columns: {embeddings_df.columns}")
        print(f"Number of rows in dataframe: {len(embeddings_df)}")

        if 'cluster' not in embeddings_df.columns:
            print("Error: 'cluster' column not found in dataframe")
            return []

        if 'image_path' not in embeddings_df.columns:
            print("Error: 'image_path' column not found in dataframe")
            return []

        cluster_df = embeddings_df[embeddings_df['cluster'] == cluster]
        print(f"Found {len(cluster_df)} images in cluster {cluster}")

        if len(cluster_df) <= num_images:
            similar_images = cluster_df['image_path'].tolist()
        else:
            similar_images = cluster_df['image_path'].sample(num_images).tolist()

        print(f"Selected {len(similar_images)} similar images")
        return similar_images

    except Exception as e:
        print(f"Error in find_similar_images: {str(e)}")
        return []


# Start command handler
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Hi! Send me a T-shirt image, and I'll recommend similar T-shirts from our collection."
    )

# Help command handler
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Send me a photo of a T-shirt, and I'll find similar T-shirts for you!"
    )

# Image handler
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Get user data from context
    if 'embeddings_df' not in context.bot_data:
        await update.message.reply_text("Loading data, please wait...")
        try:
            context.bot_data['embeddings_df'] = load_embeddings()
            context.bot_data['model'], context.bot_data['kmeans'] = load_models()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            await update.message.reply_text(f"Error loading necessary data: {str(e)}")
            return
    
    embeddings_df = context.bot_data['embeddings_df']
    model = context.bot_data['model']
    kmeans = context.bot_data['kmeans']
    
    # Get photo file
    photo_file = await update.message.photo[-1].get_file()
    photo_bytes = await photo_file.download_as_bytearray()
    
    await update.message.reply_text("Processing your image...")
    
    try:
        # Preprocess image
        img_array = preprocess_image(photo_bytes)
        
        # Get embedding
        embedding = get_embedding(model, img_array)
        
        # Debug info
        print(f"Embedding shape: {embedding.shape}")
        
        # Predict cluster
        cluster = predict_cluster(kmeans, embedding)
        print(f"Image assigned to cluster: {cluster}")
        
        # Find similar images
        similar_images = find_similar_images(embeddings_df, cluster)
        
        if not similar_images:
            await update.message.reply_text(f"Could not find similar T-shirts in cluster {cluster}. Please try again with a different image.")
            return
        
        # Send back the similar images
        await update.message.reply_text(f"Found {len(similar_images)} similar T-shirts in cluster {cluster}:")
        
        images_sent = 0
        for img_path in similar_images:
            try:
                # Check if the file exists
                if os.path.exists(img_path):
                    await update.message.reply_photo(open(img_path, 'rb'))
                    images_sent += 1
                else:
                    print(f"Image not found: {img_path}")
            except Exception as img_error:
                print(f"Error sending image {img_path}: {str(img_error)}")
        
        if images_sent == 0:
            await update.message.reply_text("Could not retrieve any of the recommended images. They may be missing from the specified paths.")
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await update.message.reply_text(f"Error processing your image: {str(e)}")

# Text handler (for messages that aren't commands)
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Please send me a photo of a T-shirt to get recommendations!"
    )

def main() -> None:
    # Create the Application
    application = Application.builder().token("7978191456:AAGk5l4tbNWV1P52qPCks2xpwykODsjn3DI").build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    # Run the bot
    print("Starting bot...")
    application.run_polling()

if __name__ == "__main__":
    main()