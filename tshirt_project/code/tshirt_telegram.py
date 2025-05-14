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
import tensorflow as tf

# Enable logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
)
logger = logging.getLogger(__name__)

# Global variables
EMBEDDINGS_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\Fashion-Recommender\tshirt_project\data\efficientnetb0_fashion_embedings_df.csv.gz"

MODEL_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\Fashion-Recommender\tshirt_project\models\efficientnetb0_fashion.keras"

KMEANS_PATH = r"C:\Users\User\Desktop\DATA SCIENCE\Github\Fashion-Recommender\tshirt_project\models\efficientnetb0_fashion_kmeans41.joblib"

# Load embeddings dataframe and perform validation
def load_embeddings():
    print("Loading embeddings dataframe...")
    try:
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
                # Create a dummy cluster column if none exists
                embeddings_df['cluster'] = 0
        
        # Extract the embedding feature columns 
        # (assuming they are numeric columns that aren't 'image_path' or 'cluster')
        embedding_cols = [col for col in embeddings_df.columns 
                        if col not in ['image_path', 'cluster'] and 
                        pd.api.types.is_numeric_dtype(embeddings_df[col])]
        
        print(f"Found {len(embedding_cols)} embedding feature columns")
        
        if len(embedding_cols) == 0:
            print("WARNING: No embedding feature columns found in the dataframe!")
            
        # Print cluster distribution
        cluster_counts = embeddings_df['cluster'].value_counts()
        print(f"Cluster distribution: {cluster_counts}")
        
        # Check for skewed distribution (one cluster has >50% of data)
        max_cluster_pct = cluster_counts.max() / cluster_counts.sum() * 100
        if max_cluster_pct > 50:
            dominant_cluster = cluster_counts.idxmax()
            print(f"WARNING: Highly skewed distribution! Cluster {dominant_cluster} contains {max_cluster_pct:.1f}% of all data")
        
        # Validate image paths
        sample_size = min(10, len(embeddings_df))
        sample_paths = embeddings_df['image_path'].sample(sample_size).tolist()
        valid_paths = sum(1 for path in sample_paths if os.path.exists(path))
        print(f"Path validation: {valid_paths}/{sample_size} sample paths exist")
        
        return embeddings_df, embedding_cols
        
    except Exception as e:
        print(f"Error loading embeddings: {str(e)}")
        # Create a minimal dummy dataframe as fallback
        print("Creating fallback embeddings dataframe")
        dummy_df = pd.DataFrame({
            'image_path': [],
            'cluster': []
        })
        return dummy_df, []

# Load models
def load_models():
    print("Loading models...")
    
    # Load the image embedding model
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    
    # Load the KMeans clustering model
    kmeans = joblib.load(KMEANS_PATH)
    print(f"KMeans n_clusters: {kmeans.n_clusters}")
    print(f"KMeans feature input dimensionality: {kmeans.cluster_centers_.shape[1]}")
    
    print("Models loaded successfully!")
    return model, kmeans

# Preprocess image for model - matching the training preprocessing
def preprocess_image(image_bytes):
    try:
        # Convert BytesIO object or bytearray to bytes if needed
        if not isinstance(image_bytes, bytes):
            if isinstance(image_bytes, BytesIO):
                image_bytes = image_bytes.getvalue()
            else:
                image_bytes = bytes(image_bytes)
        
        # Convert bytes to string tensor (tf.io.decode_image expects string tensor)
        img_bytes_tensor = tf.convert_to_tensor(image_bytes, dtype=tf.string)
        
        # Decode the image
        img_tensor = tf.io.decode_image(
            img_bytes_tensor, 
            channels=3,  # Ensure RGB
            expand_animations=False  # Don't handle GIFs
        )
        
        # Log original size before resizing
        print(f"Original image shape: {img_tensor.shape}")
        
        # Resize to match training input size (224x224)
        img_tensor = tf.image.resize(img_tensor, (224, 224))
        
        # Normalize to [0,1] as was done during training
        img_tensor = img_tensor / 255.0
        
        # Set proper shape
        img_tensor.set_shape((224, 224, 3))
        
        # Check tensor properties
        print(f"Processed tensor shape: {img_tensor.shape}, Range: [{tf.reduce_min(img_tensor)}, {tf.reduce_max(img_tensor)}]")
        
        # Add batch dimension
        img_tensor = tf.expand_dims(img_tensor, 0)
        
        return img_tensor
        
    except Exception as e:
        print(f"Error in preprocess_image: {str(e)}")
        raise

# Get image embedding
def get_embedding(model, image_tensor):
    try:
        # Check if image tensor has correct shape for model input
        expected_shape = tuple(model.input_shape[1:])  # Remove batch dimension
        actual_shape = tuple(image_tensor.shape[1:])
        
        if expected_shape != actual_shape:
            print(f"WARNING: Input shape mismatch! Expected {expected_shape}, got {actual_shape}")
            # Try to reshape if dimensions are compatible
            image_tensor = tf.reshape(image_tensor, (-1,) + expected_shape)
            
        # Get the embedding from the model
        embedding = model.predict(image_tensor, verbose=0)
        
        # Print embedding statistics for debugging
        print(f"Embedding shape: {embedding.shape}")
        print(f"Embedding stats: Min={np.min(embedding)}, Max={np.max(embedding)}, Mean={np.mean(embedding)}")
        
        # Return the embedding without batch dimension
        return embedding[0]
        
    except Exception as e:
        print(f"Error in get_embedding: {str(e)}")
        raise

# Find similar t-shirts based on embedding
def find_similar_tshirts(embedding, kmeans, embeddings_df, embedding_cols, num_results=5):
    try:
        print("Finding similar t-shirts...")
        
        # 1. Get the cluster for the input embedding
        cluster_id = kmeans.predict([embedding])[0]
        print(f"Predicted cluster: {cluster_id}")
        
        # 2. Get all t-shirts from the same cluster
        cluster_items = embeddings_df[embeddings_df['cluster'] == cluster_id]
        print(f"Found {len(cluster_items)} items in the same cluster")
        
        # If we don't have enough items in this cluster, broaden the search
        if len(cluster_items) < num_results:
            print(f"Not enough items in cluster {cluster_id}, using all items")
            cluster_items = embeddings_df
        
        # 3. Compute distances between input embedding and all cluster items
        # First, convert embedding_cols to a list of embeddings for each item
        cluster_embeddings = cluster_items[embedding_cols].values
        
        # Compute Euclidean distance between input embedding and all cluster embeddings
        distances = np.linalg.norm(cluster_embeddings - embedding, axis=1)
        
        # 4. Get indices of the closest items
        # Add distances as a new column
        cluster_items_with_dist = cluster_items.copy()
        cluster_items_with_dist['distance'] = distances
        
        # Sort by distance and get top n results
        closest_items = cluster_items_with_dist.sort_values('distance').iloc[:num_results]
        
        print(f"Top {num_results} closest items:")
        for i, (_, item) in enumerate(closest_items.iterrows()):
            print(f"{i+1}. {os.path.basename(item['image_path'])}, distance={item['distance']:.4f}")
        
        return closest_items[['image_path', 'distance']].values.tolist()
        
    except Exception as e:
        print(f"Error in find_similar_tshirts: {str(e)}")
        # Return a random selection of t-shirts as fallback
        print("Using fallback random selection")
        random_items = embeddings_df.sample(min(num_results, len(embeddings_df)))
        return random_items['image_path'].tolist()

# Telegram command handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        f"Hi {user.mention_html()}! I'm the T-shirt Recommendation Bot.\n\n"
        f"Send me a photo of a T-shirt you like, and I'll recommend similar ones from our catalog!"
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Here's how to use this bot:\n\n"
        "1. Send a photo of a T-shirt you like\n"
        "2. Wait for me to analyze it\n"
        "3. I'll send you photos of similar T-shirts from our catalog\n\n"
        "Available commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message"
    )

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process photos sent by the user."""
    try:
        # Inform user that processing has started
        processing_msg = await update.message.reply_text("Processing your T-shirt image... Please wait.")
        
        # Get the largest photo available
        photo = update.message.photo[-1]
        
        # Download the photo
        photo_file = await context.bot.get_file(photo.file_id)
        photo_bytes = await photo_file.download_as_bytearray()
        
        # Convert to bytes for TensorFlow processing
        photo_bytes_data = bytes(photo_bytes)
        
        # Log for debugging
        print(f"Downloaded photo of size: {len(photo_bytes_data)} bytes")
        
        # Preprocess the image
        image_tensor = preprocess_image(photo_bytes_data)
        
        # Get embedding
        embedding = get_embedding(context.bot_data['model'], image_tensor)
        
        # Find similar t-shirts
        similar_tshirts = find_similar_tshirts(
            embedding,
            context.bot_data['kmeans'],
            context.bot_data['embeddings_df'],
            context.bot_data['embedding_cols']
        )
        
        # Update user on progress
        await processing_msg.edit_text("Found similar t-shirts! Sending recommendations...")
        
        # Send results
        await update.message.reply_text(f"Here are {len(similar_tshirts)} t-shirts similar to yours:")
        
        # Send each recommendation with similarity score
        for i, (tshirt_path, distance) in enumerate(similar_tshirts):
            # Skip if file doesn't exist
            if not os.path.exists(tshirt_path):
                print(f"Warning: File not found: {tshirt_path}")
                continue
                
            try:
                # Get filename for caption
                filename = os.path.basename(tshirt_path)
                
                # Calculate similarity score (inverse of distance, normalized to 0-100%)
                # Lower distance = higher similarity
                max_distance = 10.0  # Set a reasonable upper bound for distance
                similarity = max(0, 100 * (1 - min(distance, max_distance) / max_distance))
                
                caption = f"#{i+1}: {filename}\nSimilarity: {similarity:.1f}%"
                
                # Send the image
                with open(tshirt_path, 'rb') as img_file:
                    await context.bot.send_photo(
                        chat_id=update.effective_chat.id,
                        photo=img_file,
                        caption=caption
                    )
            except Exception as img_e:
                print(f"Error sending recommendation {i+1}: {str(img_e)}")
                await update.message.reply_text(f"Couldn't load recommendation #{i+1}: {filename}")
        
        # Final message
        await update.message.reply_text("That's all! Send another T-shirt photo for more recommendations.")
        
    except Exception as e:
        logger.error(f"Error processing photo: {str(e)}", exc_info=True)
        await update.message.reply_text(
            "Sorry, I encountered an error while processing your image. Please try again with a different photo."
        )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle text messages."""
    await update.message.reply_text(
        "Please send me a photo of a T-shirt you like, and I'll recommend similar ones!"
    )

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log errors caused by updates."""
    logger.error(f"Update {update} caused error {context.error}", exc_info=context.error)
    
    # Notify user of error
    if update and update.effective_message:
        await update.effective_message.reply_text(
            "Sorry, something went wrong. Please try again later."
        )

def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token("your_token").build()
    
    # Load data and models
    print("Initializing bot...")
    embeddings_df, embedding_cols = load_embeddings()
    model, kmeans = load_models()
    
    # Store loaded data in application context
    application.bot_data['embeddings_df'] = embeddings_df
    application.bot_data['embedding_cols'] = embedding_cols
    application.bot_data['model'] = model
    application.bot_data['kmeans'] = kmeans
    
    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    
    # Add photo handler
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    
    # Add message handler for text messages
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Register error handler
    application.add_error_handler(error_handler)
    
    # Run the bot
    print("Starting bot...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
