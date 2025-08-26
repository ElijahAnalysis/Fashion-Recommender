import logging
import requests
import json
import os
from io import BytesIO
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_BOT_TOKEN = "8201617946:AAEewYwqU47dtergdGJTPB9rQqcMkTFTGCg"  # Replace with your bot token
FASHION_API_URL = "http://127.0.0.1:4321"  # Your fashion API endpoint

class FashionBot:
    def __init__(self):
        self.user_preferences = {}  # Store user preferences in memory
        
    def get_user_prefs(self, user_id):
        """Get user preferences or set defaults"""
        if user_id not in self.user_preferences:
            self.user_preferences[user_id] = {
                'gender': 'men',
                'num_items': 3
            }
        return self.user_preferences[user_id]
    
    def call_fashion_api(self, text):
        """Call the fashion API"""
        try:
            response = requests.post(
                f"{FASHION_API_URL}/make_look",
                json={"text": text},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"API error: {response.status_code}"}
        except Exception as e:
            return {"error": f"Connection failed: {str(e)}"}
    
    def format_user_prompt(self, text, gender, num_items):
        """Format user prompt with gender and item count specification"""
        return f"{text} ({gender}) ({num_items} items)"

# Initialize bot instance
fashion_bot = FashionBot()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Send a message when the command /start is issued"""
    welcome_message = """
✨ *Welcome to AI Fashion Stylist Bot!* ✨

I'm powered by OpenAI CLIP + Gemma to help you create amazing outfits!

*Commands:*
• /start - Show this welcome message
• /settings - Change gender and item preferences
• /presets - Quick outfit presets
• /help - Get help

*How to use:*
Just describe the style you want, like:
• "minimal black outfit"
• "business casual for work"
• "summer vacation vibes"

Ready to get styled? 👗👔
    """
    
    await update.message.reply_text(
        welcome_message, 
        parse_mode='Markdown'
    )

async def settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show settings menu"""
    user_id = update.effective_user.id
    prefs = fashion_bot.get_user_prefs(user_id)
    
    keyboard = [
        [
            InlineKeyboardButton("👨 Men", callback_data="gender_men"),
            InlineKeyboardButton("👩 Women", callback_data="gender_women")
        ],
        [
            InlineKeyboardButton("3 items", callback_data="items_3"),
            InlineKeyboardButton("4 items", callback_data="items_4"),
            InlineKeyboardButton("5 items", callback_data="items_5"),
            InlineKeyboardButton("6 items", callback_data="items_6")
        ]
    ]
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    settings_text = f"""
⚙️ *Current Settings*

Gender: {prefs['gender'].title()}
Number of items: {prefs['num_items']}

Tap to change:
    """
    
    await update.message.reply_text(
        settings_text,
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def presets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show preset outfit options"""
    preset_options = [
        "minimal black outfit", "business casual", "summer casual", "winter layers",
        "gym wear", "date night", "office formal", "weekend comfort"
    ]
    
    keyboard = []
    for i in range(0, len(preset_options), 2):
        row = []
        row.append(InlineKeyboardButton(preset_options[i], callback_data=f"preset_{preset_options[i]}"))
        if i + 1 < len(preset_options):
            row.append(InlineKeyboardButton(preset_options[i + 1], callback_data=f"preset_{preset_options[i + 1]}"))
        keyboard.append(row)
    
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "👗 *Quick Outfit Presets*\n\nChoose a style:",
        reply_markup=reply_markup,
        parse_mode='Markdown'
    )

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button callbacks"""
    query = update.callback_query
    user_id = query.from_user.id
    data = query.data
    
    await query.answer()
    
    # Handle settings changes
    if data.startswith("gender_"):
        gender = data.split("_")[1]
        fashion_bot.user_preferences[user_id]['gender'] = gender
        await query.edit_message_text(f"✅ Gender set to: {gender.title()}")
        
    elif data.startswith("items_"):
        num_items = int(data.split("_")[1])
        fashion_bot.user_preferences[user_id]['num_items'] = num_items
        await query.edit_message_text(f"✅ Number of items set to: {num_items}")
        
    # Handle preset selections
    elif data.startswith("preset_"):
        preset = data.replace("preset_", "")
        await generate_outfit(query, preset, is_callback=True)

async def generate_outfit(update_or_query, text, is_callback=False):
    """Generate outfit from text description"""
    if is_callback:
        user_id = update_or_query.from_user.id
        message_func = update_or_query.edit_message_text
    else:
        user_id = update_or_query.effective_user.id
        message_func = update_or_query.message.reply_text
    
    prefs = fashion_bot.get_user_prefs(user_id)
    formatted_prompt = fashion_bot.format_user_prompt(text, prefs['gender'], prefs['num_items'])
    
    # Show loading message
    loading_msg = await message_func("🎨 Generating your outfit... Please wait!")
    
    try:
        # Call fashion API
        result = fashion_bot.call_fashion_api(formatted_prompt)
        
        if "error" in result:
            await loading_msg.edit_text(f"❌ Error: {result['error']}")
            return
        
        # Format response
        response_text = f"✨ *Outfit Generated!*\n\n📝 *Request:* {text}\n👤 *Gender:* {prefs['gender'].title()}\n🔢 *Items:* {prefs['num_items']}\n\n"
        
        # Add items if available
        if "items" in result and result["items"]:
            response_text += "*Recommended Items:*\n"
            for i, item in enumerate(result["items"], 1):
                response_text += f"{i}. {item}\n"
        
        await loading_msg.edit_text(response_text, parse_mode='Markdown')
        
        # Send images if available
        if "image_paths" in result and result["image_paths"]:
            await send_reference_images(update_or_query, result["image_paths"], is_callback)
            
    except Exception as e:
        await loading_msg.edit_text(f"❌ Error generating outfit: {str(e)}")

async def send_reference_images(update_or_query, image_paths, is_callback=False):
    """Send reference images to user"""
    if is_callback:
        chat_id = update_or_query.message.chat_id
        context = update_or_query._bot
    else:
        chat_id = update_or_query.effective_chat.id
        context = update_or_query.get_bot()
    
    for idx, image_path in enumerate(image_paths):
        try:
            if os.path.exists(image_path):
                with open(image_path, 'rb') as photo:
                    await context.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=f"📸 Reference Image {idx + 1}"
                    )
        except Exception as e:
            logger.error(f"Error sending image {image_path}: {e}")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle text messages"""
    text = update.message.text.strip()
    
    if not text:
        await update.message.reply_text("Please describe the outfit style you want! 👗")
        return
    
    await generate_outfit(update, text)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show help information"""
    help_text = """
🤖 *AI Fashion Stylist Bot Help*

*How to use:*
1. Describe any outfit style you want
2. Use /settings to set gender and item count
3. Use /presets for quick options

*Example prompts:*
• "casual Friday office look"
• "elegant dinner date outfit"
• "comfortable travel clothes"
• "sporty weekend style"

*Commands:*
• /start - Welcome message
• /settings - Change preferences
• /presets - Quick outfit styles
• /help - This help message

*Features:*
✅ AI-powered outfit generation
✅ Gender-specific recommendations
✅ Customizable item count (3-6 items)
✅ Reference images
✅ Quick presets

Need more help? Just ask! 💬
    """
    
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    """Handle errors"""
    logger.error(msg="Exception while handling an update:", exc_info=context.error)

def main():
    """Start the bot"""
    # Create the Application
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("settings", settings))
    application.add_handler(CommandHandler("presets", presets))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Add error handler
    application.add_error_handler(error_handler)
    
    # Start the bot
    print("🤖 Fashion Stylist Bot is starting...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == '__main__':
    main()