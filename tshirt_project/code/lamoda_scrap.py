import requests
from bs4 import BeautifulSoup
import csv
import os
import time
import random
import re
from urllib.parse import urljoin

def get_user_agent():
    """Return a random user agent string to avoid detection."""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    ]
    return random.choice(user_agents)

def download_image(url, folder_path, filename):
    """Download an image from URL and save it to disk."""
    try:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        # Fix for protocol-relative URLs (starting with //)
        if url.startswith('//'):
            url = 'https:' + url
            
        response = requests.get(url, headers={'User-Agent': get_user_agent()})
        if response.status_code == 200:
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'wb') as f:
                f.write(response.content)
            return file_path
        else:
            print(f"Failed to download image: {url}, Status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading image {url}: {e}")
        return None

def format_price(price_str):
    """Format price by removing currency sign and non-numeric characters."""
    # Extract only digits and decimal point
    if price_str and price_str != "Unknown":
        # Remove all non-numeric characters except decimal point
        clean_price = re.sub(r'[^\d.]', '', price_str)
        return clean_price
    return price_str

def scrape_lamoda(url, num_pages=1):
    """Scrape product data from Lamoda website."""
    base_url = "https://www.lamoda.kz"
    products = []
    
    # Create a folder to store images
    image_folder = "lamoda_images"
    if not os.path.exists(image_folder):
        os.makedirs(image_folder)
    
    for page in range(1, num_pages + 1):
        page_url = f"{url}?page={page}"
        print(f"Scraping page {page}: {page_url}")
        
        try:
            # Add a delay between requests to avoid being blocked
            time.sleep(random.uniform(1, 3))
            
            headers = {
                'User-Agent': get_user_agent(),
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Referer': 'https://www.lamoda.kz/'
            }
            
            response = requests.get(page_url, headers=headers)
            if response.status_code != 200:
                print(f"Failed to retrieve page {page}: Status code {response.status_code}")
                continue
                
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Find product containers
            product_cards = soup.select("div.x-product-card__card")
            
            if not product_cards:
                print(f"No product cards found on page {page}. The selector might need updating.")
                continue
                
            print(f"Found {len(product_cards)} products on page {page}")
            
            for card in product_cards:
                try:
                    # Extract product data
                    product = {}
                    
                    # Product name
                    name_elem = card.select_one("div.x-product-card-description__product-name")
                    product['name'] = name_elem.text.strip() if name_elem else "Unknown"
                    
                    # Brand name
                    brand_elem = card.select_one("div.x-product-card-description__brand-name")
                    product['brand'] = brand_elem.text.strip() if brand_elem else "Unknown"
                    
                    # Price - format to remove currency sign
                    price_elem = card.select_one("span.x-product-card-description__price-single")
                    raw_price = price_elem.text.strip() if price_elem else "Unknown"
                    product['price'] = format_price(raw_price)
                    
                    # Sale price if available - format to remove currency sign
                    sale_price_elem = card.select_one("span.x-product-card-description__price-new")
                    if sale_price_elem:
                        raw_sale_price = sale_price_elem.text.strip()
                        product['sale_price'] = format_price(raw_sale_price)
                        
                        # If there's a sale price, the original price is usually shown differently
                        original_price_elem = card.select_one("span.x-product-card-description__price-old")
                        if original_price_elem:
                            raw_original_price = original_price_elem.text.strip()
                            product['price'] = format_price(raw_original_price)
                    else:
                        product['sale_price'] = ""
                    
                    # Product URL
                    link_elem = card.select_one("a.x-product-card__link")
                    if link_elem and 'href' in link_elem.attrs:
                        product['url'] = urljoin(base_url, link_elem['href'])
                    else:
                        product['url'] = ""
                    
                    # Product image URL
                    img_elem = card.select_one("img.x-product-card__pic-img")
                    if img_elem and 'src' in img_elem.attrs:
                        product['image_url'] = img_elem['src']
                        # Fix protocol-relative URLs here too
                        if product['image_url'].startswith('//'):
                            product['image_url'] = 'https:' + product['image_url']
                        
                        # Download and save the image
                        if product['image_url']:
                            # Create a unique filename for the image
                            image_filename = f"{product['brand']}_{product['name'].replace(' ', '_')}_{random.randint(1000, 9999)}.jpg"
                            image_filename = "".join(c for c in image_filename if c.isalnum() or c in ['_', '.'])  # Clean filename
                            
                            # Download the image
                            local_image_path = download_image(product['image_url'], image_folder, image_filename)
                            product['local_image_path'] = local_image_path if local_image_path else ""
                    else:
                        product['image_url'] = ""
                        product['local_image_path'] = ""
                        
                    products.append(product)
                except Exception as e:
                    print(f"Error processing a product: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error scraping page {page}: {e}")
            continue
            
    return products

def save_to_csv(products, filename="lamoda_products.csv"):
    """Save product data to a CSV file."""
    if not products:
        print("No products to save.")
        return
        
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['name', 'brand', 'price', 'sale_price', 'url', 'image_url', 'local_image_path']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for product in products:
                writer.writerow(product)
                
        print(f"Data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving to CSV: {e}")

def main():
    """Main function to run the scraper."""
    # Configuration variables
    url = "https://www.lamoda.kz/c/2508/clothes-tolstovki-i-olimpiyki/"
    num_pages = 100  # Set the number of pages to scrape directly in the script
    
    print(f"Starting to scrape {num_pages} page(s) from Lamoda...")
    products = scrape_lamoda(url, num_pages)
    
    if products:
        print(f"Successfully scraped {len(products)} products.")
        save_to_csv(products)
    else:
        print("No products were scraped.")

if __name__ == "__main__":
    main()