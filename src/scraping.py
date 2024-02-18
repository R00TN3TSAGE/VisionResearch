import requests
import os
import sqlite3
from PIL import Image
from io import BytesIO

# Configuration
BASE_DIR = 'path/to/your/directory' # To edit
DATABASE_PATH = os.path.join(BASE_DIR, 'images.db')
TARGET_SIZE = (224, 224)

# Ensure BASE_DIR exists
os.makedirs(BASE_DIR, exist_ok=True)

# Initialize database
conn = sqlite3.connect(DATABASE_PATH)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS images (id INTEGER PRIMARY KEY, url TEXT, label TEXT)''')
conn.commit()

def fetch_from_google(api_key, cse_id, query, num_images):
    """Fetch image URLs using Google Custom Search JSON API."""
    google_urls = []
    search_url = "https://www.googleapis.com/customsearch/v1"
    params = {
        'q': query,
        'cx': cse_id,
        'key': api_key,
        'searchType': 'image',
        'num': min(num_images, 10)  # API allows max 10 results per query
    }
    response = requests.get(search_url, params=params)
    results = response.json()
    for item in results.get('items', []):
        if 'link' in item:
            google_urls.append(item['link'])
    return google_urls

def fetch_from_flickr(api_key, query, num_images):
    """Fetch image URLs from the Flickr API."""
    flickr_urls = []
    params = {
        'method': 'flickr.photos.search',
        'api_key': api_key,
        'text': query,
        'format': 'json',
        'nojsoncallback': 1,
        'per_page': num_images,
        'media': 'photos',
        'extras': 'url_o',  # Attempt to get the original image URL
    }
    response = requests.get('https://api.flickr.com/services/rest/', params=params)
    data = response.json()
    
    for photo in data['photos']['photo']:
        if 'url_o' in photo:
            flickr_urls.append(photo['url_o'])
    
    return flickr_urls

def fetch_image_urls(api_key, cse_id, query, num_images):
    """Fetch image URLs from multiple platforms."""
    image_urls = []
    # Fetch from Google
    image_urls.extend(fetch_from_google(api_key, cse_id, query, num_images))
    # Fetch from Flickr (assuming same API key, adjust as necessary)
    image_urls.extend(fetch_from_flickr(api_key, query, num_images))
    return image_urls[:num_images]

def download_and_preprocess(url):
    """Download an image from a URL and preprocess it."""
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content))
        image = image.resize(TARGET_SIZE)
        return image
    except Exception as e:
        print(f"Error processing image from {url}: {e}")
        return None

def save_to_database(cursor, url, label):
    """Save image URL and label to the database."""
    cursor.execute('''INSERT INTO images (url, label) VALUES (?, ?)''', (url, label))
    conn.commit()

def main(api_key, cse_id, search_query, num_images, label):
    """Main script logic."""
    image_urls = fetch_image_urls(api_key, cse_id, search_query, num_images)
    for url in image_urls:
        image = download_and_preprocess(url)
        if image:
            image_path = os.path.join(BASE_DIR, f"{label}_{os.path.basename(url)}")
            image.save(image_path)
            save_to_database(cursor, url, label)
            print(f"Saved {url} with label '{label}' to '{image_path}'")
        else:
            print(f"Failed to process image from {url}")

if __name__ == "__main__":
    API_KEY = 'your_api_key_here'
    CSE_ID = 'your_custom_search_engine_id_here'
    search_query = 'example_search_query'
    num_images = 20
    label = 'example_label'
    
    main(API_KEY, CSE_ID, search_query, num_images, label)
    
    conn.close()
