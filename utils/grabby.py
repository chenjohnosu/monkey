import requests
from bs4 import BeautifulSoup
import time
import random
import sys

FILE = "urls.urls"

def scrape_website(url, headers, file_index):
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"Failed to retrieve {url}: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text()

    # Save the text to an enumerated file
    filename = f"file_{file_index}.txt"
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(text)

    # Add a random delay between 1 to 5 seconds
    time.sleep(random.uniform(1, 5))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <file_with_urls>")
        sys.exit(1)

    file_with_urls = sys.argv
    print(f"Reading URLs from file: {file_with_urls}")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        with open(FILE, 'r') as file:
            urls = file.readlines()
    except Exception as e:
        print(f"Error opening file: {e}")
        sys.exit(1)

    for index, url in enumerate(urls, start=1):
        url = url.strip()
        if url:
            scrape_website(url, headers, index)