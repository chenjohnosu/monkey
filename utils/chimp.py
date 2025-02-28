import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import argparse
import re


def clean_text(text):
    # """Clean up extracted text from a PDF."""
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,;!?]', '', text)
    return text.strip()


def pdf_to_text(pdf_path):
    pdf_document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(pdf_document)):
        print(".", end="")
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += clean_text(pytesseract.image_to_string(img))
    return text


def convert_pdfs_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(directory, filename)
            if not filename.lower().endswith('.txt'):
                if filename.lower().endswith('.pdf'):
                    print("\nConverting " + str(file_path), end="")
                    text = pdf_to_text(file_path)

                    root, ext = os.path.splitext(file_path)
                    output_path = root + '.txt'

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(text)


# Example usage
parser = argparse.ArgumentParser(prog="chimp", description='Usage for monkey helper app to convert image PDFs.')
parser.add_argument('-d', '--dir', type=str, help='Directory of documents: (optional, default: src)')
args = parser.parse_args()
if args.dir:
    convert_pdfs_in_directory(args.dir)
