import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loaders.pdf_loaders import extract_text_from_pdf


file_path = "docs\\Iranian attack on Israel.pdf"

text = extract_text_from_pdf(file_path)
print(text[:1000]) 