import json
import re
from typing import List, Dict, Any
from processing.chunking import chunk_text
from loaders.pdf_loaders import extract_text_from_pdf

def preprocess_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for doc in documents:
        # Extract text from PDF
        pdf_text = extract_text_from_pdf(doc["file_path"])
        doc["content"] = pdf_text

        # Split text into chunks
        text_chunks = chunk_text(pdf_text, chunk_size=500, chunk_overlap=200)
        doc["text_chunks"] = text_chunks
    return documents

def group_chunks_by_section(chunks: List[str], section_patterns: List[Dict]) -> List[Dict]:
    sections = []
    current_section = None

    for chunk in chunks:
        matched = False
        for pattern in section_patterns:
            if re.search(pattern["start_pattern"], chunk, re.IGNORECASE):
                # Start of a new section
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "title": pattern["title"],
                    "content": chunk,
                    "matched_keywords": [kw for kw in pattern["keywords"] if kw.lower() in chunk.lower()]
                }
                matched = True
                break
        if not matched and current_section:
            current_section["content"] += "\n" + chunk

    if current_section:
        sections.append(current_section)

    return sections

def process_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    section_patterns = [
        {
            "title": "Introduction",
            "start_pattern": r"Iran's direct military response to Israel's attack",
            "keywords": ["Iran", "Israel", "IRGC", "Damascus", "escalation"]
        },
        {
            "title": "Assessing the Iranian Strikes Against Israel",
            "start_pattern": r"The Iranian attack on Israel has sparked a divide",
            "keywords": ["Iranian attack", "specialists", "strategic experts", "drones", "missiles"]
        },
        {
            "title": "Limited and Symbolic Iranian Attack",
            "start_pattern": r"The Iranian response attack did not achieve the same momentum",
            "keywords": ["surprise", "secrecy", "military objectives", "Gaza envelope"]
        },
        {
            "title": "The Attack's Implications for Iran",
            "start_pattern": r"In recent years, Iran has faced significant strikes",
            "keywords": ["Iranian establishment", "prestige", "legitimacy", "Shiite community"]
        },
        {
            "title": "The Attack's Implications for the United States",
            "start_pattern": r"These volatile developments in the Middle East coincide with the US elections",
            "keywords": ["US elections", "Biden", "Trump", "regional conflict"]
        },
        {
            "title": "The Attack's Implications for Israel",
            "start_pattern": r"Israel, particularly Prime Minister Netanyahu, finds itself deeply affected",
            "keywords": ["Netanyahu", "Israeli society", "security", "opposition"]
        },
        {
            "title": "Potential Scenarios",
            "start_pattern": r"The unprecedented Iranian attack on targets within Israel",
            "keywords": ["scenarios", "engagement", "confrontation", "escalation"]
        }
    ]

    processed_docs = []

    for doc in documents:
        filtered_chunks = filter_relevant_text(doc["text_chunks"])
        cleaned_chunks = [clean_text(chunk) for chunk in filtered_chunks]

        sections = group_chunks_by_section(cleaned_chunks, section_patterns)
        if sections:
            processed_docs.append({"filename": doc.get("filename", "unknown"), "sections": sections})

    return processed_docs


def clean_text(text: str) -> str:

    text = re.sub(r'www\.Rasanah-iiis\.org', '', text)
    text = re.sub(r'info@rasanahiiis\.com', '', text)
    text = re.sub(r'\+966112166696', '', text)
    text = re.sub(r'@rasanahiiis', '', text)
    text = re.sub(r'April 16, 2024', '', text)
    text = re.sub(r'Position Paper', '', text)
    text = re.sub(r'Rasanahiiis', '', text)
    text = re.sub(r'\d+\s*$', '', text)  # Remove trailing page numbers
    text = re.sub(r'\s*Contents\s*', '', text, flags=re.IGNORECASE)  # Remove "Contents" header
    # Remove any URLs
    text = re.sub(r'http[s]?://\S+', '', text)
    # Remove any email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    # Remove any special characters except for basic punctuation
    text = re.sub(r'[^\w\s.,;:!?\'\"-]', ' ', text)
    text = re.sub(r'w w w . R a s a n a h - i i i s . o r g', '', text)
    text = re.sub(r'i n f o @ r a s a n a h i i i s . c o m', '', text)
    text = re.sub(r'\+ 9 6 6 1 1 2 1 6 6 6 9 6', '', text)
    text = re.sub(r'@ r a s a n a h i i i s', '', text)
    text = re.sub(r'A p r i l 1 6 , 20 2 4', '', text)
    text = re.sub(r'P o s i t i o n P a p e r', '', text)
    text = re.sub(r'R a s a n a h i i i s', '', text)   
    text = re.sub(r'Iran’s Attack on Israel: Assessment, Repercussions and Scenarios', '', text)

    # Remove any leading/trailing whitespace
    text = text.strip()

    # Remove any non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.strip()

def is_header_or_footer(text: str) -> bool:
    """
    Identify headers, footers, and metadata lines to exclude.
    """
    # Common header/footer patterns
    header_footer_patterns = [
        r'^Contents\s*$',
        r'^Iran\'s Attack on Israel: Assessment, Repercussions and Scenarios\s*$',
        r'^\d+\s*$',  # Just page numbers
        r'^www\.Rasanah-iiis\.org\s*$',
        r'^info@rasanahiiis\.com\s*$',
        r'^\+966112166696\s*$',
        r'^@rasanahiiis\s*$',
        r'^April 16, 2024\s*$',
        r'^Position Paper\s*$',
        r'^Rasanahiiis\s*$',
    ]
    
    for pattern in header_footer_patterns:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    
    return False

def filter_relevant_text(text_chunks):
    filtered = []
    for text in text_chunks:
        if not is_header_or_footer(text):
            filtered.append(text)
    return filtered


# def process_documents(documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     processed_docs = []
#     for doc in documents:
#         # Filter and clean each chunk
#         filtered_content = filter_relevant_text(doc["text_chunks"])
#         cleaned_chunks = [clean_text(chunk) for chunk in filtered_content]

#         if cleaned_chunks:
#             doc["cleaned_chunks"] = cleaned_chunks
#             processed_docs.append(doc)

#     return processed_docs



if __name__ == "__main__":

    document = {"file_path": "docs/Iranian attack on Israel.pdf"}
    preprocessed_docs = preprocess_documents([document])

    # Process the documents
    processed_docs = process_documents(preprocessed_docs)

    # Save the processed documents to a JSON file
    with open("docs/processed_sections.json", "w", encoding="utf-8") as f:
        json.dump(processed_docs, f, ensure_ascii=False, indent=4)
    print("Processed documents saved to 'processed_sections.json'")
    print("Processing complete.")
