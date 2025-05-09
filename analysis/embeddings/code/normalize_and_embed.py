#!/usr/bin/env python3
import json
import time
import re
import unicodedata
import numpy as np
from typing import List, Dict, Any, Optional
import os
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv, find_dotenv

# Load environment variables
load_dotenv(find_dotenv())
OPENAI_API = os.getenv("OPENAI_API")
if not OPENAI_API:
    raise ValueError("OPENAI_API not found in environment variables")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API)

# Constants
MAX_TEXT_LENGTH = 8000  # Maximum text length to process (OpenAI has token limits)
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 16  # Number of texts to batch in each API call
OUTPUT_FILE = "295-openai-emb.json"

# Retry decorator for API calls
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError))
)
def get_embeddings_batch(texts: List[str], model: str = EMBEDDING_MODEL) -> List[Dict[str, Any]]:
    """Get embeddings for a batch of texts with retry logic"""
    response = client.embeddings.create(
        input=texts,
        model=model
    )
    
    # Process and return results
    results = []
    for i, embedding_data in enumerate(response.data):
        results.append({
            "text": texts[i],
            "data": [{"embedding": embedding_data.embedding}]
        })
    
    return results

def normalize_text(text: str) -> Optional[str]:
    """
    Normalize text with safety checks:
    - Convert to string if not already
    - Remove non-ASCII characters
    - Remove excessive whitespace
    - Check if empty after normalization
    - Truncate if too long
    """
    # Convert to string if not already
    if not isinstance(text, str):
        try:
            text = str(text)
        except:
            return None
    
    # Check if empty
    if not text or text.isspace():
        return None
    
    # Normalize Unicode characters
    text = unicodedata.normalize('NFKD', text)
    
    # Replace non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Check if empty after normalization
    if not text:
        return None
    
    # Truncate if too long
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]
    
    return text

def process_texts_to_embeddings(texts: List[str]) -> List[Dict[str, Any]]:
    """Process a list of texts: normalize each text and generate embeddings"""
    normalized_texts = []
    skipped_indices = []
    
    print("Normalizing texts...")
    for i, text in enumerate(texts):
        normalized = normalize_text(text)
        if normalized:
            normalized_texts.append(normalized)
        else:
            skipped_indices.append(i)
            print(f"Warning: Text at index {i} was skipped due to normalization issues")
    
    print(f"Normalized {len(normalized_texts)} texts. Skipped {len(skipped_indices)} texts.")
    
    # Process in batches to avoid rate limits
    all_embeddings = []
    total_batches = (len(normalized_texts) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"Getting embeddings for {len(normalized_texts)} texts in {total_batches} batches...")
    
    for i in tqdm(range(0, len(normalized_texts), BATCH_SIZE)):
        batch_texts = normalized_texts[i:i+BATCH_SIZE]
        
        try:
            # Get embeddings for this batch
            batch_embeddings = get_embeddings_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
            
            # Sleep briefly to avoid hitting rate limits
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Error in batch starting at index {i}: {str(e)}")
            # Continue with the next batch
    
    return all_embeddings

def save_embeddings(embeddings: List[Dict[str, Any]], output_file: str = OUTPUT_FILE):
    """Save embeddings to a JSON file with error handling"""
    # Create backup of existing file if it exists
    output_path = Path(output_file)
    if output_path.exists():
        backup_file = f"{output_file}.backup.{int(time.time())}"
        output_path.rename(backup_file)
        print(f"Created backup of existing file: {backup_file}")
    
    try:
        # First write to a temporary file
        temp_file = f"{output_file}.temp"
        with open(temp_file, 'w') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
        
        # Then rename to final filename
        os.replace(temp_file, output_file)
        print(f"Successfully saved {len(embeddings)} embeddings to {output_file}")
        
    except Exception as e:
        print(f"Error saving embeddings: {str(e)}")
        if os.path.exists(temp_file):
            print(f"Temporary file exists at: {temp_file}")

def main():
    """Main function to process texts from a file or list"""
    try:
        # Option 1: Load from DataFrame (if you have the data in a DataFrame)
        # df = pd.read_csv("your_data.csv")  # Replace with your file
        # texts = df["calculation_description"].to_list()
        
        # Option 2: Load from JSON file
        # with open("your_data.json", "r") as f:
        #     data = json.load(f)
        #     texts = [item["text"] for item in data]
        
        # Option 3: For testing, create sample data
        # texts = ["Sample text " + str(i) for i in range(295)]
        
        # *** Replace this section with your actual data loading code ***
        print("Please modify the script to load your 295 items.")
        print("Uncomment one of the data loading options above or add your own.")
        print("For now, we'll create a small test dataset.")
        texts = ["Sample text " + str(i) for i in range(10)]  # Just for testing
        # *** End of section to replace ***
        
        # Process texts and get embeddings
        embeddings = process_texts_to_embeddings(texts)
        
        # Save embeddings to file
        save_embeddings(embeddings)
        
    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 