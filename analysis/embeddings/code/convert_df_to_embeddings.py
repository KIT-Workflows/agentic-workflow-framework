#!/usr/bin/env python3
import json
import time
import re
import unicodedata
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from openai import OpenAI, RateLimitError, APIError, APIConnectionError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Constants
MAX_TEXT_LENGTH = 8000  # Maximum text length for API (tokens are roughly ~4 chars)
EMBEDDING_MODEL = "text-embedding-3-large"
BATCH_SIZE = 16  # Process in batches to avoid rate limits
OUTPUT_FILE = "295-openai-emb.json"

def normalize_text(text: str) -> Optional[str]:
    """
    Normalize and clean text:
    - Convert to string if not already
    - Check if empty or None
    - Remove non-ASCII characters
    - Normalize whitespace
    - Truncate if too long
    """
    # Handle None or non-string types
    if text is None:
        return None
    
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
    
    # Remove excessive whitespace, newlines, tabs
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Check if empty after normalization
    if not text:
        return None
    
    # Truncate if too long
    if len(text) > MAX_TEXT_LENGTH:
        print(f"Warning: Text truncated from {len(text)} to {MAX_TEXT_LENGTH} characters")
        text = text[:MAX_TEXT_LENGTH]
    
    return text

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception_type((RateLimitError, APIError, APIConnectionError))
)
def get_embeddings_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Get embeddings for a batch of texts with retry logic"""
    try:
        response = client.embeddings.create(
            input=texts,
            model=EMBEDDING_MODEL
        )
        
        # Format results
        results = []
        for i, embedding_data in enumerate(response.data):
            results.append({
                "text": texts[i],
                "data": [{"embedding": embedding_data.embedding}]
            })
        
        return results
    
    except Exception as e:
        print(f"Error in embedding API call: {str(e)}")
        raise  # Re-raise for retry mechanism

def main():
    """Process the list of 295 texts, generate embeddings, and save to JSON"""
    try:
        # Step 1: Load the DataFrame
        print("Loading data...")
        
        try:
            # Try to load from JSON file first (295_inputprompt-difficulty.json)
            input_file = "295_inputprompt-difficulty.json"
            if os.path.exists(input_file):
                with open(input_file, 'r') as f:
                    data = json.load(f)
                    # Assume it's a list of objects with a 'calculation_description' field
                    if isinstance(data, list) and 'calculation_description' in data[0]:
                        texts = [item.get('calculation_description', '') for item in data]
                    else:
                        # Try as a dict with a items/results field
                        texts = data.get('items', data.get('results', []))
                        if not texts:
                            raise ValueError("Could not find texts in JSON structure")
            else:
                # If JSON doesn't exist, try to load from a DataFrame
                # This section needs to be customized based on your actual data source
                df = pd.read_csv("your_data_file.csv")  # Replace with your file
                texts = df["calculation_description"].to_list()
        
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            print("Using example data for demonstration")
            # Generate example data for testing - replace with your actual 295 items
            texts = [f"Sample calculation description {i}" for i in range(295)]

        # Step 2: Normalize the texts
        print(f"Normalizing {len(texts)} texts...")
        normalized_texts = []
        skipped_indices = []
        
        for i, text in enumerate(texts):
            normalized = normalize_text(text)
            if normalized:
                normalized_texts.append(normalized)
            else:
                skipped_indices.append(i)
                print(f"Warning: Text at index {i} was skipped due to normalization issues")
        
        print(f"Normalized {len(normalized_texts)} texts. Skipped {len(skipped_indices)} texts.")
        
        # Step 3: Get embeddings in batches
        all_embeddings = []
        total_batches = (len(normalized_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        print(f"Getting embeddings for {len(normalized_texts)} texts in {total_batches} batches...")
        
        for i in tqdm(range(0, len(normalized_texts), BATCH_SIZE)):
            batch_texts = normalized_texts[i:i+BATCH_SIZE]
            
            try:
                batch_embeddings = get_embeddings_batch(batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Add a small delay to respect rate limits
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Failed to process batch starting at index {i}: {str(e)}")
                # Continue with next batch instead of stopping everything
        
        # Step 4: Save to JSON file
        if all_embeddings:
            # Create backup of existing file if it exists
            output_path = Path(OUTPUT_FILE)
            if output_path.exists():
                backup_file = f"{OUTPUT_FILE}.backup.{int(time.time())}"
                output_path.rename(backup_file)
                print(f"Created backup of existing file: {backup_file}")
            
            # Write to temporary file first
            temp_file = f"{OUTPUT_FILE}.temp"
            with open(temp_file, 'w') as f:
                json.dump(all_embeddings, f, ensure_ascii=False, indent=2)
            
            # Then rename to final filename
            os.replace(temp_file, OUTPUT_FILE)
            
            print(f"Successfully saved {len(all_embeddings)} embeddings to {OUTPUT_FILE}")
        else:
            print("No embeddings were generated.")
    
    except Exception as e:
        print(f"Error in main function: {str(e)}")

if __name__ == "__main__":
    main() 