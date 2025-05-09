#!/usr/bin/env python3
import json
import numpy as np
from scipy.spatial.distance import cosine
import time
import argparse
from typing import List, Dict, Any, Tuple

class EmbeddingSearcher:
    """Simple vector search engine for embeddings"""
    
    def __init__(self, embeddings_file: str = 'embeddings.json'):
        """Initialize with embeddings file path"""
        self.embeddings_file = embeddings_file
        self.embeddings = []
        self.embeddings_np = None
        self.metadata = []
        self.loaded = False
    
    def load(self) -> bool:
        """Load embeddings from file"""
        print(f"Loading embeddings from {self.embeddings_file}...")
        start_time = time.time()
        
        try:
            with open(self.embeddings_file, 'r') as f:
                data = json.load(f)
            
            # Extract embeddings and metadata based on format
            if isinstance(data, list):
                if len(data) > 0:
                    if isinstance(data[0], dict) and 'data' in data[0]:
                        # Format with potential metadata
                        for item in data:
                            if 'data' in item and isinstance(item['data'], list) and len(item['data']) > 0:
                                if 'embedding' in item['data'][0]:
                                    self.embeddings.append(item['data'][0]['embedding'])
                                    
                                    # Extract metadata
                                    meta = {k: v for k, v in item.items() if k != 'data'}
                                    # Add any other fields from data[0] except embedding
                                    for k, v in item['data'][0].items():
                                        if k != 'embedding':
                                            meta[k] = v
                                    self.metadata.append(meta)
                    elif isinstance(data[0], list):
                        # Just a list of embedding vectors
                        self.embeddings = data
                        # Create empty metadata
                        self.metadata = [{} for _ in range(len(self.embeddings))]
            
            self.embeddings_np = np.array(self.embeddings)
            self.loaded = True
            
            print(f"Loaded {len(self.embeddings)} embeddings with dimension {self.embeddings_np.shape[1]}")
            print(f"Loading took {time.time() - start_time:.2f} seconds")
            
            return True
            
        except Exception as e:
            print(f"Error loading embeddings: {str(e)}")
            return False
    
    def search_by_index(self, index: int, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Search for similar embeddings by index"""
        if not self.loaded:
            if not self.load():
                return []
                
        if index < 0 or index >= len(self.embeddings):
            print(f"Error: Index {index} out of range (0-{len(self.embeddings)-1})")
            return []
        
        return self._similarity_search(self.embeddings_np[index], top_k)
    
    def search_by_text(self, text: str, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Search for embeddings by text (requires OpenAI API)
        
        This is a placeholder function. In a real application, you would:
        1. Generate an embedding for the text
        2. Use that embedding to search
        """
        print("Text search not implemented - requires API integration")
        print("To implement, you would need to:")
        print("1. Call an embedding API (OpenAI, etc.)")
        print("2. Convert the text to a vector")
        print("3. Search using that vector")
        return []
    
    def search_by_vector(self, vector: List[float], top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Search for similar embeddings using a raw vector"""
        if not self.loaded:
            if not self.load():
                return []
                
        # Convert to numpy array
        vector_np = np.array(vector)
        
        # Check if dimensions match
        if vector_np.shape[0] != self.embeddings_np.shape[1]:
            print(f"Error: Vector has {vector_np.shape[0]} dimensions, but embeddings have {self.embeddings_np.shape[1]}")
            return []
            
        return self._similarity_search(vector_np, top_k)
    
    def _similarity_search(self, query_vector: np.ndarray, top_k: int = 5) -> List[Tuple[int, float, Dict]]:
        """Core similarity search function"""
        start_time = time.time()
        
        # Calculate cosine similarity (1 - cosine distance)
        # More efficient than looping through each vector
        distances = np.array([1 - cosine(query_vector, emb) for emb in self.embeddings_np])
        
        # Get top K indices
        top_indices = np.argsort(distances)[-top_k:][::-1]
        
        # Format results
        results = []
        for idx in top_indices:
            results.append((
                int(idx),  # index
                float(distances[idx]),  # similarity score
                self.metadata[idx] if idx < len(self.metadata) else {}  # metadata
            ))
            
        print(f"Search completed in {time.time() - start_time:.4f} seconds")
        return results
    
    def get_metadata(self, index: int) -> Dict:
        """Get metadata for a specific embedding"""
        if not self.loaded:
            if not self.load():
                return {}
                
        if index < 0 or index >= len(self.metadata):
            return {}
            
        return self.metadata[index]

def print_results(results: List[Tuple[int, float, Dict]]):
    """Print search results in a readable format"""
    print("\nSearch Results:")
    print("-" * 50)
    
    for idx, score, meta in results:
        print(f"Index: {idx}, Similarity: {score:.4f}")
        
        # Print metadata if it exists
        if meta:
            print("Metadata:")
            for key, value in meta.items():
                # Print short value directly, truncate long values
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value[:100]}...")
                else:
                    print(f"  {key}: {value}")
        
        print("-" * 50)

def main():
    """Command line interface for embedding search"""
    parser = argparse.ArgumentParser(description='Search in embedding vectors')
    parser.add_argument('--file', '-f', type=str, default='embeddings.json',
                        help='Path to embeddings JSON file')
    parser.add_argument('--index', '-i', type=int, default=None, 
                        help='Index of query vector')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='Number of results to return')
    
    args = parser.parse_args()
    
    searcher = EmbeddingSearcher(args.file)
    
    # Load embeddings
    if not searcher.load():
        return
    
    # If index is provided, search by that index
    if args.index is not None:
        results = searcher.search_by_index(args.index, args.top_k)
        print_results(results)
    else:
        # Interactive mode
        while True:
            try:
                print("\nEmbedding Search CLI")
                print("1. Search by index")
                print("2. Show random example")
                print("3. Exit")
                
                choice = input("Enter your choice (1-3): ").strip()
                
                if choice == '1':
                    index = int(input("Enter vector index: "))
                    top_k = int(input("Number of results (default 5): ") or "5")
                    results = searcher.search_by_index(index, top_k)
                    print_results(results)
                    
                elif choice == '2':
                    # Show a random example
                    import random
                    random_idx = random.randint(0, len(searcher.embeddings) - 1)
                    print(f"Randomly selected vector index: {random_idx}")
                    results = searcher.search_by_index(random_idx, args.top_k)
                    print_results(results)
                    
                elif choice == '3':
                    break
                    
                else:
                    print("Invalid choice. Please try again.")
                    
            except ValueError as e:
                print(f"Error: {str(e)}")
            except KeyboardInterrupt:
                break
    
    print("Goodbye!")

if __name__ == "__main__":
    main() 