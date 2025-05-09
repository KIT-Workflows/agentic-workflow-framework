#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
import time
from typing import List, Dict, Tuple, Any
import os
from scipy.spatial.distance import cosine
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns

def load_embeddings(filepath='295-openai-emb.json'):
    """Load embeddings from JSON file and convert to numpy array"""
    print(f"Loading embeddings from {filepath}...")
    start_time = time.time()
    
    with open(filepath, 'r') as f:
        embeddings_data = json.load(f)
    
    # Handle different possible formats based on structure
    if isinstance(embeddings_data, list) and 'data' in embeddings_data[0]:
        # Format: [{"data": [{"embedding": [...]}]}, ...]
        embeddings = [item['data'][0]['embedding'] for item in embeddings_data]
        # If there are texts attached to embeddings, extract them too
        try:
            texts = [item['data'][0]['text'] for item in embeddings_data]
        except KeyError:
            texts = [f"Text {i}" for i in range(len(embeddings))]
    elif isinstance(embeddings_data, list) and isinstance(embeddings_data[0], list):
        # Format: [[0.1, 0.2, ...], [...], ...]
        embeddings = embeddings_data
        texts = [f"Text {i}" for i in range(len(embeddings))]
    else:
        # Try to extract in a different way or raise error
        print(f"Unknown format. First item: {embeddings_data[0]}")
        raise ValueError("Unknown embeddings format")
    
    embeddings_np = np.array(embeddings)
    
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings_np.shape[1]}")
    print(f"Loading took {time.time() - start_time:.2f} seconds")
    print(f"Memory usage: {embeddings_np.nbytes / (1024 * 1024):.2f} MB")
    
    return embeddings_np, texts, embeddings_data

def train_som(embeddings: np.ndarray, som_size: Tuple[int, int] = (20, 20), 
              sigma: float = 1.0, learning_rate: float = 0.5, 
              num_iterations: int = 10000, random_seed: int = 42) -> MiniSom:
    """
    Train a Self-Organizing Map on the embeddings
    
    Args:
        embeddings: Numpy array of shape (n_samples, n_features)
        som_size: Tuple with (width, height) of the SOM grid
        sigma: Initial neighborhood radius
        learning_rate: Initial learning rate
        num_iterations: Number of training iterations
        random_seed: Random seed for reproducibility
        
    Returns:
        Trained SOM model
    """
    print(f"Training SOM with grid size {som_size}...")
    start_time = time.time()
    
    # Get dimensions from data
    n_samples, n_features = embeddings.shape
    
    # Initialize the SOM
    som = MiniSom(som_size[0], som_size[1], n_features, 
                 sigma=sigma, learning_rate=learning_rate, 
                 neighborhood_function='gaussian', 
                 random_seed=random_seed)
    
    # Initialize weights
    som.random_weights_init(embeddings)
    
    # Train the SOM
    som.train(embeddings, num_iterations, verbose=True)
    
    print(f"SOM training completed in {time.time() - start_time:.2f} seconds")
    
    return som

def visualize_som(som: MiniSom, embeddings: np.ndarray, texts: List[str] = None, 
                 output_file: str = 'embeddings_som.png', figsize: Tuple[int, int] = (15, 15)):
    """
    Visualize the SOM by showing:
    1. U-Matrix (distances between neurons)
    2. Sample distribution
    3. Optional: text labels at positions
    
    Args:
        som: Trained SOM model
        embeddings: Numpy array of embeddings
        texts: Optional list of texts corresponding to embeddings
        output_file: Path to save the visualization
        figsize: Size of the figure
    """
    # Get SOM dimensions
    width, height = som.get_weights().shape[0], som.get_weights().shape[1]
    
    # Get the BMU (Best Matching Unit) for each sample
    bmu_indices = np.array([som.winner(x) for x in embeddings])
    
    # Create a grid image of the U-matrix (distances between neurons)
    plt.figure(figsize=figsize)
    
    # 1. Plot the U-matrix (distance between adjacent neurons)
    plt.subplot(1, 2, 1)
    umatrix = som.distance_map()
    plt.imshow(umatrix, cmap='Greys_r')
    plt.colorbar(label='Distance')
    plt.title('U-Matrix (neuron distances)')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # 2. Plot sample distribution (count of samples mapped to each neuron)
    plt.subplot(1, 2, 2)
    
    # Create a heatmap of sample counts
    sample_counts = np.zeros((width, height))
    for x, y in bmu_indices:
        sample_counts[x, y] += 1
    
    plt.imshow(sample_counts, cmap='viridis')
    plt.colorbar(label='Sample count')
    plt.title('Sample Distribution')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    # Optional: add text labels (if not too many)
    if texts and len(texts) <= 50:  # Only show labels if not too crowded
        for i, (x, y) in enumerate(bmu_indices):
            plt.text(y, x, texts[i][:10], ha='center', va='center', 
                     bbox=dict(facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.show()
    
    return bmu_indices, sample_counts

def analyze_som_clusters(som: MiniSom, embeddings: np.ndarray, 
                        bmu_indices: np.ndarray, texts: List[str] = None) -> pd.DataFrame:
    """
    Analyze the clusters formed by the SOM
    
    Args:
        som: Trained SOM model
        embeddings: Numpy array of embeddings
        bmu_indices: BMU indices for each sample
        texts: Optional list of texts corresponding to embeddings
        
    Returns:
        DataFrame with cluster statistics
    """
    # Convert BMU indices to a single cluster ID
    width = som.get_weights().shape[0]
    cluster_ids = [x * width + y for x, y in bmu_indices]
    unique_clusters = set(cluster_ids)
    
    # Prepare cluster statistics
    cluster_stats = []
    for cluster_id in unique_clusters:
        # Find samples in this cluster
        cluster_samples = [i for i, c in enumerate(cluster_ids) if c == cluster_id]
        if not cluster_samples:
            continue
            
        # Get BMU coordinates
        x, y = cluster_id // width, cluster_id % width
        
        # Calculate centroid of cluster samples
        centroid = np.mean([embeddings[i] for i in cluster_samples], axis=0)
        
        # Calculate intra-cluster distance (average distance to centroid)
        distances = [np.linalg.norm(embeddings[i] - centroid) for i in cluster_samples]
        
        # Add to stats
        cluster_stats.append({
            'cluster_id': cluster_id,
            'x': x,
            'y': y,
            'size': len(cluster_samples),
            'avg_distance': np.mean(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'sample_indices': cluster_samples,
            'example_texts': [texts[i][:50] + "..." if texts and len(texts[i]) > 50 else texts[i] 
                             for i in cluster_samples[:3]] if texts else None
        })
    
    # Convert to DataFrame and sort by size
    df = pd.DataFrame(cluster_stats).sort_values('size', ascending=False)
    return df

def main():
    # Load embeddings (use your actual embedding file)
    filepath = '295-openai-emb.json'  # Update with your embedding file
    embeddings_np, texts, _ = load_embeddings(filepath)
    
    # Optional: normalize or scale the embeddings if needed
    # embeddings_np = normalize(embeddings_np)
    
    # Train SOM - adjust parameters based on your data size and compute resources
    # Start with smaller grid for faster testing, increase for better resolution
    som_size = (20, 20)  # Creates a 20x20 grid of neurons
    som = train_som(embeddings_np, som_size=som_size, num_iterations=5000)
    
    # Visualize SOM
    bmu_indices, _ = visualize_som(som, embeddings_np, texts=texts)
    
    # Analyze clusters
    cluster_stats = analyze_som_clusters(som, embeddings_np, bmu_indices, texts=texts)
    print("\nTop 5 clusters by size:")
    print(cluster_stats[['cluster_id', 'x', 'y', 'size', 'avg_distance']].head(5))
    
    # Save the SOM model if needed
    # som.save('text_embeddings_som.p')
    
if __name__ == "__main__":
    main() 