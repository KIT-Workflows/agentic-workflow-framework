#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
import pandas as pd
from typing import List, Dict, Any
import os
import time
from scipy.spatial.distance import cosine

# 1. Load and inspect embeddings
def load_embeddings(filepath='embeddings.json'):
    """Load embeddings from JSON file and convert to numpy array"""
    print(f"Loading embeddings from {filepath}...")
    start_time = time.time()
    
    with open(filepath, 'r') as f:
        embeddings_data = json.load(f)
    
    # Handle different possible formats based on what we saw in the codebase
    if isinstance(embeddings_data, list) and 'data' in embeddings_data[0]:
        # Format: [{"data": [{"embedding": [...]}]}, ...]
        embeddings = [item['data'][0]['embedding'] for item in embeddings_data]
    elif isinstance(embeddings_data, list) and isinstance(embeddings_data[0], list):
        # Format: [[0.1, 0.2, ...], [...], ...]
        embeddings = embeddings_data
    else:
        # Try to extract in a different way or raise error
        print(f"Unknown format. First item: {embeddings_data[0]}")
        raise ValueError("Unknown embeddings format")
    
    embeddings_np = np.array(embeddings)
    
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings_np.shape[1]}")
    print(f"Loading took {time.time() - start_time:.2f} seconds")
    print(f"Memory usage: {embeddings_np.nbytes / (1024 * 1024):.2f} MB")
    
    return embeddings_np, embeddings_data

# 2. Basic statistics and exploration
def analyze_embeddings(embeddings_np):
    """Compute basic statistics about the embeddings"""
    print(f"Embeddings shape: {embeddings_np.shape}")
    print(f"Mean values: {np.mean(embeddings_np, axis=0)[:5]}... (first 5)")
    print(f"Min value: {np.min(embeddings_np)}")
    print(f"Max value: {np.max(embeddings_np)}")
    print(f"Standard deviation: {np.std(embeddings_np)}")
    
    # Calculate vector norms
    norms = np.linalg.norm(embeddings_np, axis=1)
    print(f"Mean vector norm: {np.mean(norms)}")
    print(f"Min vector norm: {np.min(norms)}")
    print(f"Max vector norm: {np.max(norms)}")
    
    # Check if vectors are normalized (all have unit norm)
    print(f"Are vectors normalized? {np.allclose(norms, 1.0, atol=1e-5)}")
    
    return norms

# 3. Dimensionality reduction for visualization
def visualize_embeddings(embeddings_np, method='pca', n_clusters=5):
    """Reduce dimensionality and visualize embeddings"""
    # Reduce to 2D
    if method.lower() == 'pca':
        reducer = PCA(n_components=2)
        title = 'PCA Visualization'
    elif method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        title = 't-SNE Visualization'
    else:
        raise ValueError("Method must be 'pca' or 'tsne'")
    
    reduced_data = reducer.fit_transform(embeddings_np)
    
    # Optional: Cluster the data
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_np)
    
    # Plot
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label='Cluster')
    plt.title(f"{title} of Embeddings with {n_clusters} Clusters")
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.tight_layout()
    plt.savefig(f'embeddings_{method.lower()}.png', dpi=300)
    plt.show()
    
    return reduced_data, clusters

# 4. Similarity search
def find_similar_vectors(embeddings_np, query_index, top_k=5):
    """Find vectors most similar to the query vector"""
    query_vector = embeddings_np[query_index]
    
    # Compute cosine similarities
    similarities = []
    for i, vec in enumerate(embeddings_np):
        if i != query_index:  # Skip self-comparison
            sim = 1 - cosine(query_vector, vec)  # Convert distance to similarity
            similarities.append((i, sim))
    
    # Sort by similarity (descending)
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Return top k
    return similarities[:top_k]

# 5. Cluster analysis
def analyze_clusters(embeddings_np, n_clusters=5):
    """Perform k-means clustering and analyze results"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings_np)
    
    # Count samples per cluster
    unique, counts = np.unique(clusters, return_counts=True)
    cluster_counts = dict(zip(unique, counts))
    
    print("Samples per cluster:")
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} samples ({count/len(embeddings_np)*100:.1f}%)")
    
    # Calculate cluster statistics
    cluster_stats = []
    for i in range(n_clusters):
        cluster_vectors = embeddings_np[clusters == i]
        centroid = kmeans.cluster_centers_[i]
        # Calculate average distance to centroid
        distances = [np.linalg.norm(v - centroid) for v in cluster_vectors]
        avg_distance = np.mean(distances) if distances else 0
        
        cluster_stats.append({
            'cluster_id': i,
            'size': len(cluster_vectors),
            'avg_distance_to_centroid': avg_distance,
            'min_distance': min(distances) if distances else 0,
            'max_distance': max(distances) if distances else 0
        })
    
    return clusters, pd.DataFrame(cluster_stats)

# Main analysis function
def main():
    # Load embeddings
    embeddings_np, raw_data = load_embeddings('embeddings.json')
    
    # Basic analysis
    norms = analyze_embeddings(embeddings_np)
    
    # Visualize
    reduced_data, clusters = visualize_embeddings(embeddings_np, method='pca', n_clusters=5)
    
    # Optional: Also try t-SNE
    # reduced_data_tsne, _ = visualize_embeddings(embeddings_np, method='tsne', n_clusters=5)
    
    # Cluster analysis
    _, cluster_stats = analyze_clusters(embeddings_np, n_clusters=5)
    print("\nCluster Statistics:")
    print(cluster_stats)
    
    # Example of similarity search
    if len(embeddings_np) > 0:
        query_idx = 0  # Use first vector as query
        similar_vectors = find_similar_vectors(embeddings_np, query_idx, top_k=5)
        print(f"\nVectors most similar to vector {query_idx}:")
        for idx, similarity in similar_vectors:
            print(f"Vector {idx}: similarity = {similarity:.4f}")

if __name__ == "__main__":
    main() 