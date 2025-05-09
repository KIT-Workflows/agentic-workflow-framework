#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import pandas as pd
import time
from scipy.spatial.distance import cdist
import seaborn as sns
from collections import Counter

# For more advanced visualization
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("UMAP not available. Install with: pip install umap-learn")

# 1. Load embeddings with metadata extraction
def load_embeddings_with_metadata(filepath='embeddings.json'):
    """Load embeddings and extract metadata if available"""
    print(f"Loading embeddings from {filepath}...")
    start_time = time.time()
    
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    embeddings = []
    metadata = []
    
    # Try to determine format and extract metadata
    if isinstance(data, list):
        if len(data) > 0:
            if isinstance(data[0], dict) and 'data' in data[0]:
                # Format with potential metadata
                for item in data:
                    if 'data' in item and isinstance(item['data'], list) and len(item['data']) > 0:
                        if 'embedding' in item['data'][0]:
                            embeddings.append(item['data'][0]['embedding'])
                            
                            # Extract all fields except the embedding as metadata
                            meta = {k: v for k, v in item.items() if k != 'data'}
                            # Add any other fields from data[0] except embedding
                            for k, v in item['data'][0].items():
                                if k != 'embedding':
                                    meta[k] = v
                            metadata.append(meta)
            elif isinstance(data[0], list):
                # Just a list of embedding vectors
                embeddings = data
                # Create empty metadata
                metadata = [{} for _ in range(len(embeddings))]
    
    embeddings_np = np.array(embeddings)
    
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings_np.shape[1]}")
    print(f"Loading took {time.time() - start_time:.2f} seconds")
    
    # Check if we have meaningful metadata
    if metadata and any(len(m) > 0 for m in metadata):
        print(f"Metadata fields: {list(metadata[0].keys())}")
    else:
        print("No metadata found")
    
    return embeddings_np, metadata

# 2. Find optimal number of clusters
def find_optimal_clusters(embeddings_np, max_clusters=20):
    """Use silhouette score to find optimal number of clusters"""
    if len(embeddings_np) < max_clusters:
        max_clusters = len(embeddings_np) // 2
    
    print("Finding optimal number of clusters...")
    scores = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings_np)
        
        # Only compute if we have more than one cluster
        if len(np.unique(labels)) > 1:
            score = silhouette_score(embeddings_np, labels)
            scores.append((n_clusters, score))
            print(f"  Clusters: {n_clusters}, Silhouette Score: {score:.4f}")
    
    # Find best score
    if scores:
        best_n_clusters = max(scores, key=lambda x: x[1])[0]
        print(f"Optimal number of clusters: {best_n_clusters}")
        return best_n_clusters
    else:
        print("Could not determine optimal clusters")
        return 5  # Default

# 3. Advanced visualization with UMAP
def visualize_umap(embeddings_np, metadata=None, labels=None, n_neighbors=15, min_dist=0.1):
    """Visualize embeddings using UMAP"""
    if not UMAP_AVAILABLE:
        print("UMAP not available. Skipping UMAP visualization.")
        return None
    
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    umap_data = reducer.fit_transform(embeddings_np)
    
    plt.figure(figsize=(12, 10))
    
    # Color by cluster if available
    if labels is not None:
        scatter = plt.scatter(umap_data[:, 0], umap_data[:, 1], c=labels, cmap='tab20', s=5, alpha=0.8)
        plt.colorbar(scatter, label='Cluster')
    else:
        plt.scatter(umap_data[:, 0], umap_data[:, 1], s=5, alpha=0.8)
    
    plt.title(f"UMAP Visualization of Embeddings")
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.tight_layout()
    plt.savefig('embeddings_umap.png', dpi=300)
    plt.show()
    
    return umap_data

# 4. Nearest neighbors for a given embedding
def nearest_neighbors(embeddings_np, query_idx, k=5, distance_metric='cosine'):
    """Find k nearest neighbors for a given embedding"""
    # Get the query vector
    query_vector = embeddings_np[query_idx:query_idx+1]
    
    # Calculate distances from query to all vectors
    distances = cdist(query_vector, embeddings_np, metric=distance_metric)[0]
    
    # Get indices of nearest neighbors (excluding self)
    nearest_indices = np.argsort(distances)[1:k+1] if query_idx < len(embeddings_np) else np.argsort(distances)[:k]
    nearest_distances = distances[nearest_indices]
    
    return list(zip(nearest_indices, nearest_distances))

# 5. Cluster content analysis
def analyze_cluster_contents(embeddings_np, clusters, metadata=None):
    """Analyze what's in each cluster based on metadata"""
    n_clusters = len(np.unique(clusters))
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        print(f"\nCluster {cluster_id}: {len(cluster_indices)} items")
        
        # If we have metadata, analyze patterns
        if metadata and len(metadata) > 0:
            # Collect all metadata keys across all items
            all_keys = set()
            for idx in cluster_indices:
                if idx < len(metadata):
                    all_keys.update(metadata[idx].keys())
            
            # For each key, analyze distribution of values
            for key in all_keys:
                values = []
                for idx in cluster_indices:
                    if idx < len(metadata) and key in metadata[idx]:
                        values.append(str(metadata[idx][key]))
                
                if values:
                    # Count frequency of each value
                    counter = Counter(values)
                    most_common = counter.most_common(3)
                    
                    # Print most common values
                    print(f"  {key}: {most_common}")

# 6. Density-based clustering (DBSCAN)
def cluster_with_dbscan(embeddings_np, eps=0.5, min_samples=5):
    """Cluster embeddings using DBSCAN algorithm"""
    print("\nRunning DBSCAN clustering...")
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(embeddings_np)
    
    # Count number of clusters and noise points
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"DBSCAN found {n_clusters} clusters and {n_noise} noise points")
    
    # Count items per cluster
    for i in range(-1, max(labels) + 1):
        count = list(labels).count(i)
        if i == -1:
            print(f"  Noise points: {count}")
        else:
            print(f"  Cluster {i}: {count} items")
    
    return labels

# 7. Vector norm analysis
def analyze_vector_norms(embeddings_np):
    """Analyze the distribution of vector norms"""
    norms = np.linalg.norm(embeddings_np, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.hist(norms, bins=50, alpha=0.75)
    plt.title('Distribution of Vector Norms')
    plt.xlabel('Vector Norm')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(norms), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(norms):.4f}')
    plt.legend()
    plt.savefig('vector_norms.png', dpi=300)
    plt.show()
    
    return norms

# 8. Advanced pairwise distance analysis
def pairwise_distance_analysis(embeddings_np, sample_size=1000):
    """Analyze pairwise distances between embeddings"""
    # Take a sample if there are too many embeddings
    if len(embeddings_np) > sample_size:
        indices = np.random.choice(len(embeddings_np), sample_size, replace=False)
        sample = embeddings_np[indices]
    else:
        sample = embeddings_np
    
    # Compute pairwise distances
    distances = cdist(sample, sample, 'cosine')
    
    # Plot distance distribution
    plt.figure(figsize=(10, 6))
    
    # Remove self-comparisons
    np.fill_diagonal(distances, np.nan)
    distances_flat = distances.flatten()
    distances_flat = distances_flat[~np.isnan(distances_flat)]
    
    plt.hist(distances_flat, bins=50, alpha=0.75)
    plt.title('Distribution of Pairwise Cosine Distances')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.axvline(np.mean(distances_flat), color='r', linestyle='dashed', linewidth=1, 
                label=f'Mean: {np.mean(distances_flat):.4f}')
    plt.legend()
    plt.savefig('distance_distribution.png', dpi=300)
    plt.show()
    
    print(f"Mean distance: {np.mean(distances_flat):.4f}")
    print(f"Min distance: {np.min(distances_flat):.4f}")
    print(f"Max distance: {np.max(distances_flat):.4f}")
    
    return distances

# Main execution function
def main():
    # Load embeddings with metadata
    embeddings_np, metadata = load_embeddings_with_metadata('embeddings.json')
    
    # Analyze vector norms
    norms = analyze_vector_norms(embeddings_np)
    
    # Find optimal number of clusters
    k = find_optimal_clusters(embeddings_np, max_clusters=15)
    
    # Cluster the data
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings_np)
    
    # Also try DBSCAN
    dbscan_labels = cluster_with_dbscan(embeddings_np, eps=0.5, min_samples=5)
    
    # Visualize with UMAP if available
    if UMAP_AVAILABLE:
        umap_data = visualize_umap(embeddings_np, metadata=metadata, labels=kmeans_labels)
    
    # Analyze cluster contents
    analyze_cluster_contents(embeddings_np, kmeans_labels, metadata)
    
    # Analyze pairwise distances (for a sample)
    pairwise_distance_analysis(embeddings_np, sample_size=1000)
    
    # Nearest neighbors example
    if len(embeddings_np) > 0:
        print("\nExample of nearest neighbors:")
        query_idx = 0  # Use first vector as query
        neighbors = nearest_neighbors(embeddings_np, query_idx, k=5)
        
        for idx, dist in neighbors:
            print(f"Vector {idx}: distance = {dist:.4f}")
            # Print metadata if available
            if metadata and idx < len(metadata) and metadata[idx]:
                for key, value in metadata[idx].items():
                    print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 