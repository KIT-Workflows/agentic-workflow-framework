#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from som_from_scratch import SelfOrganizingMap

# Define a 3D sinusoidal function
def sinusoidal_3d(x, y, z):
    return np.sin(x) * np.cos(y) + np.sin(z)

# Generate 3D data points by sampling the function
def generate_data(n_samples=1000):
    # Sample points in 3D space
    x = np.random.uniform(-np.pi, np.pi, n_samples)
    y = np.random.uniform(-np.pi, np.pi, n_samples)
    z = np.random.uniform(-np.pi, np.pi, n_samples)
    
    # Calculate function values
    values = sinusoidal_3d(x, y, z)
    
    # Create input vectors [x, y, z]
    input_vectors = np.column_stack((x, y, z))
    
    return input_vectors, values

def visualize_original_data(data, values):
    """Visualize the original 3D data with color based on function value"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    scatter = ax.scatter(data[:, 0], data[:, 1], data[:, 2], 
               c=values, cmap='rainbow', alpha=0.8)
    
    plt.colorbar(scatter, label='f(x,y,z) value')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Original 3D Sinusoidal Function')
    
    plt.savefig('original_3d_data.png', dpi=300)
    plt.show()

def visualize_som_mapping(som, data, values):
    """Visualize how the SOM maps the 3D data to a 2D grid"""
    # Get BMUs for all data points
    bmu_coords = som.predict(data)
    
    # Create a grid to store the function values
    value_grid = np.zeros((som.width, som.height))
    count_grid = np.zeros((som.width, som.height))
    
    # Assign average function values to each neuron
    for i, (x, y) in enumerate(bmu_coords):
        value_grid[x, y] += values[i]
        count_grid[x, y] += 1
    
    # Average the values (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_grid = np.divide(value_grid, count_grid)
        avg_grid = np.nan_to_num(avg_grid)
    
    # Visualize the mapping
    plt.figure(figsize=(10, 8))
    plt.imshow(avg_grid, cmap='rainbow')
    plt.colorbar(label='Average f(x,y,z) value')
    plt.title('SOM Mapping of 3D Function to 2D Grid')
    plt.xlabel('X')
    plt.ylabel('Y')
    
    plt.savefig('som_mapping_3d.png', dpi=300)
    plt.show()
    
    # Visualize using a hexagonal grid
    som.plot_umatrix_hex(output_file='som_3d_umatrix_hex.png', colormap='rainbow')
    
    return avg_grid

def visualize_som_components(som):
    """Visualize the component planes showing how x, y, and z are organized"""
    som.plot_component_planes(
        component_indices=[0, 1, 2],
        component_names=['X Component', 'Y Component', 'Z Component'],
        output_file='som_3d_components.png',
        colormap='rainbow'
    )

def main():
    # Generate 3D data
    print("Generating 3D sinusoidal data...")
    data, values = generate_data(n_samples=2000)
    
    # Visualize original data
    visualize_original_data(data, values)
    
    # Create and train SOM
    print("Creating and training SOM...")
    som = SelfOrganizingMap(
        width=10, 
        height=10, 
        input_dim=3,  # 3D data
        learning_rate=0.5, 
        sigma=None,
        random_seed=42
    )
    
    # Train the SOM
    som.train(
        data=data, 
        num_iterations=5000, 
        batch_size=50, 
        verbose=True,
        verbose_interval=100
    )
    
    # Visualize the mapping
    print("Visualizing SOM mapping...")
    avg_grid = visualize_som_mapping(som, data, values)
    
    # Visualize component planes
    print("Visualizing component planes...")
    visualize_som_components(som)
    
    # Print training history
    som.plot_training_history(output_file='som_3d_training_history.png')
    
    print("All visualizations have been saved.")
    
    # Analyze how well the SOM preserves the topology
    print("\nAnalyzing SOM topology preservation:")
    
    # Calculate mean quantization error
    distances = []
    for i, x in enumerate(data):
        bmu = som.find_bmu(x)
        bmu_weight = som.weights[bmu[0], bmu[1]]
        distance = np.linalg.norm(x - bmu_weight)
        distances.append(distance)
    
    mean_error = np.mean(distances)
    print(f"Mean quantization error: {mean_error:.4f}")
    
    # Calculate neuron utilization
    bmu_coords = som.predict(data)
    unique_neurons = set([(x, y) for x, y in bmu_coords])
    utilization = len(unique_neurons) / (som.width * som.height) * 100
    print(f"Neuron utilization: {len(unique_neurons)}/{som.width * som.height} ({utilization:.1f}%)")

if __name__ == "__main__":
    main() 