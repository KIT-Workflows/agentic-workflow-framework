#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import pickle
import os
from typing import List, Dict, Tuple, Any, Optional, Callable
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import RegularPolygon
import matplotlib.cm as cm

class SelfOrganizingMap:
    """
    Self-Organizing Map (SOM) implementation from scratch.
    
    This class implements the core SOM algorithm as described by Teuvo Kohonen.
    It creates a 2D grid of neurons where each neuron has a weight vector of the
    same dimensionality as the input data.
    """
    
    def __init__(self, width: int, height: int, input_dim: int, 
                 learning_rate: float = 0.5, sigma: float = None,
                 random_seed: Optional[int] = None):
        """
        Initialize a new Self-Organizing Map.
        
        Args:
            width: Width of the SOM grid (number of neurons in x direction)
            height: Height of the SOM grid (number of neurons in y direction)
            input_dim: Dimensionality of the input vectors
            learning_rate: Initial learning rate (alpha)
            sigma: Initial neighborhood radius (if None, will default to max(width, height)/2)
            random_seed: Seed for reproducibility
        """
        self.width = width
        self.height = height
        self.input_dim = input_dim
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        
        # Set initial neighborhood radius
        if sigma is None:
            self.initial_sigma = max(width, height) / 2.0
        else:
            self.initial_sigma = sigma
        self.sigma = self.initial_sigma
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        self.random_seed = random_seed
        
        # Initialize weights randomly (shape: width x height x input_dim)
        self.weights = np.random.rand(width, height, input_dim)
        
        # Create a grid of 2D coordinates for the neurons
        self.neuron_positions = np.array([[(i, j) for j in range(height)] 
                                         for i in range(width)])
        
        # Training state
        self.current_iter = 0
        self.max_iter = 1
        self.training_history = {
            'learning_rate': [],
            'sigma': [],
            'iteration': []
        }
        
        # Store the distances between neurons for future use
        self._calculate_neuron_distances()
    
    def _calculate_neuron_distances(self):
        """Calculate distances between all neurons in the grid for neighborhood calculations."""
        # Reshape coordinates to a flat list of positions
        positions = self.neuron_positions.reshape(self.width * self.height, 2)
        
        # Calculate pairwise distances between all neuron positions
        # This creates a matrix of shape (width*height, width*height)
        self.neuron_distances = cdist(positions, positions, 'euclidean')
        
        # Reshape to 4D tensor for easier indexing later
        self.neuron_distances = self.neuron_distances.reshape(
            self.width, self.height, self.width, self.height)
    
    def initialize_weights_with_samples(self, samples: np.ndarray):
        """
        Initialize the weights using random samples from the input data.
        
        This can provide better starting points than purely random initialization.
        
        Args:
            samples: Input data to sample from (shape: num_samples x input_dim)
        """
        # Verify input shape
        if samples.shape[1] != self.input_dim:
            raise ValueError(f"Samples must have {self.input_dim} dimensions, got {samples.shape[1]}")
        
        # Randomly sample from the input data
        indices = np.random.choice(samples.shape[0], 
                                  size=self.width * self.height, 
                                  replace=samples.shape[0] < self.width * self.height)
        
        # Assign samples to weights
        self.weights = samples[indices].reshape(self.width, self.height, self.input_dim)
    
    def find_bmu(self, x: np.ndarray) -> Tuple[int, int]:
        """
        Find the Best Matching Unit (BMU) for input vector x.
        
        The BMU is the neuron whose weight vector is closest to the input vector.
        
        Args:
            x: Input vector (shape: input_dim)
            
        Returns:
            Tuple (i, j) with coordinates of the BMU
        """
        # Calculate Euclidean distances between input vector and all weight vectors
        distances = np.linalg.norm(self.weights - x, axis=2)
        
        # Find the coordinates of the minimum distance
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def _decay_parameters(self, t: int, max_iter: int):
        """
        Decay learning rate and neighborhood radius with time.
        
        Args:
            t: Current iteration
            max_iter: Maximum number of iterations
        """
        # Exponential decay for learning rate
        self.learning_rate = self.initial_learning_rate * np.exp(-t / max_iter)
        
        # Exponential decay for neighborhood radius
        self.sigma = self.initial_sigma * np.exp(-t / max_iter)
        
        # Record training history
        self.training_history['learning_rate'].append(self.learning_rate)
        self.training_history['sigma'].append(self.sigma)
        self.training_history['iteration'].append(t)
    
    def _neighborhood_function(self, bmu_pos: Tuple[int, int], neuron_pos: Tuple[int, int]) -> float:
        """
        Calculate the neighborhood influence of the BMU on a particular neuron.
        
        Uses a Gaussian neighborhood function.
        
        Args:
            bmu_pos: (i, j) coordinates of the BMU
            neuron_pos: (i, j) coordinates of the neuron
            
        Returns:
            Neighborhood influence (between 0 and 1)
        """
        # Calculate distance between BMU and the neuron
        distance = self.neuron_distances[bmu_pos[0], bmu_pos[1], 
                                        neuron_pos[0], neuron_pos[1]]
        
        # Calculate Gaussian neighborhood function
        return np.exp(-(distance**2) / (2 * self.sigma**2))
    
    def train_single_step(self, x: np.ndarray, iteration: int, max_iterations: int):
        """
        Perform a single training step for one input vector.
        
        Args:
            x: Input vector (shape: input_dim)
            iteration: Current iteration number
            max_iterations: Maximum number of iterations
        """
        # Update learning parameters based on current iteration
        self._decay_parameters(iteration, max_iterations)
        
        # Find the BMU for this input
        bmu_pos = self.find_bmu(x)
        
        # Update all neurons based on their proximity to the BMU
        for i in range(self.width):
            for j in range(self.height):
                # Calculate the neighborhood influence
                influence = self._neighborhood_function(bmu_pos, (i, j))
                
                # Update weight vector based on influence
                self.weights[i, j] += self.learning_rate * influence * (x - self.weights[i, j])
    
    def train(self, data: np.ndarray, num_iterations: int = 10000, batch_size: int = None, 
              verbose: bool = True, verbose_interval: int = 100,
              continue_training: bool = False):
        """
        Train the SOM with the given data.
        
        Args:
            data: Training data (shape: num_samples x input_dim)
            num_iterations: Number of training iterations
            batch_size: Number of samples to use in each iteration (if None, use all data)
            verbose: Whether to print progress
            verbose_interval: How often to print progress (in iterations)
            continue_training: Whether to continue training from current state
        """
        if not continue_training:
            # Reset training state if not continuing
            self.current_iter = 0
            self.max_iter = num_iterations
        else:
            # If continuing, adjust max_iter and preserve current parameters
            old_max_iter = self.max_iter
            self.max_iter = self.current_iter + num_iterations
            
            # Store current learning rate and sigma as new initials for this training session
            if self.current_iter > 0:
                # Calculate what the parameters should be at current iteration
                current_lr = self.initial_learning_rate * np.exp(-self.current_iter / old_max_iter) 
                current_sigma = self.initial_sigma * np.exp(-self.current_iter / old_max_iter)
                
                # Set these as the new "initial" values for this session
                self.initial_learning_rate = current_lr
                self.initial_sigma = current_sigma
        
        start_time = time.time()
        
        # If batch_size is None, use the entire dataset in each iteration
        if batch_size is None:
            batch_size = data.shape[0]
        
        # Setup progress bar
        pbar = tqdm(total=num_iterations, desc="Training SOM", disable=not verbose)
        pbar.set_postfix({
            'lr': self.learning_rate, 
            'sigma': self.sigma
        })
        
        for t in range(self.current_iter, self.current_iter + num_iterations):
            # Sample a batch of data
            indices = np.random.choice(data.shape[0], size=batch_size, replace=True)
            batch = data[indices]
            
            # Train on each sample in the batch
            for sample in batch:
                self.train_single_step(sample, t, self.max_iter)
            
            # Update progress bar
            if verbose and (t + 1) % verbose_interval == 0:
                pbar.update(verbose_interval)
                pbar.set_postfix({
                    'lr': f"{self.learning_rate:.4f}", 
                    'sigma': f"{self.sigma:.4f}"
                })
        
        # Update current iteration
        self.current_iter += num_iterations
        
        # Close progress bar
        pbar.close()
        
        if verbose:
            print(f"Training completed in {time.time() - start_time:.2f} seconds")
    
    def get_distance_map(self) -> np.ndarray:
        """
        Calculate the U-Matrix (unified distance matrix).
        
        The U-Matrix visualizes distances between adjacent neurons.
        
        Returns:
            2D matrix with average distances to adjacent neurons
        """
        u_matrix = np.zeros((self.width, self.height))
        
        for i in range(self.width):
            for j in range(self.height):
                # Get coordinates of adjacent neurons (with boundary checking)
                neighbors = []
                if i > 0:
                    neighbors.append((i-1, j))  # left
                if i < self.width-1:
                    neighbors.append((i+1, j))  # right
                if j > 0:
                    neighbors.append((i, j-1))  # up
                if j < self.height-1:
                    neighbors.append((i, j+1))  # down
                
                # Add diagonal neighbors for hexagonal grid simulation
                if j % 2 == 0:  # Even rows
                    if i > 0 and j > 0:
                        neighbors.append((i-1, j-1))  # top-left
                    if i > 0 and j < self.height-1:
                        neighbors.append((i-1, j+1))  # bottom-left
                else:  # Odd rows
                    if i < self.width-1 and j > 0:
                        neighbors.append((i+1, j-1))  # top-right
                    if i < self.width-1 and j < self.height-1:
                        neighbors.append((i+1, j+1))  # bottom-right
                
                # Calculate average distance to adjacent neurons
                distances = [np.linalg.norm(self.weights[i, j] - self.weights[x, y]) 
                             for x, y in neighbors]
                u_matrix[i, j] = np.mean(distances) if distances else 0
        
        return u_matrix
    
    def predict(self, data: np.ndarray) -> np.ndarray:
        """
        Map input data to BMU coordinates.
        
        Args:
            data: Input data (shape: num_samples x input_dim)
            
        Returns:
            Array of BMU coordinates for each input (shape: num_samples x 2)
        """
        bmu_coords = np.zeros((data.shape[0], 2), dtype=int)
        
        for i, x in enumerate(data):
            bmu = self.find_bmu(x)
            bmu_coords[i] = bmu
        
        return bmu_coords
    
    def get_weights(self) -> np.ndarray:
        """Get the current weight matrix."""
        return self.weights
    
    def save(self, filepath: str):
        """
        Save the trained SOM model to disk.
        
        Args:
            filepath: Path to save the model to
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """
        Load a saved SOM model from disk.
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded SOM model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        print(f"Model loaded from {filepath}")
        return model
    
    def plot_umatrix_hex(self, output_file: str = 'som_umatrix_hex.png', 
                        figsize: Tuple[int, int] = (10, 8),
                        colormap: str = 'rainbow'):
        """
        Plot the U-Matrix (unified distance matrix) using a hexagonal grid.
        
        Args:
            output_file: Path to save the visualization
            figsize: Size of the figure
            colormap: Colormap to use for visualization
        """
        plt.figure(figsize=figsize)
        
        # Calculate the U-Matrix
        umatrix = self.get_distance_map()
        
        # Determine the coordinates for the hexagons
        # For a hexagonal grid, the y-coordinates of odd rows are offset
        hex_coordinates = []
        for j in range(self.height):
            for i in range(self.width):
                # Offset odd rows to create hexagonal pattern
                x_offset = 0.5 if j % 2 == 1 else 0
                # Use 0.866 instead of 0.75 for perfect hexagonal packing (sqrt(3)/2)
                hex_coordinates.append((i + x_offset, j * 0.866))
        
        # Normalize U-matrix for color mapping
        umatrix_flat = umatrix.flatten()
        norm = plt.Normalize(umatrix_flat.min(), umatrix_flat.max())
        
        # Create hexagonal grid
        ax = plt.gca()
        for idx, (x, y) in enumerate(hex_coordinates):
            i, j = idx // self.height, idx % self.height
            color = plt.cm.get_cmap(colormap)(norm(umatrix_flat[idx]))
            # Increase radius to fully fill the grid without gaps (0.45 -> 0.5)
            hex = RegularPolygon((x, y), numVertices=6, radius=0.5, 
                                 orientation=0, 
                                 facecolor=color, edgecolor='gray', alpha=0.9)
            ax.add_patch(hex)
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Distance')
        
        # Set axis limits and remove axes and labels
        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height * 0.866 + 0.5)
        plt.title('U-Matrix')
        ax.set_aspect('equal')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()
        
        return umatrix
    
    def plot_sample_distribution_hex(self, data: np.ndarray, 
                                   output_file: str = 'som_distribution_hex.png', 
                                   figsize: Tuple[int, int] = (10, 8),
                                   colormap: str = 'viridis'):
        """
        Plot the distribution of samples across the SOM using a hexagonal grid.
        
        Args:
            data: Input data (shape: num_samples x input_dim)
            output_file: Path to save the visualization
            figsize: Size of the figure
            colormap: Colormap to use for visualization
        """
        plt.figure(figsize=figsize)
        
        # Get BMUs for all data points
        bmu_coords = self.predict(data)
        
        # Count occurrences of each BMU
        sample_counts = np.zeros((self.width, self.height))
        for i, j in bmu_coords:
            sample_counts[i, j] += 1
        
        # Determine the coordinates for the hexagons
        hex_coordinates = []
        for j in range(self.height):
            for i in range(self.width):
                # Offset odd rows to create hexagonal pattern
                x_offset = 0.5 if j % 2 == 1 else 0
                # Use 0.866 for perfect hexagonal packing (sqrt(3)/2)
                hex_coordinates.append((i + x_offset, j * 0.866))
        
        # Normalize sample counts for color mapping
        sample_counts_flat = sample_counts.flatten()
        norm = plt.Normalize(sample_counts_flat.min(), sample_counts_flat.max())
        
        # Create hexagonal grid
        ax = plt.gca()
        for idx, (x, y) in enumerate(hex_coordinates):
            i, j = idx // self.height, idx % self.height
            color = plt.cm.get_cmap(colormap)(norm(sample_counts_flat[idx]))
            # Perfect hexagonal packing with radius 0.5
            hex = RegularPolygon((x, y), numVertices=6, radius=0.5, 
                                 orientation=0, 
                                 facecolor=color, edgecolor='gray', alpha=0.9)
            ax.add_patch(hex)
            
            # Add count text for cells with samples
            if sample_counts_flat[idx] > 0:
                plt.text(x, y, int(sample_counts_flat[idx]), 
                         ha='center', va='center', fontsize=8, 
                         color='black' if norm(sample_counts_flat[idx]) > 0.5 else 'white')
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Sample count')
        
        # Set axis limits and remove axes and labels
        ax.set_xlim(-0.5, self.width + 0.5)
        ax.set_ylim(-0.5, self.height * 0.866 + 0.5)
        plt.title('Distribution of Samples')
        ax.set_aspect('equal')
        
        # Remove axes
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()
        
        return sample_counts
    
    def plot_component_planes_hex(self, component_indices: List[int] = None, 
                                component_names: List[str] = None,
                                output_file: str = 'som_components_hex.png',
                                figsize: Tuple[int, int] = None,
                                colormap: str = 'rainbow'):
        """
        Plot component planes using a hexagonal grid.
        
        Args:
            component_indices: Indices of components to plot (if None, plot first min(10, input_dim))
            component_names: Names for the components (if None, use indices)
            output_file: Path to save the visualization
            figsize: Size of the figure
            colormap: Colormap to use for visualization
        """
        # Determine which components to plot
        if component_indices is None:
            component_indices = list(range(min(10, self.input_dim)))
        
        n_components = len(component_indices)
        
        # Set up grid for plotting
        n_cols = min(5, n_components)
        n_rows = int(np.ceil(n_components / n_cols))
        
        if figsize is None:
            figsize = (4 * n_cols, 3 * n_rows)
        
        plt.figure(figsize=figsize)
        
        # Determine the coordinates for the hexagons
        hex_coordinates = []
        for j in range(self.height):
            for i in range(self.width):
                x_offset = 0.5 if j % 2 == 1 else 0
                # Use 0.866 for perfect hexagonal packing (sqrt(3)/2)
                hex_coordinates.append((i + x_offset, j * 0.866))
        
        # Plot each component
        for k, comp_idx in enumerate(component_indices):
            plt.subplot(n_rows, n_cols, k + 1)
            
            # Extract the component values
            component = self.weights[:, :, comp_idx].flatten()
            
            # Normalize for color mapping
            norm = plt.Normalize(component.min(), component.max())
            
            # Create hexagonal grid
            ax = plt.gca()
            for idx, (x, y) in enumerate(hex_coordinates):
                color = plt.cm.get_cmap(colormap)(norm(component[idx]))
                # Increase radius to remove gaps
                hex = RegularPolygon((x, y), numVertices=6, radius=0.5, 
                                    orientation=0, 
                                    facecolor=color, edgecolor='gray', alpha=0.9)
                ax.add_patch(hex)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
            sm.set_array([])
            plt.colorbar(sm, ax=ax)
            
            # Set title and limits
            if component_names is not None and k < len(component_names):
                title = component_names[k]
            else:
                title = f"Component {comp_idx}"
            plt.title(title)
            
            ax.set_xlim(-0.5, self.width + 0.5)
            ax.set_ylim(-0.5, self.height * 0.866 + 0.5)
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Remove spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()
    
    def plot_umatrix(self, output_file: str = 'som_umatrix.png', 
                    figsize: Tuple[int, int] = (10, 8),
                    colormap: str = 'rainbow'):
        """
        Plot the U-Matrix (unified distance matrix).
        
        Args:
            output_file: Path to save the visualization
            figsize: Size of the figure
            colormap: Colormap to use for visualization
        """
        plt.figure(figsize=figsize)
        
        # Calculate and plot the U-Matrix
        umatrix = self.get_distance_map()
        plt.imshow(umatrix, cmap=colormap)
        plt.colorbar(label='Distance')
        plt.title('U-Matrix (distances between adjacent neurons)')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()
        
        return umatrix
    
    def plot_sample_distribution(self, data: np.ndarray, 
                                output_file: str = 'som_distribution.png', 
                                figsize: Tuple[int, int] = (10, 8),
                                colormap: str = 'rainbow'):
        """
        Plot the distribution of samples across the SOM.
        
        Args:
            data: Input data (shape: num_samples x input_dim)
            output_file: Path to save the visualization
            figsize: Size of the figure
            colormap: Colormap to use for visualization
        """
        plt.figure(figsize=figsize)
        
        # Get BMUs for all data points
        bmu_coords = self.predict(data)
        
        # Count occurrences of each BMU
        sample_counts = np.zeros((self.width, self.height))
        for i, j in bmu_coords:
            sample_counts[i, j] += 1
        
        # Plot the distribution
        plt.imshow(sample_counts, cmap=colormap)
        plt.colorbar(label='Sample count')
        plt.title('Distribution of Samples')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()
        
        return sample_counts
    
    def plot_component_planes(self, component_indices: List[int] = None, 
                             component_names: List[str] = None,
                             output_file: str = 'som_components.png',
                             figsize: Tuple[int, int] = None,
                             colormap: str = 'rainbow'):
        """
        Plot component planes to visualize how individual dimensions are organized.
        
        Args:
            component_indices: Indices of components to plot (if None, plot first min(10, input_dim))
            component_names: Names for the components (if None, use indices)
            output_file: Path to save the visualization
            figsize: Size of the figure
            colormap: Colormap to use for visualization
        """
        # Determine which components to plot
        if component_indices is None:
            component_indices = list(range(min(10, self.input_dim)))
        
        n_components = len(component_indices)
        
        # Set up grid for plotting
        n_cols = min(5, n_components)
        n_rows = int(np.ceil(n_components / n_cols))
        
        if figsize is None:
            figsize = (4 * n_cols, 3 * n_rows)
        
        plt.figure(figsize=figsize)
        
        # Plot each component
        for i, comp_idx in enumerate(component_indices):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Extract the component values
            component = self.weights[:, :, comp_idx]
            
            # Plot the component plane
            plt.imshow(component, cmap=colormap)
            plt.colorbar()
            
            # Set title
            if component_names is not None and i < len(component_names):
                title = component_names[i]
            else:
                title = f"Component {comp_idx}"
            plt.title(title)
            
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()

    def plot_training_history(self, output_file: str = 'som_training_history.png',
                             figsize: Tuple[int, int] = (10, 6)):
        """
        Plot the training history (learning rate and sigma over iterations).
        
        Args:
            output_file: Path to save the visualization
            figsize: Size of the figure
        """
        plt.figure(figsize=figsize)
        
        # Plot learning rate
        plt.subplot(2, 1, 1)
        plt.plot(self.training_history['iteration'], self.training_history['learning_rate'])
        plt.title('Learning Rate Decay')
        plt.xlabel('Iteration')
        plt.ylabel('Learning Rate')
        plt.grid(True)
        
        # Plot sigma (neighborhood radius)
        plt.subplot(2, 1, 2)
        plt.plot(self.training_history['iteration'], self.training_history['sigma'])
        plt.title('Neighborhood Radius Decay')
        plt.xlabel('Iteration')
        plt.ylabel('Sigma')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.show()

# Function to load embeddings, consistent with previous implementation
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

def analyze_som_clusters(som: SelfOrganizingMap, data: np.ndarray, 
                        texts: List[str] = None) -> pd.DataFrame:
    """
    Analyze the clusters formed by the SOM
    
    Args:
        som: Trained SOM model
        data: Input data
        texts: Optional list of texts corresponding to data points
        
    Returns:
        DataFrame with cluster statistics
    """
    # Get BMUs for all data points
    bmu_coords = som.predict(data)
    
    # Convert BMU coordinates to cluster IDs
    cluster_ids = [x * som.height + y for x, y in bmu_coords]
    unique_clusters = set(cluster_ids)
    
    # Prepare cluster statistics
    cluster_stats = []
    for cluster_id in unique_clusters:
        # Find samples in this cluster
        cluster_samples = [i for i, c in enumerate(cluster_ids) if c == cluster_id]
        if not cluster_samples:
            continue
            
        # Get BMU coordinates
        x, y = cluster_id // som.height, cluster_id % som.height
        
        # Calculate centroid of cluster samples
        centroid = np.mean([data[i] for i in cluster_samples], axis=0)
        
        # Calculate intra-cluster distance (average distance to centroid)
        distances = [np.linalg.norm(data[i] - centroid) for i in cluster_samples]
        
        # Add to stats
        cluster_stats.append({
            'cluster_id': cluster_id,
            'x': x,
            'y': y,
            'size': len(cluster_samples),
            'avg_distance': np.mean(distances),
            'min_distance': np.min(distances) if distances else 0,
            'max_distance': np.max(distances) if distances else 0,
            'sample_indices': cluster_samples,
            'example_texts': [texts[i][:50] + "..." if texts and len(texts[i]) > 50 else texts[i] 
                             for i in cluster_samples[:3]] if texts else None
        })
    
    # Convert to DataFrame and sort by size
    df = pd.DataFrame(cluster_stats).sort_values('size', ascending=False)
    
    # Save the results to disk
    df.to_csv('som_cluster_analysis.csv', index=False)
    
    return df

def analyze_sample_distribution(sample_counts: np.ndarray, expected_total: int = None) -> Dict:
    """
    Analyze the distribution of samples across SOM neurons.
    
    Args:
        sample_counts: 2D array with count of samples per neuron
        expected_total: Expected total number of samples (for validation)
        
    Returns:
        Dictionary with distribution statistics
    """
    # Flatten the counts array for easier analysis
    counts = sample_counts.flatten()
    
    # Calculate basic statistics
    total_samples = int(np.sum(counts))
    occupied_neurons = np.sum(counts > 0)
    empty_neurons = np.sum(counts == 0)
    max_samples = int(np.max(counts))
    
    # Count neurons by sample count
    neuron_histogram = {}
    for i in range(max_samples + 1):
        count = np.sum(counts == i)
        if count > 0:
            neuron_histogram[i] = int(count)
    
    # Calculate percentages
    total_neurons = len(counts)
    percent_occupied = occupied_neurons / total_neurons * 100
    
    # Validate against expected total
    total_matches = "Unknown (no expected total provided)"
    if expected_total is not None:
        total_matches = total_samples == expected_total
    
    # Create statistics dictionary
    stats = {
        "total_samples": total_samples,
        "total_neurons": total_neurons,
        "occupied_neurons": int(occupied_neurons),
        "empty_neurons": int(empty_neurons),
        "percent_occupied": f"{percent_occupied:.1f}%",
        "max_samples_per_neuron": max_samples,
        "neuron_histogram": neuron_histogram,
        "matches_expected": total_matches
    }
    
    return stats

def main():
    # Path for model saving/loading
    model_dir = 'models'
    model_path = os.path.join(model_dir, 'text_embeddings_som.pkl')
    
    # Create model directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Load embeddings
    filepath = 'analysis/embeddings/data/295-openai-emb.json'
    embeddings_np, texts, _ = load_embeddings(filepath)
    
    # Check if a previously trained model exists
    if os.path.exists(model_path):
        print(f"Found existing model at {model_path}")
        choice = input("Load existing model? (y/n): ").strip().lower()
        
        if choice == 'y':
            # Load existing model
            som = SelfOrganizingMap.load(model_path)
            
            # Check if we should continue training
            choice = input("Continue training? (y/n): ").strip().lower()
            if choice == 'y':
                # Continue training the model
                n_iterations = int(input("Enter number of additional iterations: ").strip())
                batch_size = int(input("Enter batch size (or 0 for all data): ").strip())
                
                # Use all data if batch_size is 0
                batch_size = None if batch_size == 0 else batch_size
                
                # Continue training
                som.train(embeddings_np, num_iterations=n_iterations, 
                         batch_size=batch_size, verbose=True, 
                         verbose_interval=100, continue_training=True)
                
                # Save the updated model
                som.save(model_path)
        else:
            # Create new model
            print("Creating new model...")
            som_size = (20, 20)  # width x height grid
            som = SelfOrganizingMap(width=som_size[0], height=som_size[1], 
                                  input_dim=embeddings_np.shape[1],
                                  learning_rate=0.5, sigma=None, random_seed=42)
            
            # Initialize weights from samples
            som.initialize_weights_with_samples(embeddings_np)
            
            # Train the SOM
            som.train(embeddings_np, num_iterations=5000, batch_size=50, verbose=True, 
                     verbose_interval=100)
            
            # Save the model
            som.save(model_path)
    else:
        # No existing model, create and train a new one
        print("No existing model found. Creating new model...")
        som_size = (20, 20)  # width x height grid
        som = SelfOrganizingMap(width=som_size[0], height=som_size[1], 
                               input_dim=embeddings_np.shape[1],
                               learning_rate=0.5, sigma=None, random_seed=42)
        
        # Initialize weights from samples
        som.initialize_weights_with_samples(embeddings_np)
        
        # Train the SOM
        som.train(embeddings_np, num_iterations=500, batch_size=50, verbose=True, 
                 verbose_interval=100)
        
        # Save the model
        som.save(model_path)
    
    # Plot training history
    som.plot_training_history(output_file='som_training_history.png')
    
    # Visualize U-Matrix (hexagonal grid with rainbow palette)
    som.plot_umatrix_hex(output_file='som_umatrix_hex.png', colormap='rainbow')
    
    # Visualize sample distribution (hexagonal grid with rainbow palette)
    sample_counts = som.plot_sample_distribution_hex(embeddings_np, 
                                                   output_file='som_distribution_hex.png', 
                                                   colormap='viridis')
    
    # Analyze sample distribution
    dist_stats = analyze_sample_distribution(sample_counts, expected_total=len(embeddings_np))
    print("\nSample Distribution Analysis:")
    print(f"Total samples: {dist_stats['total_samples']} (Expected: {len(embeddings_np)}, Match: {dist_stats['matches_expected']})")
    print(f"Neurons: {dist_stats['occupied_neurons']} occupied, {dist_stats['empty_neurons']} empty ({dist_stats['percent_occupied']} occupied)")
    print(f"Maximum samples per neuron: {dist_stats['max_samples_per_neuron']}")
    print("\nNeuron histogram (samples → count):")
    for samples, count in sorted(dist_stats['neuron_histogram'].items()):
        print(f"  {samples} samples: {count} neurons")
    
    # Visualize component planes (hexagonal grid with rainbow palette)
    som.plot_component_planes_hex(component_indices=list(range(5)), 
                                output_file='som_components_hex.png',
                                colormap='rainbow')
    
    # Analyze clusters
    cluster_stats = analyze_som_clusters(som, embeddings_np, texts=texts)
    print("\nTop 5 clusters by size:")
    print(cluster_stats[['cluster_id', 'x', 'y', 'size', 'avg_distance']].head(5))
    
    # Print info about saved data
    print("\nResults have been saved:")
    print("- Model saved to:", model_path)
    print("- Cluster analysis saved to: som_cluster_analysis.csv")
    print("- Visualization saved to: som_umatrix_hex.png, som_distribution_hex.png, som_components_hex.png")

if __name__ == "__main__":
    main() 