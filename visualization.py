"""
Visualization tools for transformer models including attention heatmaps,
embedding visualizations, and training progress plots.

This module provides comprehensive visualization utilities for understanding
transformer model behavior and debugging.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple, Union
import os
from datetime import datetime


class AttentionVisualizer:
    """
    Visualizer for attention weights and patterns in transformer models.
    
    Provides tools for creating heatmaps, attention flow diagrams,
    and head comparison visualizations.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize the attention visualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for plots.
        """
        self.figsize = figsize
        plt.style.use('default')
        sns.set_palette("viridis")
    
    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        title: str = "Attention Heatmap",
        save_path: Optional[str] = None,
        head_idx: Optional[int] = None
    ) -> plt.Figure:
        """
        Create an attention heatmap visualization.
        
        Args:
            attention_weights (np.ndarray): Attention weights matrix (seq_len x seq_len).
            tokens (List[str]): List of tokens corresponding to sequence positions.
            title (str): Plot title.
            save_path (str, optional): Path to save the figure.
            head_idx (int, optional): Head index for multi-head attention.
            
        Returns:
            plt.Figure: The created figure.
        """
        if attention_weights.ndim != 2:
            raise ValueError(f"Expected 2D attention weights, got {attention_weights.ndim}D")
        
        if len(tokens) != attention_weights.shape[0]:
            raise ValueError(f"Number of tokens ({len(tokens)}) doesn't match attention matrix size ({attention_weights.shape[0]})")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', rotation=270, labelpad=20)
        
        # Set labels and title
        ax.set_xlabel('Key Tokens')
        ax.set_ylabel('Query Tokens')
        
        if head_idx is not None:
            title = f"{title} - Head {head_idx}"
        ax.set_title(title)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_multi_head_attention(
        self,
        attention_weights: List[np.ndarray],
        tokens: List[str],
        title: str = "Multi-Head Attention",
        save_path: Optional[str] = None,
        max_heads: int = 8
    ) -> plt.Figure:
        """
        Visualize attention patterns across multiple heads.
        
        Args:
            attention_weights (List[np.ndarray]): List of attention matrices for each head.
            tokens (List[str]): List of tokens.
            title (str): Main plot title.
            save_path (str, optional): Path to save the figure.
            max_heads (int): Maximum number of heads to display.
            
        Returns:
            plt.Figure: The created figure.
        """
        num_heads = min(len(attention_weights), max_heads)
        cols = min(4, num_heads)
        rows = (num_heads + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
        if num_heads == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        
        for i in range(num_heads):
            ax = axes[i]
            
            # Create heatmap for this head
            im = ax.imshow(attention_weights[i], cmap='Blues', aspect='auto')
            
            # Set ticks and labels
            ax.set_xticks(range(len(tokens)))
            ax.set_yticks(range(len(tokens)))
            
            if len(tokens) <= 20:  # Only show labels for short sequences
                ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
                ax.set_yticklabels(tokens, fontsize=8)
            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])
            
            ax.set_title(f'Head {i+1}', fontsize=10)
            
            # Add colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.ax.tick_params(labelsize=8)
        
        # Hide extra subplots
        for i in range(num_heads, len(axes)):
            axes[i].set_visible(False)
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_attention_flow(
        self,
        attention_weights: np.ndarray,
        tokens: List[str],
        threshold: float = 0.1,
        title: str = "Attention Flow",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Create an attention flow diagram showing strong connections.
        
        Args:
            attention_weights (np.ndarray): Attention weights matrix.
            tokens (List[str]): List of tokens.
            threshold (float): Minimum attention weight to display.
            title (str): Plot title.
            save_path (str, optional): Path to save the figure.
            
        Returns:
            plt.Figure: The created figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        seq_len = len(tokens)
        positions = np.arange(seq_len)
        
        # Plot token positions
        ax.scatter(positions, [0] * seq_len, s=100, c='darkblue', alpha=0.7)
        
        # Add token labels
        for i, token in enumerate(tokens):
            ax.annotate(token, (i, 0), xytext=(0, 20), 
                       textcoords='offset points', ha='center', fontsize=10)
        
        # Draw attention connections
        for i in range(seq_len):
            for j in range(seq_len):
                if attention_weights[i, j] > threshold and i != j:
                    # Draw arrow from j to i (key to query)
                    ax.annotate('', xy=(i, 0), xytext=(j, 0),
                               arrowprops=dict(arrowstyle='->', 
                                             alpha=attention_weights[i, j],
                                             lw=attention_weights[i, j]*3,
                                             color='red'))
        
        ax.set_xlim(-0.5, seq_len - 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_xlabel('Token Position')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Remove y-axis
        ax.set_yticks([])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class EmbeddingVisualizer:
    """
    Visualizer for embedding spaces and token representations.
    
    Provides tools for dimensionality reduction, clustering visualization,
    and semantic similarity analysis.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (10, 8)):
        """
        Initialize the embedding visualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for plots.
        """
        self.figsize = figsize
    
    def plot_embedding_pca(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        title: str = "Embedding PCA",
        save_path: Optional[str] = None,
        n_components: int = 2
    ) -> plt.Figure:
        """
        Visualize embeddings using PCA dimensionality reduction.
        
        Args:
            embeddings (np.ndarray): Embedding matrix (n_tokens x embedding_dim).
            labels (List[str]): Labels for each embedding.
            title (str): Plot title.
            save_path (str, optional): Path to save the figure.
            n_components (int): Number of PCA components (2 or 3).
            
        Returns:
            plt.Figure: The created figure.
        """
        from sklearn.decomposition import PCA
        
        if embeddings.shape[0] != len(labels):
            raise ValueError(f"Number of embeddings ({embeddings.shape[0]}) doesn't match number of labels ({len(labels)})")
        
        # Apply PCA
        pca = PCA(n_components=n_components)
        embeddings_pca = pca.fit_transform(embeddings)
        
        if n_components == 2:
            fig, ax = plt.subplots(figsize=self.figsize)
            scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], 
                               alpha=0.7, s=50)
            
            # Add labels
            for i, label in enumerate(labels):
                ax.annotate(label, (embeddings_pca[i, 0], embeddings_pca[i, 1]),
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
            
        elif n_components == 3:
            fig = plt.figure(figsize=self.figsize)
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(embeddings_pca[:, 0], embeddings_pca[:, 1], embeddings_pca[:, 2],
                               alpha=0.7, s=50)
            
            # Add labels (subset for readability)
            for i, label in enumerate(labels[:20]):  # Limit to first 20 for readability
                ax.text(embeddings_pca[i, 0], embeddings_pca[i, 1], embeddings_pca[i, 2], 
                       label, fontsize=8)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
        
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_similarity_matrix(
        self,
        embeddings: np.ndarray,
        labels: List[str],
        title: str = "Embedding Similarity Matrix",
        save_path: Optional[str] = None,
        metric: str = 'cosine'
    ) -> plt.Figure:
        """
        Visualize pairwise similarity between embeddings.
        
        Args:
            embeddings (np.ndarray): Embedding matrix.
            labels (List[str]): Labels for embeddings.
            title (str): Plot title.
            save_path (str, optional): Path to save the figure.
            metric (str): Similarity metric ('cosine', 'euclidean').
            
        Returns:
            plt.Figure: The created figure.
        """
        if metric == 'cosine':
            # Cosine similarity
            norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            similarity_matrix = np.dot(norm_embeddings, norm_embeddings.T)
        elif metric == 'euclidean':
            # Negative euclidean distance (so higher values = more similar)
            distances = np.linalg.norm(embeddings[:, None] - embeddings[None, :], axis=2)
            similarity_matrix = -distances
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create heatmap
        im = ax.imshow(similarity_matrix, cmap='RdYlBu_r', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(f'{metric.capitalize()} Similarity', rotation=270, labelpad=20)
        
        ax.set_title(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class TrainingVisualizer:
    """
    Visualizer for training progress and model performance metrics.
    
    Provides tools for plotting loss curves, learning rates, and other
    training diagnostics.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize the training visualizer.
        
        Args:
            figsize (Tuple[int, int]): Default figure size for plots.
        """
        self.figsize = figsize
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: Optional[List[float]] = None,
        title: str = "Training Progress",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses (List[float]): Training losses per epoch.
            val_losses (List[float], optional): Validation losses per epoch.
            title (str): Plot title.
            save_path (str, optional): Path to save the figure.
            
        Returns:
            plt.Figure: The created figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(train_losses) + 1)
        
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
        
        if val_losses:
            if len(val_losses) != len(train_losses):
                print("Warning: Training and validation losses have different lengths")
            val_epochs = range(1, len(val_losses) + 1)
            ax.plot(val_epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Set log scale if losses span multiple orders of magnitude
        if max(train_losses) / min(train_losses) > 100:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_learning_rate_schedule(
        self,
        learning_rates: List[float],
        title: str = "Learning Rate Schedule",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot learning rate schedule over training.
        
        Args:
            learning_rates (List[float]): Learning rates per epoch.
            title (str): Plot title.
            save_path (str, optional): Path to save the figure.
            
        Returns:
            plt.Figure: The created figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        epochs = range(1, len(learning_rates) + 1)
        ax.plot(epochs, learning_rates, 'g-', linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        
        # Use log scale for learning rate if it varies significantly
        if max(learning_rates) / min(learning_rates) > 10:
            ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_gradient_norms(
        self,
        gradient_norms: Dict[str, List[float]],
        title: str = "Gradient Norms",
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Plot gradient norms for different layers/components.
        
        Args:
            gradient_norms (Dict[str, List[float]]): Gradient norms by component.
            title (str): Plot title.
            save_path (str, optional): Path to save the figure.
            
        Returns:
            plt.Figure: The created figure.
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for component, norms in gradient_norms.items():
            epochs = range(1, len(norms) + 1)
            ax.plot(epochs, norms, label=component, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class VisualizationManager:
    """
    Manager class for coordinating different visualization tools.
    
    Provides a unified interface for creating comprehensive
    visualization reports and managing output directories.
    """
    
    def __init__(self, output_dir: str = "./visualizations"):
        """
        Initialize the visualization manager.
        
        Args:
            output_dir (str): Directory to save visualizations.
        """
        self.output_dir = output_dir
        self.attention_viz = AttentionVisualizer()
        self.embedding_viz = EmbeddingVisualizer()
        self.training_viz = TrainingVisualizer()
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def create_model_report(
        self,
        model_data: Dict[str, Any],
        report_name: str = "model_analysis"
    ) -> str:
        """
        Create a comprehensive visualization report for a model.
        
        Args:
            model_data (Dict[str, Any]): Dictionary containing model data.
            report_name (str): Name for the report directory.
            
        Returns:
            str: Path to the created report directory.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = os.path.join(self.output_dir, f"{report_name}_{timestamp}")
        os.makedirs(report_dir, exist_ok=True)
        
        # Generate attention visualizations
        if 'attention_weights' in model_data:
            attention_data = model_data['attention_weights']
            tokens = model_data.get('tokens', [f"token_{i}" for i in range(len(attention_data[0]))])
            
            for i, attn_weights in enumerate(attention_data):
                save_path = os.path.join(report_dir, f"attention_head_{i}.png")
                self.attention_viz.plot_attention_heatmap(
                    attn_weights, tokens, f"Attention Head {i}", save_path
                )
            
            # Multi-head overview
            save_path = os.path.join(report_dir, "multi_head_attention.png")
            self.attention_viz.plot_multi_head_attention(
                attention_data, tokens, "Multi-Head Attention Overview", save_path
            )
        
        # Generate embedding visualizations
        if 'embeddings' in model_data:
            embeddings = model_data['embeddings']
            labels = model_data.get('embedding_labels', [f"emb_{i}" for i in range(len(embeddings))])
            
            # PCA visualization
            save_path = os.path.join(report_dir, "embeddings_pca.png")
            self.embedding_viz.plot_embedding_pca(
                embeddings, labels, "Embedding PCA", save_path
            )
            
            # Similarity matrix
            save_path = os.path.join(report_dir, "embedding_similarity.png")
            self.embedding_viz.plot_similarity_matrix(
                embeddings, labels, "Embedding Similarity", save_path
            )
        
        # Generate training visualizations
        if 'training_history' in model_data:
            history = model_data['training_history']
            
            if 'train_losses' in history:
                save_path = os.path.join(report_dir, "training_curves.png")
                self.training_viz.plot_training_curves(
                    history['train_losses'],
                    history.get('val_losses'),
                    "Training Progress",
                    save_path
                )
            
            if 'learning_rates' in history:
                save_path = os.path.join(report_dir, "learning_rate.png")
                self.training_viz.plot_learning_rate_schedule(
                    history['learning_rates'],
                    "Learning Rate Schedule",
                    save_path
                )
        
        # Create summary HTML report
        self._create_html_report(report_dir, model_data)
        
        return report_dir
    
    def _create_html_report(self, report_dir: str, model_data: Dict[str, Any]):
        """Create an HTML summary report."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Transformer Model Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .image {{ text-align: center; margin: 20px 0; }}
                img {{ max-width: 800px; height: auto; border: 1px solid #ddd; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Transformer Model Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            </div>
            
            <div class="section">
                <h2>Model Overview</h2>
                <ul>
                    <li>Embedding Dimension: {model_data.get('embedding_dim', 'N/A')}</li>
                    <li>Number of Heads: {model_data.get('num_heads', 'N/A')}</li>
                    <li>Sequence Length: {model_data.get('seq_len', 'N/A')}</li>
                </ul>
            </div>
        """
        
        # Add sections for each visualization type
        image_files = [f for f in os.listdir(report_dir) if f.endswith('.png')]
        
        if any('attention' in f for f in image_files):
            html_content += """
            <div class="section">
                <h2>Attention Analysis</h2>
            """
            for img_file in sorted(image_files):
                if 'attention' in img_file:
                    html_content += f"""
                    <div class="image">
                        <h3>{img_file.replace('_', ' ').replace('.png', '').title()}</h3>
                        <img src="{img_file}" alt="{img_file}">
                    </div>
                    """
            html_content += "</div>"
        
        if any('embedding' in f for f in image_files):
            html_content += """
            <div class="section">
                <h2>Embedding Analysis</h2>
            """
            for img_file in sorted(image_files):
                if 'embedding' in img_file:
                    html_content += f"""
                    <div class="image">
                        <h3>{img_file.replace('_', ' ').replace('.png', '').title()}</h3>
                        <img src="{img_file}" alt="{img_file}">
                    </div>
                    """
            html_content += "</div>"
        
        if any('training' in f or 'learning' in f for f in image_files):
            html_content += """
            <div class="section">
                <h2>Training Analysis</h2>
            """
            for img_file in sorted(image_files):
                if 'training' in img_file or 'learning' in img_file:
                    html_content += f"""
                    <div class="image">
                        <h3>{img_file.replace('_', ' ').replace('.png', '').title()}</h3>
                        <img src="{img_file}" alt="{img_file}">
                    </div>
                    """
            html_content += "</div>"
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        html_path = os.path.join(report_dir, "report.html")
        with open(html_path, 'w') as f:
            f.write(html_content)