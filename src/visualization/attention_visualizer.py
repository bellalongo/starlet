"""
    Attention visualization tools for transformer-based flare detection.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class AttentionVisualizer:
    """
        Visualizes and analyzes transformer attention patterns.
        
        Attributes:
            model (FlareClassifier): Trained transformer model
            device (torch.device): Device for computation
    """
    
    def __init__(self, model):
        """
            Initialize the attention visualizer.
            
            Arguments:
                model (FlareClassifier): Trained model with attention mechanisms
        """
        self.model = model
        self.device = next(model.parameters()).device
        
    def get_attention_rollout(self, input_tensor, layer_idx=None):
        """
            Compute attention rollout across transformer layers.

            Arguments:
                input_tensor (torch.Tensor): Input of shape [batch_size, seq_len, input_dim]
                layer_idx (int, optional): Stop at specific layer. If None, use all layers.
                
            Returns:
                torch.Tensor: Attention rollout matrix [batch_size, seq_len, seq_len]
                    or None if no attention weights available
        """
        self.model.eval()
        
        # Store attention weights for each layer
        attn_matrices = []
        
        # Process input through embedding and positional encoding
        x = self.model.input_embedding(input_tensor)
        x = self.model.positional_encoding(x)
        
        # Collect attention from each encoder layer
        for i, layer in enumerate(self.model.transformer_encoder.layers):
            # Get attention weights
            _ = layer.self_attn(x, x, x)
            
            # Average across attention heads
            attn_weights = layer.self_attn.attn_weights.mean(dim=1)
            attn_matrices.append(attn_weights)
            
            # Process through full layer
            x = layer(x)
            
            # Stop at specified layer
            if layer_idx is not None and i == layer_idx:
                break
        
        if not attn_matrices:
            return None
        
        # Initialize with identity matrix
        batch_size, seq_len = input_tensor.shape[0], input_tensor.shape[1]
        attention_rollout = torch.eye(seq_len).unsqueeze(0).repeat(
            batch_size, 1, 1
        ).to(self.device)
        
        # Propagate attention through layers
        for attn in attn_matrices:
            attention_rollout = torch.bmm(attn, attention_rollout)
        
        # Row-wise normalization
        attention_rollout = attention_rollout / attention_rollout.sum(
            dim=-1, keepdim=True
        )
        
        return attention_rollout
    
    def get_attention_maps(self, input_tensor, layer_idx=0, head_idx=None):
        """
            Extract attention weights for a specific layer and head.
            
            Arguments:
                input_tensor (torch.Tensor): Input tensor
                layer_idx (int): Encoder layer index (0-based)
                head_idx (int, optional): Attention head index. If None, average all heads.
                
            Returns:
                torch.Tensor: Attention weights [batch_size, seq_len, seq_len]
                    or [batch_size, num_heads, seq_len, seq_len] if head_idx is None
        """
        self.model.eval()
        
        # Process through embedding and positional encoding
        x = self.model.input_embedding(input_tensor)
        x = self.model.positional_encoding(x)
        
        # Process through layers up to target layer
        for i, layer in enumerate(self.model.transformer_encoder.layers):
            if i == layer_idx:
                # Get attention weights
                _ = layer.self_attn(x, x, x)
                attn_weights = layer.self_attn.attn_weights
                
                # Return specific head or average
                if head_idx is not None:
                    return attn_weights[:, head_idx, :, :]
                else:
                    return attn_weights.mean(dim=1)
            else:
                # Just process through layer
                x = layer(x)
        
        return None
    
    def integrate_gradients(self, input_tensor, target_class=1, steps=50):
        """
            Compute integrated gradients for input attribution.
            
            Attributes model predictions to input features, helping understand
            which time steps contribute most to classifications.
            
            Arguments:
                input_tensor (torch.Tensor): Input tensor
                target_class (int): Class to attribute (0=non-flare, 1=flare)
                steps (int): Number of integration steps
                
            Returns:
                torch.Tensor: Attribution scores [batch_size, seq_len]
        """
        self.model.eval()
        
        # Create baseline (zero tensor)
        baseline = torch.zeros_like(input_tensor, device=self.device)
        
        # Enable gradient tracking
        input_tensor.requires_grad_(True)
        
        # Collect gradients at interpolation steps
        gradients = []
        
        for step in range(steps + 1):
            # Interpolate between baseline and input
            alpha = step / steps
            interpolated = baseline + alpha * (input_tensor - baseline)
            interpolated.requires_grad_(True)
            
            # Forward pass
            output = self.model(interpolated)
            
            # Clear previous gradients
            if interpolated.grad is not None:
                interpolated.grad.zero_()
            
            # Compute gradient for target class
            grad = torch.autograd.grad(
                outputs=output,
                inputs=interpolated,
                grad_outputs=torch.ones_like(output),
                create_graph=False,
                retain_graph=False
            )[0]
            
            gradients.append(grad.detach())
        
        # Compute average gradient
        avg_grad = torch.stack(gradients).mean(dim=0)
        
        # Compute attribution
        attribution = (input_tensor - baseline) * avg_grad
        
        # Sum across feature dimension if multi-dimensional
        if len(attribution.shape) > 2:
            attribution = attribution.sum(dim=-1)
        
        return attribution
    
    def plot_attention_map(
        self, 
        input_tensor, 
        layer_idx=None, 
        head_idx=None,
        save_path=None, 
        title=None
    ):
        """
            Plot attention map heatmap
            
            Arguments:
                input_tensor (torch.Tensor): Input tensor
                layer_idx (int, optional): Layer index
                head_idx (int, optional): Head index
                save_path (str, optional): Path to save figure
                title (str, optional): Custom title
        """
        if layer_idx is None:
            # Use attention rollout
            attention = self.get_attention_rollout(input_tensor)
            default_title = "Attention Rollout (All Layers)"
        else:
            # Use specific layer/head
            attention = self.get_attention_maps(input_tensor, layer_idx, head_idx)
            layer_str = f"Layer {layer_idx + 1}"
            head_str = f", Head {head_idx + 1}" if head_idx is not None else ""
            default_title = f"Attention Map ({layer_str}{head_str})"
        
        if attention is None:
            print("Unable to compute attention weights")
            return
        
        # Create figure
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attention[0].cpu().numpy(), 
            cmap='viridis',
            cbar_kws={'label': 'Attention Weight'}
        )
        
        plt.title(title if title else default_title)
        plt.xlabel("Key Position (Time Step)")
        plt.ylabel("Query Position (Time Step)")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_attention_with_lightcurve(
        self, 
        input_tensor, 
        layer_idx=0,
        save_path=None
    ):
        """
            Plot light curve with attention overlay
            
            Arguments:
                input_tensor (torch.Tensor): Input tensor [1, seq_len, input_dim]
                layer_idx (int): Layer index to visualize
                save_path (str, optional): Path to save figure
        """
        # Get attention weights (average across heads)
        attention = self.get_attention_maps(input_tensor, layer_idx, head_idx=None)
        
        if attention is None:
            print("Unable to compute attention weights")
            return
        
        # Extract light curve and attention
        lightcurve = input_tensor[0, :, 0].cpu().numpy()
        attn_weights = attention[0].cpu().numpy()
        
        # Average attention across query positions for visualization
        avg_attention = attn_weights.mean(axis=0)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                        gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot light curve
        ax1.plot(lightcurve, color='darkgreen', linewidth=2)
        ax1.set_ylabel('Normalized Flux', fontsize=12)
        ax1.set_title('Light Curve with Attention Weights', fontsize=14, 
                      fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Overlay attention as background color
        time_steps = np.arange(len(lightcurve))
        ax1_twin = ax1.twinx()
        ax1_twin.fill_between(
            time_steps, 
            0, 
            avg_attention,
            alpha=0.3, 
            color='pink',
            label='Attention Weight'
        )
        ax1_twin.set_ylabel('Attention Weight', fontsize=12)
        ax1_twin.set_ylim(0, avg_attention.max() * 1.2)
        
        # Plot attention heatmap
        im = ax2.imshow(
            attn_weights, 
            aspect='auto', 
            cmap='Pinks',
            interpolation='nearest'
        )
        ax2.set_xlabel('Time Step (Key Position)', fontsize=12)
        ax2.set_ylabel('Query Position', fontsize=12)
        ax2.set_title(f'Attention Map - Layer {layer_idx + 1}', fontsize=12)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax2, label='Attention Weight')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def plot_attribution(
        self, 
        input_tensor, 
        target_class=1, 
        save_path=None
    ):
        """
            Plot integrated gradients attribution scores.
            
            Arguments:
                input_tensor (torch.Tensor): Input tensor
                target_class (int): Target class for attribution
                save_path (str, optional): Path to save figure
        """
        attribution = self.integrate_gradients(input_tensor, target_class)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6))
        
        # Plot original light curve
        ax1.plot(input_tensor[0, :, 0].cpu().numpy(), color='darkgreen')
        ax1.set_ylabel('Normalized Flux')
        ax1.set_title('Original Light Curve')
        ax1.grid(True, alpha=0.3)
        
        # Plot attribution scores
        ax2.bar(
            range(len(attribution[0])), 
            attribution[0].cpu().numpy(),
            color='skyblue',
            alpha=0.7
        )
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Attribution Score')
        ax2.set_title(f'Integrated Gradients Attribution (Class {target_class})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
    
    def analyze_attention_statistics(self, data_loader, num_samples=100):
        """
            Analyze attention patterns across multiple samples.
            
            Arguments:
                data_loader (DataLoader): Data loader with samples
                num_samples (int): Number of samples to analyze
                
            Returns:
                dict: Statistics including mean attention per position,
                    attention entropy, and focus patterns
        """
        self.model.eval()
        
        all_attentions = []
        sample_count = 0
        
        with torch.no_grad():
            for data, targets, _ in data_loader:
                data = data.to(self.device)
                
                # Get attention for each sample
                for i in range(len(data)):
                    if sample_count >= num_samples:
                        break
                    
                    sample = data[i:i+1]
                    attention = self.get_attention_rollout(sample)
                    
                    if attention is not None:
                        # Average across query positions
                        avg_attn = attention[0].mean(dim=0).cpu().numpy()
                        all_attentions.append(avg_attn)
                        sample_count += 1
                
                if sample_count >= num_samples:
                    break
        
        # Compute statistics
        all_attentions = np.array(all_attentions)
        
        stats = {
            'mean_attention': all_attentions.mean(axis=0),
            'std_attention': all_attentions.std(axis=0),
            'max_position': np.argmax(all_attentions.mean(axis=0)),
            'attention_spread': all_attentions.std(axis=0).mean(),
        }
        
        return stats