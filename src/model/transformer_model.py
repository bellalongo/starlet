"""
    Transformer architecture for stellar flare detection
"""

import torch
import torch.nn as nn
import math

from attention import MultiHeadAttention


class PositionalEncoding(nn.Module):
    """
        Uses sine and cosine functions of different frequencies to encode temporal 
        information, following the formulation:
        
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Attributes:
            dropout (nn.Dropout): Dropout layer
            pe (torch.Tensor): Positional encoding matrix [1, max_len, d_model]
    """
    
    def __init__(self, d_model, max_len=100, dropout=0.1):
        """
            Initialize the positional encoding layer.
            
            Arguments:
                d_model (int): Dimensionality of the model
                max_len (int): Maximum sequence length
                dropout (float): Dropout rate
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # Apply sine to even indices and cosine to odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension and register as buffer (not a parameter)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
            Add positional encoding to input.
            
            Arguments:
                x (torch.Tensor): Input tensor of shape [batch_size, seq_len, d_model]
                
            Returns:
                torch.Tensor: Input with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
        Consists of multi-head self-attention followed by a feed-forward network,
        with residual connections and layer normalization.
        
        Attributes:
            self_attn (MultiHeadAttention): Multi-head attention mechanism
            feed_forward (nn.Sequential): Feed-forward network
            norm1 (nn.LayerNorm): First layer normalization
            norm2 (nn.LayerNorm): Second layer normalization
            dropout (nn.Dropout): Dropout layer
    """
    
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
            Initialize the encoder layer.
            
            Arguments:
                d_model (int): Model dimensionality
                num_heads (int): Number of attention heads
                d_ff (int): Feed-forward network dimensionality
                dropout (float): Dropout rate
        """
        super(TransformerEncoderLayer, self).__init__()
        
        # Multi-head attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        # Layer normalization and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        """
            Process input through encoder layer.
            
            Arguments:
                x (torch.Tensor): Input tensor of shape [batch, seq_len, d_model]
                mask (torch.Tensor, optional): Attention mask
                
            Returns:
                torch.Tensor: Output after encoder layer
        """
        # Multi-head attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward network with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerEncoder(nn.Module):
    """
        Transformer encoder consisting of multiple encoder layers.
        
        Attributes:
            layers (nn.ModuleList): List of encoder layers
    """
    
    def __init__(self, num_layers, d_model, num_heads, d_ff, dropout=0.1):
        """
            Initialize the transformer encoder.
            
            Arguments:
                num_layers (int): Number of encoder layers
                d_model (int): Model dimensionality
                num_heads (int): Number of attention heads
                d_ff (int): Feed-forward network dimensionality
                dropout (float): Dropout rate
        """
        super(TransformerEncoder, self).__init__()
        
        # Create multiple encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        """
            Process input through all encoder layers.
            
            Arguments:
                x (torch.Tensor): Input tensor of shape [batch, seq_len, d_model]
                mask (torch.Tensor, optional): Attention mask
                
            Returns:
                torch.Tensor: Output after all encoder layers
        """
        # Pass through each encoder layer sequentially
        for layer in self.layers:
            x = layer(x, mask)
        
        return x


class FlareClassifier(nn.Module):
    """
        Consists of input embedding, positional encoding, transformer encoder, 
        and classification head.
        
        Attributes:
            input_embedding (nn.Linear): Projects input features to model dimensions
            positional_encoding (PositionalEncoding): Adds temporal information
            transformer_encoder (TransformerEncoder): Core transformer layers
            classification_head (nn.Sequential): Final classification layers
    """
    
    def __init__(
        self, 
        input_dim=3, 
        d_model=256, 
        num_heads=8, 
        num_layers=2, 
        d_ff=512, 
        max_seq_len=100, 
        dropout=0.1
    ):
        """
            Initialize the flare classifier.
            
            Arguments:
                input_dim (int): Dimensionality of input features per time step
                d_model (int): Model dimensionality
                num_heads (int): Number of attention heads
                num_layers (int): Number of encoder layers
                d_ff (int): Feed-forward network dimensionality
                max_seq_len (int): Maximum sequence length
                dropout (float): Dropout rate
        """
        super(FlareClassifier, self).__init__()
        
        # Input embedding to convert input features to model dimensions
        self.input_embedding = nn.Linear(input_dim, d_model)
        
        # Positional encoding to add temporal information
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder
        self.transformer_encoder = TransformerEncoder(
            num_layers, d_model, num_heads, d_ff, dropout
        )
        
        # Classification head as described in methodology
        self.classification_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model * max_seq_len, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x, mask=None):
        """
            Process input through complete model.
            
            Arguments:
                x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim]
                mask (torch.Tensor, optional): Attention mask
                
            Returns:
                torch.Tensor: Output probability of flare [batch_size, 1]
        """
        # Input embedding
        x = self.input_embedding(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through transformer encoder
        encoder_output = self.transformer_encoder(x, mask)
        
        # Classification head
        output = self.classification_head(encoder_output)
        
        return output
    
    def get_attention_maps(self, x, layer_idx=0, head_idx=0):
        """
            Extract attention weights for visualization
            
            Arguments:
                x (torch.Tensor): Input tensor
                layer_idx (int): Index of the encoder layer to visualize
                head_idx (int): Index of the attention head to visualize
                
            Returns:
                torch.Tensor: Attention weights or None if unavailable
        """
        # Input embedding and positional encoding
        x = self.positional_encoding(self.input_embedding(x))
        
        # Process through layers up to the specified layer
        for i in range(layer_idx + 1):
            layer = self.transformer_encoder.layers[i]
            if i == layer_idx:
                # Process through attention mechanism to get attention weights
                _ = layer.self_attn(x, x, x)
                # Extract weights for the specified head
                attn_weights = layer.self_attn.attn_weights[:, head_idx]
                return attn_weights
            else:
                # Just process through the layer
                x = layer(x)
                
        return None


def create_model(
    input_dim=3, 
    d_model=256, 
    num_heads=8, 
    num_layers=2, 
    d_ff=512, 
    max_seq_len=100, 
    dropout=0.1
):
    """
        Create the flare detection model with specified hyperparameters.
        
        Arguments:
            input_dim (int): Dimensionality of input features per time step
            d_model (int): Model dimensionality
            num_heads (int): Number of attention heads
            num_layers (int): Number of encoder layers
            d_ff (int): Feed-forward network dimensionality
            max_seq_len (int): Maximum sequence length
            dropout (float): Dropout rate
            
        Returns:
            FlareClassifier: Initialized model
    """
    model = FlareClassifier(
        input_dim=input_dim, 
        d_model=d_model, 
        num_heads=num_heads, 
        num_layers=num_layers,
        d_ff=d_ff, 
        max_seq_len=max_seq_len, 
        dropout=dropout
    )
    
    return model