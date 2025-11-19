"""
    Multi-head attention mechanisms for time series flare detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
        This implementation follows the mathematical formulation:
        Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        
        Attributes:
            d_model (int): Model dimensionality
            num_heads (int): Number of attention heads
            d_k (int): Dimensionality per head (d_model // num_heads)
            q_linear (nn.Linear): Query projection
            k_linear (nn.Linear): Key projection
            v_linear (nn.Linear): Value projection
            out (nn.Linear): Output projection
            dropout (nn.Dropout): Dropout layer
            attn_weights (torch.Tensor): Stored attention weights for visualization
    """
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
            Initialize multi-head attention.
            
            Arguments:
                d_model (int): Model dimensionality
                num_heads (int): Number of attention heads
                dropout (float): Dropout rate
                
            Raises:
                AssertionError: If d_model is not divisible by num_heads
        """
        super(MultiHeadAttention, self).__init__()
        
        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        # Linear projections for Query, Key, Value
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for visualization/interpretability
        self.attn_weights = None
        
    def split_heads(self, x):
        """
            Split the last dimension into (num_heads, d_k).
            
            Arguments:
                x (torch.Tensor): Input tensor of shape [batch, seq_len, d_model]
                
            Returns:
                torch.Tensor: Reshaped tensor of shape [batch, num_heads, seq_len, d_k]
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
    
    def combine_heads(self, x):
        """
            Combine the heads back into original shape.
            
            Arguments:
                x (torch.Tensor): Input tensor of shape [batch, num_heads, seq_len, d_k]
                
            Returns:
                torch.Tensor: Combined tensor of shape [batch, seq_len, d_model]
        """
        batch_size, num_heads, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
    
    def forward(self, query, key, value, mask=None):
        """
            Compute multi-head attention following Section 3.2.1 methodology.
            
            Arguments:
                query (torch.Tensor): Query tensor of shape [batch, seq_len, d_model]
                key (torch.Tensor): Key tensor of shape [batch, seq_len, d_model]
                value (torch.Tensor): Value tensor of shape [batch, seq_len, d_model]
                mask (torch.Tensor, optional): Attention mask
                
            Returns:
                torch.Tensor: Output after attention of shape [batch, seq_len, d_model]
        """
        batch_size = query.size(0)
        
        # Linear projections and split heads
        q = self.split_heads(self.q_linear(query))  # [batch, num_heads, seq_len, d_k]
        k = self.split_heads(self.k_linear(key))    # [batch, num_heads, seq_len, d_k]
        v = self.split_heads(self.v_linear(value))  # [batch, num_heads, seq_len, d_k]
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        self.attn_weights = attn_weights.detach()  # Store for visualization
        
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention weights to values
        context = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, d_k]
        
        # Combine heads
        context = self.combine_heads(context)  # [batch, seq_len, d_model]
        
        # Final linear projection
        output = self.out(context)
        
        return output