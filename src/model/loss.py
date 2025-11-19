"""
    Loss functions for stellar flare detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
        Binary Cross Entropy loss with class weights.
        L_BCE = -1/n * Σ[y_i * log(ŷ_i) + (1 - y_i) * log(1 - ŷ_i)] * w_i
        
        where w_i is the weight for class i, helping address class imbalance.
        
        Attributes:
            weights (torch.Tensor): Class weights [weight_0, weight_1]
    """
    
    def __init__(self, weights):
        """
            Initialize the loss function.
            
            Arguments:
                weights (torch.Tensor): Class weights [weight_0, weight_1]
                    Default values from config: [0.7, 1.4]
        """
        super(WeightedBCELoss, self).__init__()
        self.weights = weights
        
    def forward(self, outputs, targets):
        """
            Calculate the weighted BCE loss.
            
            Arguments:
                outputs (torch.Tensor): Model predictions [batch_size, 1]
                targets (torch.Tensor): True labels [batch_size, 1]
                
            Returns:
                torch.Tensor: Weighted loss (scalar)
        """
        # Get weights for each sample based on its class
        weights = torch.zeros_like(targets)
        for i in range(len(self.weights)):
            weights[targets == i] = self.weights[i]
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy(outputs, targets, reduction='none')
        
        # Apply weights
        weighted_loss = bce_loss * weights
        
        return weighted_loss.mean()


def get_class_weights(train_loader):
    """
        Calculate class weights based on class distribution in training data.
        
        Uses inverse frequency weighting to address class imbalance:
        weight_i = total_samples / (num_classes * count_i)
        
        Arguments:
            train_loader (DataLoader): Training data loader
            
        Returns:
            torch.Tensor: Class weights [weight_non_flare, weight_flare]
    """
    # Count the number of samples in each class
    class_counts = [0, 0]  # [non-flare, flare]
    
    for _, targets, _ in train_loader:
        for target in targets:
            class_counts[int(target.item())] += 1
    
    # Calculate inverse frequency
    total_samples = sum(class_counts)
    class_weights = [
        total_samples / (len(class_counts) * count) if count > 0 else 1.0 
        for count in class_counts
    ]
    
    return torch.tensor(class_weights)