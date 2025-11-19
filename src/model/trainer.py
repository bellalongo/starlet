"""
    Training and evaluation procedures for stellar flare detection.
"""

import torch
import numpy as np


class Trainer:
    """
        Handles model training, validation, and checkpointing.
        
        Attributes:
            model (nn.Module): Model to train
            optimizer (torch.optim.Optimizer): Optimizer for training
            criterion (callable): Loss function
            device (torch.device): Device to use for training
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler
    """
    
    def __init__(self, model, optimizer, criterion, device, scheduler=None):
        """
            Initialize the trainer.
            
            Arguments:
                model (nn.Module): Model to train
                optimizer (torch.optim.Optimizer): Optimizer for training
                criterion (callable): Loss function
                device (torch.device): Device to use for training
                scheduler (torch.optim.lr_scheduler._LRScheduler, optional): Learning rate scheduler
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        
        # Move model to device
        self.model = self.model.to(self.device)
        
    def train_epoch(self, train_loader, epoch):
        """
            Train for one epoch
            
            Arguments:
                train_loader (DataLoader): Training data loader
                epoch (int): Current epoch number
                
            Returns:
                tuple: (average training loss, training accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, targets, _) in enumerate(train_loader):
            # Move data to device
            data, targets = data.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            
            # Calculate loss
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            
            # Accumulate statistics
            total_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(
                    f'Epoch: {epoch}, Batch: {batch_idx + 1}/{len(train_loader)}, '
                    f'Loss: {loss.item():.4f}'
                )
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """ 
            Calculates metrics as defined in the thesis:
            - Accuracy = (TP + TN) / (TP + TN + FP + FN)
            - Precision = TP / (TP + FP)
            - Recall = TP / (TP + FN)
            - F1 = 2 * Precision * Recall / (Precision + Recall)
            
            Arguments:
                val_loader (DataLoader): Validation data loader
                
            Returns:
                tuple: (average validation loss, metrics dictionary)
        """
        self.model.eval()
        total_loss = 0
        all_targets = []
        all_predictions = []
        
        with torch.no_grad():
            for data, targets, _ in val_loader:
                # Move data to device
                data, targets = data.to(self.device), targets.to(self.device)
                
                # Forward pass
                outputs = self.model(data)
                
                # Calculate loss
                loss = self.criterion(outputs, targets)
                
                # Accumulate statistics
                total_loss += loss.item()
                
                # Store predictions and targets for metric calculation
                predicted = (outputs > 0.5).float()
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Calculate average loss
        avg_loss = total_loss / len(val_loader)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_targets, all_predictions)
        
        return avg_loss, metrics
    
    def calculate_metrics(self, targets, predictions):
        """
            Calculate evaluation metrics
            
            Arguments:
                targets (list): True labels
                predictions (list): Predicted labels
                
            Returns:
                dict: Dictionary containing accuracy, precision, recall, and F1 score
        """
        # Convert lists to numpy arrays
        targets = np.array(targets)
        predictions = np.array(predictions)
        
        # Calculate true positives, false positives, true negatives, false negatives
        tp = np.sum((predictions == 1) & (targets == 1))
        fp = np.sum((predictions == 1) & (targets == 0))
        tn = np.sum((predictions == 0) & (targets == 0))
        fn = np.sum((predictions == 0) & (targets == 1))
        
        # Calculate metrics following thesis formulations
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall) 
            if (precision + recall) > 0 
            else 0
        )
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def train(
        self, 
        train_loader, 
        val_loader, 
        num_epochs, 
        checkpoint_path, 
        early_stopping_patience=10
    ):
        """
            Train the model for multiple epochs with early stopping.
            
            Implements the complete training procedure:
            - Multiple epochs of training and validation
            - Learning rate scheduling
            - Early stopping based on F1 score
            - Model checkpointing
            
            Arguments:
                train_loader (DataLoader): Training data loader
                val_loader (DataLoader): Validation data loader
                num_epochs (int): Number of epochs to train
                checkpoint_path (str): Path to save model checkpoints
                early_stopping_patience (int): Number of epochs to wait for improvement
                
            Returns:
                dict: Training history containing losses and metrics
        """
        # Initialize variables for early stopping
        best_val_f1 = 0
        patience_counter = 0
        
        # Initialize history dictionary
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_metrics': []
        }
        
        for epoch in range(1, num_epochs + 1):
            # Train for one epoch
            train_loss, train_acc = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss, val_metrics = self.validate(val_loader)
            
            # Update learning rate if scheduler is provided
            if self.scheduler is not None:
                self.scheduler.step(val_loss)
            
            # Print epoch results
            print(
                f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, '
                f'Val F1: {val_metrics["f1"]:.4f}'
            )
            
            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Check if this is the best model so far
            if val_metrics['f1'] > best_val_f1:
                best_val_f1 = val_metrics['f1']
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                }, f'{checkpoint_path}/best_model.pth')
                
                print(f'New best model saved! F1: {val_metrics["f1"]:.4f}')
            else:
                patience_counter += 1
                
                # Check for early stopping
                if patience_counter >= early_stopping_patience:
                    print(f'Early stopping triggered after {epoch} epochs')
                    break
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_metrics': val_metrics,
                }, f'{checkpoint_path}/checkpoint_epoch_{epoch}.pth')
        
        return history


def train_flare_detection_model(
    train_loader, 
    val_loader, 
    input_dim=3, 
    device='cuda',
    d_model=256, 
    num_heads=8, 
    num_layers=2, 
    d_ff=512, 
    max_seq_len=100, 
    dropout=0.1, 
    lr=3e-4, 
    weight_decay=1e-5,
    num_epochs=100, 
    checkpoint_path='./checkpoints'
):
    """
        Complete training procedure
        
        Arguments:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            input_dim (int): Dimensionality of input features per time step
            device (str): Device to use for training ('cuda' or 'cpu')
            d_model (int): Model dimensionality
            num_heads (int): Number of attention heads
            num_layers (int): Number of encoder layers
            d_ff (int): Feed-forward network dimensionality
            max_seq_len (int): Maximum sequence length
            dropout (float): Dropout rate
            lr (float): Learning rate
            weight_decay (float): Weight decay for regularization
            num_epochs (int): Number of epochs to train
            checkpoint_path (str): Path to save model checkpoints
            
        Returns:
            tuple: (Trained model, training history)
    """
    from transformer_model import create_model
    from loss import WeightedBCELoss, get_class_weights
    
    # Create model
    model = create_model(
        input_dim=input_dim, 
        d_model=d_model, 
        num_heads=num_heads, 
        num_layers=num_layers,
        d_ff=d_ff, 
        max_seq_len=max_seq_len, 
        dropout=dropout
    )
    
    # Calculate class weights
    class_weights = get_class_weights(train_loader).to(device)
    
    # Define loss function
    criterion = WeightedBCELoss(class_weights)
    
    # Define optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr, 
        weight_decay=weight_decay
    )
    
    # Define learning rate scheduler (ReduceLROnPlateau as specified in thesis)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Create trainer
    trainer = Trainer(model, optimizer, criterion, device, scheduler)
    
    # Train model
    history = trainer.train(train_loader, val_loader, num_epochs, checkpoint_path)
    
    return model, history