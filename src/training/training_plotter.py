from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import torch


class TrainingPlotter:
    """
        Handles all plotting and visualization for training results.
        
        This class creates plots for training history, attention maps, and other
        visualizations to aid in understanding model behavior.
        
        Attributes:
            save_dir (Path): Directory to save plots
    """
    
    def __init__(self, save_dir):
        """
            Initializes the TrainingPlotter.
            
            Arguments:
                save_dir (Path): Directory to save generated plots
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, history):
        """
            Plot training and validation metrics over epochs.
            
            Arguments:
                history (dict): Training history containing loss and metric values
                
            Returns:
                None
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot training and validation loss
        axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
        axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot training and validation accuracy
        axes[0, 1].plot(history['train_acc'], label='Train Accuracy')
        axes[0, 1].plot(
            [m['accuracy'] for m in history['val_metrics']], 
            label='Validation Accuracy'
        )
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot validation precision and recall
        axes[1, 0].plot(
            [m['precision'] for m in history['val_metrics']], 
            label='Precision'
        )
        axes[1, 0].plot(
            [m['recall'] for m in history['val_metrics']], 
            label='Recall'
        )
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Validation Precision and Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot validation F1 score
        axes[1, 1].plot(
            [m['f1'] for m in history['val_metrics']], 
            label='F1 Score'
        )
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].set_title('Validation F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png')
        plt.close()
    
    def plot_attention_maps(self, model, data_loader, num_samples=5):
        """
            Plot attention maps for sample inputs.
            
            Arguments:
                model (FlareClassifier): Trained model
                data_loader (DataLoader): Data loader for samples
                num_samples (int): Number of samples to visualize
                
            Returns:
                None
        """
        device = next(model.parameters()).device
        model.eval()
        
        # Collect sample data (both flare and non-flare examples)
        flare_samples = []
        non_flare_samples = []
        
        for data, target, _ in data_loader:
            for i, t in enumerate(target):
                if t.item() == 1 and len(flare_samples) < num_samples:
                    flare_samples.append((data[i].numpy(), 1))
                elif t.item() == 0 and len(non_flare_samples) < num_samples:
                    non_flare_samples.append((data[i].numpy(), 0))
            
            if (len(flare_samples) >= num_samples and 
                len(non_flare_samples) >= num_samples):
                break
        
        samples = flare_samples + non_flare_samples
        
        for i, (sample, label) in enumerate(samples):
            self._plot_single_attention(model, sample, label, i+1, device)
    
    def _plot_single_attention(self, model, sample, label, sample_num, device):
        """
            Plot attention maps for a single sample.
            
            Arguments:
                model (FlareClassifier): Trained model
                sample (ndarray): Input sample
                label (int): True label (0 or 1)
                sample_num (int): Sample number for filename
                device: PyTorch device
                
            Returns:
                None
        """
        # Plot light curve
        plt.figure(figsize=(12, 4))
        plt.plot(sample.squeeze())
        plt.title(f"Sample {sample_num}: {'Flare' if label else 'No Flare'} Light Curve")
        plt.xlabel("Time Step")
        plt.ylabel("Normalized Flux")
        plt.grid(True)
        plt.savefig(self.save_dir / f"sample_{sample_num}_lightcurve.png")
        plt.close()
        
        # Convert to tensor
        sample_tensor = torch.tensor(
            sample, 
            dtype=torch.float32
        ).unsqueeze(0).to(device)
        
        # Get attention weights for each layer
        for layer_idx in range(len(model.transformer_encoder.layers)):
            fig, axes = plt.subplots(2, 4, figsize=(20, 10))
            fig.suptitle(
                f"Sample {sample_num} (Label: {'Flare' if label else 'No Flare'}) - "
                f"Layer {layer_idx+1} Attention Maps",
                fontsize=16
            )
            
            axes = axes.flatten()
            
            for head_idx in range(8):
                attn_weights = model.get_attention_maps(
                    sample_tensor, 
                    layer_idx, 
                    head_idx
                )
                
                if attn_weights is not None:
                    sns.heatmap(
                        attn_weights[0].cpu().numpy(),
                        ax=axes[head_idx],
                        cmap="viridis"
                    )
                    axes[head_idx].set_title(f"Head {head_idx+1}")
                    axes[head_idx].set_xlabel("Key position")
                    axes[head_idx].set_ylabel("Query position")
                else:
                    axes[head_idx].text(
                        0.5, 0.5,
                        "Attention weights not available",
                        horizontalalignment='center',
                        verticalalignment='center'
                    )
            
            plt.tight_layout()
            plt.savefig(
                self.save_dir / f"sample_{sample_num}_layer_{layer_idx+1}_attention.png"
            )
            plt.close()
