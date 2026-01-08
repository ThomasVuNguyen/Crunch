import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import numpy as np


class TrainingPlotter:
    """Real-time training curve visualization with live saving."""

    def __init__(self, save_path="training_curves.png"):
        self.save_path = save_path
        self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 6))
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss (MSE)')
        self.ax.set_title('Training Progress: Sensor -> Image')
        self.ax.grid(True, alpha=0.3)

    def update(self, losses):
        """Update and save the plot with new loss values."""
        self.ax.clear()
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss (MSE)')
        self.ax.set_title('Training Progress: Sensor -> Image')
        self.ax.grid(True, alpha=0.3)

        epochs = list(range(1, len(losses) + 1))
        self.ax.plot(epochs, losses, 'b-', linewidth=2, label='Train Loss')
        self.ax.legend()

        # Save immediately for real-time viewing
        self.fig.savefig(self.save_path, dpi=150, bbox_inches='tight')

    def close(self):
        plt.close(self.fig)


def plot_predictions(model, sensor_hub, num_samples=4, save_path="predictions.png", device="cpu"):
    """Visualize model predictions vs actual camera images."""
    import torch

    model.eval()
    fig, axes = plt.subplots(num_samples, 2, figsize=(8, 4 * num_samples))

    if num_samples == 1:
        axes = axes.reshape(1, 2)

    with torch.no_grad():
        for i in range(num_samples):
            sensors, actual_image = sensor_hub.get_sample()

            if actual_image is None:
                continue

            # Predict
            x = torch.tensor(sensors, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model(x)
            pred_image = pred.squeeze().permute(1, 2, 0).cpu().numpy()

            # Denormalize sensor values for display
            dist_cm = sensors[0] * 398 + 2

            # Plot actual
            axes[i, 0].imshow(actual_image)
            axes[i, 0].set_title(f"Actual (d={dist_cm:.0f}cm, m={sensors[1]:.0f}, s={sensors[2]:.0f})")
            axes[i, 0].axis('off')

            # Plot predicted
            axes[i, 1].imshow(np.clip(pred_image, 0, 1))
            axes[i, 1].set_title("Predicted")
            axes[i, 1].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved predictions to {save_path}")
