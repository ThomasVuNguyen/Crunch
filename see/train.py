#!/usr/bin/env python3
"""Training script with live sensor data collection."""

import os
import time
import subprocess
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    BATCH_SIZE, LEARNING_RATE, NUM_EPOCHS,
    SAMPLES_PER_EPOCH, SAMPLE_DELAY,
    CHECKPOINT_DIR, SAVE_EVERY, PLOT_EVERY, DEVICE
)
from sensors import SensorHub
from model import SensorToImageMLP
from visualize import TrainingPlotter, plot_predictions


def display_in_terminal(actual_img, pred_img, sensors, loss, epoch, sample_idx, best_loss=None):
    """Display images and stats in terminal using chafa."""
    # Convert to uint8
    actual_uint8 = (actual_img * 255).astype(np.uint8)
    pred_uint8 = (np.clip(pred_img, 0, 1) * 255).astype(np.uint8)

    # Combine images side by side: [ACTUAL | PREDICTED]
    combined = np.hstack([actual_uint8, pred_uint8])
    cv2.imwrite('/tmp/train_combined.png', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

    # Denormalize sensor values
    dist_cm = sensors[0] * 398 + 2

    # Clear screen and display
    print("\033[2J\033[H", end="")  # Clear screen
    print("=" * 60)
    print(f"  TRAINING - Epoch {epoch+1}/{NUM_EPOCHS} | Sample {sample_idx+1}/{SAMPLES_PER_EPOCH}")
    print("=" * 60)
    print(f"  Loss: {loss:.6f}", end="")
    if best_loss is not None and best_loss < float('inf'):
        print(f"  |  Best: {best_loss:.6f}")
    else:
        print()
    print(f"  Distance: {dist_cm:.1f} cm  |  Motion: {int(sensors[1])}  |  Sound: {int(sensors[2])}")
    print("-" * 60)
    print("         ACTUAL                      PREDICTED")
    subprocess.run(["chafa", "--size=60x20", "/tmp/train_combined.png"], check=False)
    print("[Ctrl+C to stop]  [Plot: training_curves.png]")


def save_checkpoint(model, optimizer, epoch, loss, losses, path):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'losses': losses,
    }
    torch.save(checkpoint, path)


def train():
    # Create checkpoint directory
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Initialize
    print("Initializing sensors...")
    sensor_hub = SensorHub()

    print("Initializing model...")
    model = SensorToImageMLP()
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    plotter = TrainingPlotter()

    # Training state
    losses = []
    best_loss = float('inf')

    print(f"\n{'='*50}")
    print(f"Training Configuration")
    print(f"{'='*50}")
    print(f"Model parameters: {model.count_parameters():,}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Samples per epoch: {SAMPLES_PER_EPOCH}")
    print(f"Device: {DEVICE}")
    print(f"{'='*50}\n")

    print("Starting training with live data collection...")
    print("Press Ctrl+C to stop.\n")

    try:
        for epoch in range(NUM_EPOCHS):
            epoch_start = time.time()
            epoch_losses = []
            batch_sensors = []
            batch_images = []
            samples_collected = 0

            model.train()

            for sample_idx in range(SAMPLES_PER_EPOCH):
                # Collect live sample
                sensors, image = sensor_hub.get_sample()

                if sensors is None or image is None:
                    continue

                batch_sensors.append(sensors)
                batch_images.append(image)
                samples_collected += 1

                # Train when batch is full
                if len(batch_sensors) >= BATCH_SIZE:
                    # Convert to tensors
                    x = torch.tensor(np.array(batch_sensors), dtype=torch.float32).to(DEVICE)
                    y = torch.tensor(np.array(batch_images), dtype=torch.float32).to(DEVICE)
                    y = y.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

                    # Forward pass
                    pred = model(x)
                    loss = criterion(pred, y)

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    current_loss = loss.item()
                    epoch_losses.append(current_loss)

                    # Display in terminal (show last sample from batch)
                    pred_img = pred[-1].detach().permute(1, 2, 0).cpu().numpy()
                    display_in_terminal(
                        batch_images[-1], pred_img, batch_sensors[-1],
                        current_loss, epoch, sample_idx, best_loss
                    )

                    # Clear batch
                    batch_sensors = []
                    batch_images = []

                time.sleep(SAMPLE_DELAY)

            # Epoch summary
            epoch_time = time.time() - epoch_start
            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            losses.append(avg_loss)

            # Update plot after each epoch (real-time)
            if (epoch + 1) % PLOT_EVERY == 0:
                plotter.update(losses)

            # Save checkpoint periodically
            if (epoch + 1) % SAVE_EVERY == 0:
                path = f"{CHECKPOINT_DIR}/checkpoint_epoch_{epoch+1}.pt"
                save_checkpoint(model, optimizer, epoch, avg_loss, losses, path)

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                path = f"{CHECKPOINT_DIR}/best_model.pt"
                save_checkpoint(model, optimizer, epoch, avg_loss, losses, path)

    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")

    finally:
        # Save final state
        print("\nSaving final model...")
        path = f"{CHECKPOINT_DIR}/final_model.pt"
        save_checkpoint(model, optimizer, epoch, losses[-1] if losses else 0, losses, path)

        # Final plot
        plotter.update(losses)
        plotter.close()

        # Generate sample predictions
        print("Generating sample predictions...")
        model.eval()
        plot_predictions(model, sensor_hub, num_samples=4, device=DEVICE)

        # Cleanup
        sensor_hub.close()

        print(f"\nTraining complete!")
        print(f"Best loss: {best_loss:.6f}")
        print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")
        print(f"Training curves saved to: training_curves.png")
        print(f"Sample predictions saved to: predictions.png")


if __name__ == "__main__":
    train()
