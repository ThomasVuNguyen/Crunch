#!/usr/bin/env python3
"""Run trained model for real-time inference."""

import argparse
import time
import subprocess
import numpy as np
import cv2
import torch

from config import CHECKPOINT_DIR, DEVICE
from sensors import SensorHub
from model import SensorToImageMLP


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    model = SensorToImageMLP()

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"Trained for {checkpoint['epoch']+1} epochs")
    print(f"Final loss: {checkpoint['loss']:.6f}")

    return model


def run_inference(checkpoint_path, display_mode='terminal', num_frames=None):
    """
    Run real-time inference.

    Args:
        checkpoint_path: Path to model checkpoint
        display_mode: 'terminal' (chafa), 'opencv', or 'save'
        num_frames: Number of frames to process (None = infinite)
    """
    model = load_model(checkpoint_path)
    sensor_hub = SensorHub()

    print(f"\nRunning inference in '{display_mode}' mode...")
    print("Press Ctrl+C to stop.\n")

    frame_count = 0

    try:
        while num_frames is None or frame_count < num_frames:
            # Get sensor data
            sensors, actual_image = sensor_hub.get_sample()

            if sensors is None or actual_image is None:
                continue

            # Predict
            with torch.no_grad():
                x = torch.tensor(sensors, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                pred = model(x)
                pred_image = pred.squeeze().permute(1, 2, 0).cpu().numpy()

            # Convert to uint8
            actual_uint8 = (actual_image * 255).astype(np.uint8)
            pred_uint8 = (np.clip(pred_image, 0, 1) * 255).astype(np.uint8)

            # Denormalize sensor values for display
            dist_cm = sensors[0] * 398 + 2

            if display_mode == 'terminal':
                # Combine images side by side
                combined = np.hstack([actual_uint8, pred_uint8])
                cv2.imwrite('/tmp/inf_combined.png', cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

                print("\033[2J\033[H", end="")  # Clear screen
                print("=" * 60)
                print(f"  INFERENCE - Frame {frame_count}")
                print("=" * 60)
                print(f"  Distance: {dist_cm:.1f} cm  |  Motion: {int(sensors[1])}  |  Sound: {int(sensors[2])}")
                print("-" * 60)
                print("         ACTUAL                      PREDICTED")
                subprocess.run(["chafa", "--size=60x20", "/tmp/inf_combined.png"], check=False)
                print("[Ctrl+C to stop]")

            elif display_mode == 'opencv':
                # Side-by-side comparison in window
                combined = np.hstack([actual_uint8, pred_uint8])
                combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)

                # Add text
                cv2.putText(combined_bgr, f"D:{dist_cm:.0f}cm M:{sensors[1]:.0f} S:{sensors[2]:.0f}",
                           (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                cv2.putText(combined_bgr, "Actual", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(combined_bgr, "Predicted", (74, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                cv2.imshow('Sensor -> Image (Press Q to quit)', combined_bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            elif display_mode == 'save':
                cv2.imwrite(f'inference_actual_{frame_count:04d}.png',
                           cv2.cvtColor(actual_uint8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(f'inference_pred_{frame_count:04d}.png',
                           cv2.cvtColor(pred_uint8, cv2.COLOR_RGB2BGR))
                print(f"Saved frame {frame_count}")

            frame_count += 1
            time.sleep(0.1)  # 10 Hz

    except KeyboardInterrupt:
        print("\n\nInference stopped.")

    finally:
        sensor_hub.close()
        if display_mode == 'opencv':
            cv2.destroyAllWindows()

    print(f"Processed {frame_count} frames")


def main():
    parser = argparse.ArgumentParser(description="Run inference with trained model")
    parser.add_argument('--checkpoint', '-c',
                       default=f'{CHECKPOINT_DIR}/best_model.pt',
                       help='Path to model checkpoint')
    parser.add_argument('--display', '-d',
                       choices=['terminal', 'opencv', 'save'],
                       default='terminal',
                       help='Display mode')
    parser.add_argument('--frames', '-n',
                       type=int, default=None,
                       help='Number of frames to process (default: infinite)')
    args = parser.parse_args()

    # Check if checkpoint exists, fall back to final_model.pt
    import os
    checkpoint = args.checkpoint
    if not os.path.exists(checkpoint):
        checkpoint = f'{CHECKPOINT_DIR}/final_model.pt'
    if not os.path.exists(checkpoint):
        print(f"No checkpoint found. Train a model first with: python train.py")
        return

    run_inference(checkpoint, args.display, args.frames)


if __name__ == "__main__":
    main()
