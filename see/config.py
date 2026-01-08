# Sensor pins (from test.py)
ULTRASONIC_TRIG = 23
ULTRASONIC_ECHO = 24
PIR_PIN = 17
SOUND_PIN = 27

# Image settings
IMG_SIZE = 64
IMG_CHANNELS = 3

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 100
SAMPLES_PER_EPOCH = 500
SAMPLE_DELAY = 0.05  # 20 Hz

# Model settings
HIDDEN_DIMS = [128, 512, 2048, 4096]

# Checkpointing
CHECKPOINT_DIR = "checkpoints"
SAVE_EVERY = 10
PLOT_EVERY = 1  # Update plot every epoch for real-time feedback

# Device
DEVICE = "cpu"
