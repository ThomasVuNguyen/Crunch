import numpy as np
import cv2
from gpiozero import DistanceSensor, MotionSensor, DigitalInputDevice

from config import (
    ULTRASONIC_TRIG, ULTRASONIC_ECHO,
    PIR_PIN, SOUND_PIN, IMG_SIZE
)


class SensorHub:
    """Unified interface for all sensors and camera."""

    def __init__(self, camera_id=0, img_size=IMG_SIZE):
        self.img_size = img_size

        # Initialize sensors
        self.distance_sensor = DistanceSensor(
            echo=ULTRASONIC_ECHO,
            trigger=ULTRASONIC_TRIG,
            max_distance=4
        )
        self.pir_sensor = MotionSensor(PIR_PIN)
        self.sound_sensor = DigitalInputDevice(SOUND_PIN)

        # Initialize camera
        self.camera = cv2.VideoCapture(camera_id)
        if not self.camera.isOpened():
            raise RuntimeError("Could not open camera")

    def read_sensors(self):
        """Read raw sensor values.

        Returns:
            tuple: (distance_cm, pir_active, sound_active)
        """
        distance = self.distance_sensor.distance * 100  # Convert to cm
        pir = 1 if self.pir_sensor.motion_detected else 0
        sound = int(self.sound_sensor.value)
        return distance, pir, sound

    def read_camera(self):
        """Capture and resize camera frame.

        Returns:
            np.ndarray: RGB image of shape (img_size, img_size, 3) or None
        """
        ret, frame = self.camera.read()
        if ret and frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (self.img_size, self.img_size))
            return frame
        return None

    def get_sample(self):
        """Get normalized sensor input and image target.

        Returns:
            tuple: (sensors_array, image_array) or (None, None) on failure
        """
        distance, pir, sound = self.read_sensors()
        image = self.read_camera()

        if image is None:
            return None, None

        # Normalize distance: 2-400cm -> 0-1
        distance_norm = (distance - 2) / (400 - 2)
        distance_norm = np.clip(distance_norm, 0, 1)

        # Sensor vector: [distance_norm, pir, sound]
        sensors = np.array([distance_norm, float(pir), float(sound)], dtype=np.float32)

        # Normalize image to 0-1
        image = image.astype(np.float32) / 255.0

        return sensors, image

    def close(self):
        """Clean up resources."""
        self.distance_sensor.close()
        self.pir_sensor.close()
        self.sound_sensor.close()
        self.camera.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
