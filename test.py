#!/usr/bin/env python3
"""
Sensor Test Script for Raspberry Pi 5
Tests: HC-SR04 Ultrasonic, PIR Motion, Big Sound Sensor
Uses gpiozero (compatible with Pi 5)
"""

import time
import sys

try:
    from gpiozero import DistanceSensor, MotionSensor, DigitalInputDevice
    from gpiozero.exc import GPIOZeroError
except ImportError:
    print("ERROR: gpiozero not installed. Run: pip install gpiozero lgpio")
    sys.exit(1)

# GPIO Pin Configuration (BCM numbering)
ULTRASONIC_TRIG = 23
ULTRASONIC_ECHO = 24
PIR_PIN = 17
SOUND_PIN = 27


def test_ultrasonic():
    """Test HC-SR04 ultrasonic sensor."""
    print("\n" + "=" * 50)
    print("TESTING: HC-SR04 Ultrasonic Sensor")
    print("Pins: Trig=GPIO23, Echo=GPIO24")
    print("=" * 50)
    print("Taking 5 distance measurements...")
    print("(Place an object in front of the sensor)\n")

    try:
        sensor = DistanceSensor(echo=ULTRASONIC_ECHO, trigger=ULTRASONIC_TRIG, max_distance=4)
        success_count = 0

        for i in range(5):
            distance = sensor.distance * 100  # Convert to cm
            distance = round(distance, 2)

            if 2 < distance < 400:
                print(f"  Reading {i+1}: {distance} cm")
                success_count += 1
            else:
                print(f"  Reading {i+1}: {distance} cm (out of range)")

            time.sleep(0.3)

        sensor.close()

        if success_count >= 3:
            print("\n[PASS] Ultrasonic sensor is working!")
            return True
        else:
            print("\n[FAIL] Ultrasonic sensor may have issues")
            return False

    except GPIOZeroError as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_pir():
    """Test PIR motion sensor."""
    print("\n" + "=" * 50)
    print("TESTING: PIR Motion Sensor")
    print("Pin: GPIO17")
    print("=" * 50)
    print("Waiting for PIR sensor to stabilize (2 seconds)...")
    time.sleep(2)

    print("Wave your hand in front of the sensor!")
    print("Waiting up to 10 seconds for motion...\n")

    try:
        sensor = MotionSensor(PIR_PIN)
        motion_detected = sensor.wait_for_motion(timeout=10)

        sensor.close()

        if motion_detected:
            print("  Motion detected!")
            print("\n[PASS] PIR sensor is working!")
            return True
        else:
            print("\n[FAIL] No motion detected - check sensor or try again")
            return False

    except GPIOZeroError as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_sound():
    """Test Big Sound sensor (digital output)."""
    print("\n" + "=" * 50)
    print("TESTING: Big Sound Sensor (Digital)")
    print("Pin: GPIO27 (D0)")
    print("=" * 50)
    print("Make a loud noise (clap, snap, etc.)!")
    print("Waiting up to 10 seconds for sound...\n")

    try:
        sensor = DigitalInputDevice(SOUND_PIN)
        initial_state = sensor.value
        print(f"  Initial state: {'HIGH' if initial_state else 'LOW'}")

        sound_detected = False
        start_time = time.time()

        while time.time() - start_time < 10:
            if sensor.value != initial_state:
                print("  Sound detected!")
                sound_detected = True
                break
            time.sleep(0.01)

        sensor.close()

        if sound_detected:
            print("\n[PASS] Sound sensor is working!")
            return True
        else:
            print("\n[WARN] No sound detected")
            print("  - Adjust sensitivity potentiometer on the sensor")
            print("  - Or try making a louder noise")
            return False

    except GPIOZeroError as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def run_all_tests():
    """Run all sensor tests."""
    print("\n" + "#" * 50)
    print("#  RASPBERRY PI 5 SENSOR TEST SUITE")
    print("#" * 50)

    results = {
        "Ultrasonic (HC-SR04)": test_ultrasonic(),
        "PIR Motion": test_pir(),
        "Sound Sensor": test_sound(),
    }

    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)

    for sensor, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {sensor}: {status}")

    passed_count = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed_count}/{total} sensors passed")

    return all(results.values())


if __name__ == "__main__":
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)
