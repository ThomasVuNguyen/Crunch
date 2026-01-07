# Electrical connections

Computer: Raspberry Pi 5 8GB

HC-SR04 Ultrasonic Sensor:

VCC → Pin 2 (5V)
Trig → GPIO 23 (Pin 16)
Echo → GPIO 24 (Pin 18) with voltage divider
GND → Pin 6 (Ground)

PIR Motion Sensor:

VCC → Pin 4 (5V)
OUT → GPIO 17 (Pin 11)
GND → Pin 9 (Ground)

Big Sound Sensor (Digital Only):

+ → Pin 2 (5V) - shared with HC-SR04
G → Pin 14 (Ground)
D0 → GPIO 27 (Pin 13)
A0 → Leave unconnected
