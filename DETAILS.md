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

Big Sound Sensor:

+ → Pin 2 (5V) - shared with HC-SR04
G → Pin 14 (Ground)
D0 → GPIO 27 (Pin 13)
A0 → ADS1115 A1

USB Camera:

Connected via USB port

ADS1115 ADC:

VDD → Pin 1 (3.3V)
GND → Pin 6 (Ground)
SCL → GPIO 3 (Pin 5)
SDA → GPIO 2 (Pin 3)

Photo Resistor:

One leg → 3.3V
Other leg → ADS1115 A0 + 10kΩ resistor to GND

---

# Sensors for See Experiment

| Sensor | Data Type | Input |
|--------|-----------|-------|
| HC-SR04 Ultrasonic | Distance (cm) | Continuous |
| PIR Motion | Motion detected | Binary + temporal |
| Sound Sensor D0 | Loud sound | Binary |
| Sound Sensor A0 | Sound level | Continuous (via ADC) |
| Photo Resistor | Light level | Continuous (via ADC) |
| USB Camera | Target image | 64x64 RGB |

# Future Sensors (if needed)

| Sensor | Use Case | Value |
|--------|----------|-------|
| Avoid sensor | Second distance reading (different angle) | Medium |
| KY-033 Line Tracking | IR reflectance - surface/floor detection | Low-Medium |
| Temp & humid sensor | Environmental context (DHT11/22) | Low |
| Light blocking | Detects obstructions | Low |
