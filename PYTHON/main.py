# Importing all needed libraries.
from config import config_fun
from flower_scanner import ApplyFlowerDetector
from serial_communication import ArduinoCommunicator

# Getting the config dictionary:
CONFIG = config_fun()

# Creating the Detector.
detector = ApplyFlowerDetector(**CONFIG['model'])

# Creating the Arduino Interface.
arduino_interface = ArduinoCommunicator(**CONFIG['communication'])

# Getting the image.
img_path = r"D:\PBL 2022\tets_flowers\test1.jpg"

# Getting the prediction from the image.
number_of_flowers = detector.predict_apple_flowers(img_path)

while True:
    # Calculating what type of solution to and how much.
    arduino_interface.send(int(CONFIG['general']['apple_flowers_threshold']) - number_of_flowers)
    #print(int(CONFIG['general']['apple_flowers_threshold']) - number_of_flowers)
