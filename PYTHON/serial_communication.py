import time

import serial

class ArduinoCommunicator:
    def __init__(self, **kwargs):
        self.port = kwargs['port']
        self.baudrate = int(kwargs['boudrate'])
        self.timeout = float(kwargs["timeout"])

        self.interface = serial.Serial(
            port = self.port,
            baudrate = self.baudrate,
            timeout = self.timeout
        )

    def send(self, data):
        self.interface.write(bytes(str(data), 'utf-8'))
        time.sleep(0.05)