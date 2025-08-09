import serial       # package name: "pyserial", but use "import serial"
import time



class ArduinoCommunicator:
    """
    A class to manage a persistent serial connection with an Arduino.
    It handles connecting, sending messages, and closing the connection gracefully.
    """
    def __init__(self, port, baud_rate=115200, timeout=1):
        """
        Initializes the communicator.

        Args:
            port (str): The COM port the Arduino is connected to (e.g., 'COM3').
            baud_rate (int): The baud rate, must match the Arduino sketch.
            timeout (int): Read timeout in seconds.
        """
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self.ser = None

    def connect(self):
        """
        Establishes the serial connection with the Arduino.
        """
        try:
            print(f"Attempting to connect to Arduino on {self.port}...")
            self.ser = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)
            # Wait for the Arduino to reset after establishing the connection
            time.sleep(2)
            print("Successfully connected to Arduino.")
            return True
        except serial.SerialException as e:
            print(f"Error: Could not connect to serial port {self.port}.")
            print(f"Details: {e}")
            print("Please check if the Arduino is connected and the port is correct.")
            self.ser = None
            return False

    def send(self, message):
        """
        Sends a string message to the Arduino.

        Args:
            message (str): The message to send. Should end with a newline character.

        Returns:
            bool: True if the message was sent successfully, False otherwise.
        """
        if self.ser and self.ser.is_open:
            try:
                encoded_message = message.encode('utf-8')
                self.ser.write(encoded_message)
                return True
            except serial.SerialException as e:
                print(f"Error writing to serial port: {e}")
                return False
        else:
            print("Serial port not connected. Cannot send message.")
            return False

    def close(self):
        """
        Closes the serial connection if it is open.
        """
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("Arduino serial port closed.")



if __name__ == '__main__':
    # This is an example of how to use the ArduinoCommunicator class
    # --- IMPORTANT ---
    # You MUST change 'COM3' to the correct port for your system.
    arduino_port = 'COM5'
    communicator = ArduinoCommunicator(port=arduino_port)

    if communicator.connect():
        try:
            # Send a message every 2 seconds for a short duration
            for i in range(5):
                msg = f"Ping! Count: {i + 1}\n"
                print(f"Sending: '{msg.strip()}'")
                communicator.send(msg)
                time.sleep(2)
        finally:
            # Ensure the connection is closed when the script is done
            communicator.close()
    else:
        print("Could not establish connection with Arduino. Exiting example.")