import gpiod
import time



# Define the GPIO pin number
pin = 17  # Example GPIO pin number

chip = gpiod.Chip('gpiochip4')

line = chip.get_line(pin)

line.request(consumer="Arduino", type=gpiod.LINE_REQ_DIR_OUT)



# Keep the output high for 10 seconds
line.set_value(1)
time.sleep(1)



line.release()
