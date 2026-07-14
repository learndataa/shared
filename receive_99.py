import serial
import glob
import time

ports = glob.glob('/dev/cu.usbmodem*')
if not ports:
    print("ST-LINK not found. Check connection.")
    exit()

ser = serial.Serial(ports[0], 9600, timeout=2)
print("━" * 40)
print("  Hardware Conversations — Episode 1")
print("━" * 40)
print(f"  Connected: {ports[0]}")
print(f"  Baud rate: 9600")
print("━" * 40)
print()
print("  STM32 is talking to Python...")
print()

count = 0
while True:
    line = ser.readline().decode('utf-8').strip()
    if line:
        count += 1
        print(f"  Message {count:3d}: STM32 says → {line}")