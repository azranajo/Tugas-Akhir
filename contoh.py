import pygame
import numpy as np
from picamera2 import Picamera2
import cv2

pygame.init()
screen = pygame.display.set_mode((320, 240))  # ukuran LCD SPI Anda
picam2 = Picamera2()
picam2.configure(picam2.create_preview_configuration(main={"size": (320, 240)}))
picam2.start()

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(np.rot90(frame))  # rotasi jika perlu
    screen.blit(surface, (0, 0))
    pygame.display.update()
