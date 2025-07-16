import pygame
import numpy as np
import pandas as pd
import time

# === Pygame init ===
WIDTH, HEIGHT = 600, 600
CENTER = (WIDTH // 2, HEIGHT // 2)
ARROW_LENGTH = 100

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("2 Arrows - Different Initial Yaw")
clock = pygame.time.Clock()

# === Colors ===
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# === Rotation matrix around Y ===
def rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c]
    ])

# === Load data and hoán đổi trục ===
df = pd.read_csv("data_record/data.csv")  # chứa 'timestamp', 'gyro_y', 'gyro_z'

gyro_y = df['gyro_y'].to_numpy()
timestamp = df['timestamp'].to_numpy()
dt = np.diff(timestamp, prepend=timestamp[0]) / 1000.0  # ms → s

# === Tính delta yaw ===
delta_yaw = gyro_y * dt

# === Hai chuỗi góc yaw với góc lệch ban đầu khác nhau ===
yaw1 = np.zeros_like(delta_yaw)
yaw2 = np.zeros_like(delta_yaw)
yaw1[0] = np.deg2rad(-90)
yaw2[0] = np.deg2rad(-40)

yaw1[1:] = yaw1[:-1]  + delta_yaw[1:]  
yaw2[1:] = yaw2[:-1]  + delta_yaw[1:]

print(yaw1[0:90:15])
print(yaw2[0:90:15])

# === Main loop ===
i = 50
running = True
while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if i >= len(yaw1):
        break

    avg_delta = np.mean(delta_yaw[i-50:i])
    yaw1[i] = yaw1[i-50] + avg_delta
    yaw2[i] = yaw2[i-50] + avg_delta
    print(f"Yaw1[{i}]: {np.rad2deg(yaw1[i]):.2f}°, Yaw2[{i}]: {np.rad2deg(yaw2[i]):.2f}°")
    print(f"yaw1[i-1]: {np.rad2deg(yaw1[i-1]):.2f}°, yaw2[{i-1}]: {np.rad2deg(yaw2[i]):.2f}°")
    print("--------------------")
    vec_body = np.array([1, 0, 0])  # hướng trục x

    # Arrow 1 (BLUE) -90°
    R1 = rotation_matrix_y(yaw1[i])
    vec_world1 = R1 @ vec_body
    dx1, dz1 = vec_world1[0], vec_world1[2]
    end1 = (CENTER[0] + ARROW_LENGTH * dx1, CENTER[1] + ARROW_LENGTH * dz1)
    pygame.draw.line(screen, BLUE, CENTER, end1, 4)

    # Arrow 2 (RED) -40°
    R2 = rotation_matrix_y(yaw2[i])
    vec_world2 = R2 @ vec_body
    dx2, dz2 = vec_world2[0], vec_world2[2]
    end2 = (CENTER[0] + ARROW_LENGTH * dx2, CENTER[1] + ARROW_LENGTH * dz2)
    pygame.draw.line(screen, RED, CENTER, end2, 4)

    pygame.display.flip()
    clock.tick(10)
    i += 50

pygame.quit()
