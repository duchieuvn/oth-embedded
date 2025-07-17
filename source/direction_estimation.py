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

# === Load data ===
df = pd.read_csv("data_record/data.csv")  
df = df.iloc[100:,:].reset_index(drop=True)  
gyro_y = df['gyro_y'].to_numpy()
timestamp = df['timestamp'].to_numpy()
dt = np.diff(timestamp, prepend=timestamp[0]) / 1000.0  # ms → s

# === Tính delta yaw ===
delta_yaw = gyro_y * dt

# === Hai chuỗi góc yaw với góc lệch ban đầu khác nhau ===
yaw = np.zeros_like(delta_yaw)
yaw[0] = np.deg2rad(-40)

yaw[1:] = yaw[:-1]  + delta_yaw[1:]  


PEAK_HEIGHT = 1.2  
PEAK_DISTANCE = 14

peaks = []
last_peak_idx = -PEAK_DISTANCE

df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
df['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)


acc_magnitude = df['acc_magnitude']

for i in range(1, len(acc_magnitude) - 1):
    if (
        acc_magnitude[i] > PEAK_HEIGHT and
        acc_magnitude[i] > acc_magnitude[i - 1] and
        acc_magnitude[i] > acc_magnitude[i + 1] and
        (i - last_peak_idx) >= PEAK_DISTANCE
    ):
        peaks.append(i)
        last_peak_idx = i

peaks = np.array(peaks)
end_steps = (peaks[:-1] + peaks[1:]) // 2  
end_steps = np.append(end_steps, -1)

# === Main loop ===
start_step = 0
running = True
count = 1
for end_step in end_steps:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()

    sum_delta = np.sum(delta_yaw[start_step:end_step])
    yaw[end_step] = yaw[start_step] + sum_delta
    vec_body = np.array([1, 0, 1]) 

    # Arrow 1 (BLUE) -90°
    R1 = rotation_matrix_y(yaw[end_step])
    vec_world1 = R1 @ vec_body
    dx1, dz1 = vec_world1[0], vec_world1[2]
    end_arrow = (CENTER[0] + ARROW_LENGTH * dx1, CENTER[1] + ARROW_LENGTH * dz1)
    pygame.draw.line(screen, BLUE, CENTER, end_arrow, 4)

    pygame.display.flip()
    clock.tick(2)
    start_step = end_step
    print(f"Step {count}")
    count += 1
    

pygame.quit()
