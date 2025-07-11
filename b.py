import pygame
import pandas as pd
import numpy as np
import math
import time

# Load data
df = pd.read_csv("position_estimation.csv")

# Tính yaw (radian) từ magnetometer
df['yaw'] = np.arctan2(-df['mag_y'], df['mag_x'])

# Convert sang độ nếu cần (tùy chọn):
# df['yaw_deg'] = (np.degrees(df['yaw']) + 360) % 360

# Khởi tạo pygame
pygame.init()
WIDTH, HEIGHT = 600, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Heading Simulation (Magnetometer)")

# Màu
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED   = (255, 0, 0)

# Tâm và chiều dài mũi tên
center = (WIDTH // 2, HEIGHT // 2)
arrow_length = 100

clock = pygame.time.Clock()

i = 0
running = True
while running and i < len(df):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear screen
    screen.fill(WHITE)

    # Lấy góc yaw tại dòng i
    yaw = df.loc[i, 'yaw']  # radians
    print(f"Yaw at index {i}: {yaw} radians")

    # Tính điểm đầu và cuối của mũi tên
    x_end = center[0] + arrow_length * np.cos(yaw)
    y_end = center[1] + arrow_length * np.sin(yaw)

    # Vẽ mũi tên
    pygame.draw.line(screen, RED, center, (x_end, y_end), 5)
    pygame.draw.circle(screen, BLACK, center, 5)

    # Hiển thị
    pygame.display.flip()

    # Chờ 150 ms mỗi khung
    clock.tick(120)  # ~6 FPS
    i += 1

pygame.quit()
