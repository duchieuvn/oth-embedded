import pygame
import pandas as pd
import numpy as np

# Load dữ liệu
df = pd.read_csv("data_record/data.csv")
acc_y = df['acc_y'].to_numpy()
acc_z = df['acc_z'].to_numpy()

# Pygame setup
WIDTH, HEIGHT = 600, 600
CENTER = (WIDTH // 2, HEIGHT // 2)
SCALE = 150  # Phóng đại vector để nhìn rõ
FPS = 10     # Số frame/giây

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Acceleration Vector (Z-Y Plane)")
clock = pygame.time.Clock()

# Màu
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)
RED = (255, 0, 0)

# Vẽ mũi tên từ start đến end (với đầu mũi tên)
def draw_arrow(surface, start, end, color=(255, 0, 0), arrow_size=10):
    pygame.draw.line(surface, color, start, end, 3)
    # Tính vector hướng
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = np.arctan2(dy, dx)
    # Vẽ đầu mũi tên (2 nhánh)
    left = (
        end[0] - arrow_size * np.cos(angle - np.pi / 6),
        end[1] - arrow_size * np.sin(angle - np.pi / 6)
    )
    right = (
        end[0] - arrow_size * np.cos(angle + np.pi / 6),
        end[1] - arrow_size * np.sin(angle + np.pi / 6)
    )
    pygame.draw.line(surface, color, end, left, 3)
    pygame.draw.line(surface, color, end, right, 3)

# Vòng lặp hiển thị từng vector như mũi tên
i = 0
running = True
while running and i < len(acc_y):
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Vẽ trục toạ độ
    pygame.draw.line(screen, GRAY, (CENTER[0], 0), (CENTER[0], HEIGHT), 1)  # Z (thẳng đứng)
    pygame.draw.line(screen, GRAY, (0, CENTER[1]), (WIDTH, CENTER[1]), 1)   # Y (nằm ngang)

    # Lấy giá trị hiện tại
    ay = acc_y[i]
    az = acc_z[i]

    # Tính vị trí đầu mũi tên
    end_x = CENTER[0] + int(az * SCALE)
    end_y = CENTER[1] - int(ay * SCALE)

    # Vẽ mũi tên
    draw_arrow(screen, CENTER, (end_x, end_y), RED)

    pygame.display.flip()
    clock.tick(FPS)
    i += 1

pygame.quit()
