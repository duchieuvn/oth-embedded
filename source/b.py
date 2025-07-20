import pygame
import pandas as pd

# === Load data ===
df = pd.read_csv("data_estimated.csv")  # Thay bằng tên file CSV của bạn
positions = df[['pos_x_world', 'pos_z_world']].to_numpy()

# === Thông số mô phỏng ===
PIXELS_PER_METER = 50     # Scale chuyển đổi mét → pixel
WIDTH_M = 20             # Kích thước vùng mô phỏng theo mét
HEIGHT_M = 12

# === Kích thước cửa sổ theo pixel ===
WIDTH = int(WIDTH_M * PIXELS_PER_METER)
HEIGHT = int(HEIGHT_M * PIXELS_PER_METER)

# === Pygame setup ===
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Absolute Position Simulation")
clock = pygame.time.Clock()

# === World to screen (không đảo trục) ===
def world_to_screen(x, y):
    return int(x * PIXELS_PER_METER), int(y * PIXELS_PER_METER)

# === Main loop ===
running = True
index = 0
trail = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if index < len(positions):
        x, y = positions[index]
        point = world_to_screen(4, y)
        trail.append(point)
        index += 1

    screen.fill((30, 30, 30))

    for point in trail:
        pygame.draw.circle(screen, (0, 255, 0), point, 4)

    if trail:
        pygame.draw.circle(screen, (255, 0, 0), trail[-1], 6)

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
