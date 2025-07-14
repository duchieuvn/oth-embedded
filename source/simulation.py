import pygame
import numpy as np
import pandas as pd

# === Load IMU CSV Data ===
imu_df = pd.read_csv("position_estimation.csv")
accel_x_series = imu_df['accel_x'].to_numpy()
accel_y_series = imu_df['accel_y'].to_numpy()

# === Map Parameters ===
MAP_WIDTH_M = 13.2
MAP_HEIGHT_M = 12.5
PIXEL_PER_METER = 50  # 10cm = 1 pixel
MAP_WIDTH_PX = int(MAP_WIDTH_M * PIXEL_PER_METER)
MAP_HEIGHT_PX = int(MAP_HEIGHT_M * PIXEL_PER_METER)
SCALE = 1

# === Pygame Init ===
WINDOW_WIDTH = MAP_WIDTH_PX * SCALE
WINDOW_HEIGHT = MAP_HEIGHT_PX * SCALE
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2D Particle Filter from CSV IMU")
clock = pygame.time.Clock()

# === Colors ===
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (100, 100, 100)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

# === Filter Parameters ===
NUM_PARTICLES = 100
SENSOR_NOISE = 20.0
MOVE_NOISE = 0.2
DT = 1.0 / 30.0  # 30Hz

# === Robot State: [x, y, theta, vx, vy] ===
robot = np.array([MAP_WIDTH_PX / 2, MAP_HEIGHT_PX / 2, 0.0, 0.0, 0.0])

# === Initialize Particles: [x, y, theta, weight] ===
particles = np.empty((NUM_PARTICLES, 4))
particles[:, 0] = np.random.uniform(0, MAP_WIDTH_PX, NUM_PARTICLES)
particles[:, 1] = np.random.uniform(0, MAP_HEIGHT_PX, NUM_PARTICLES)
particles[:, 2] = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
particles[:, 3] = 1.0 / NUM_PARTICLES

def move_with_accel(state, accel):
    x, y, theta, vx, vy = state
    ax, ay = accel
    vx += ax * DT
    vy += ay * DT
    x += vx * DT
    y += vy * DT
    return np.array([x, y, theta, vx, vy])

def move_particles(particles, ax, ay):
    ax_n = ax + np.random.normal(0, MOVE_NOISE, NUM_PARTICLES)
    ay_n = ay + np.random.normal(0, MOVE_NOISE, NUM_PARTICLES)
    vx = ax_n * DT
    vy = ay_n * DT
    particles[:, 0] += vx * DT
    particles[:, 1] += vy * DT

def sense(state):
    x = state[0] + np.random.normal(0, SENSOR_NOISE)
    y = state[1] + np.random.normal(0, SENSOR_NOISE)
    return np.array([x, y])

def update_weights(particles, measurement):
    dx = particles[:, 0] - measurement[0]
    dy = particles[:, 1] - measurement[1]
    distances = np.sqrt(dx**2 + dy**2)
    weights = np.exp(-distances**2 / (2 * SENSOR_NOISE**2))
    weights += 1e-300
    weights /= np.sum(weights)
    particles[:, 3] = weights

def resample(particles):
    weights = particles[:, 3]
    indices = np.random.choice(NUM_PARTICLES, NUM_PARTICLES, p=weights)
    resampled = particles[indices].copy()
    resampled[:, 3] = 1.0 / NUM_PARTICLES
    return resampled

def estimate(particles):
    x = np.average(particles[:, 0], weights=particles[:, 3])
    y = np.average(particles[:, 1], weights=particles[:, 3])
    return np.array([x, y])

# === Main Loop ===
i = 0
running = True
while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if i >= len(accel_x_series):
        break

    # === Acceleration from CSV ===
    ax = accel_x_series[i] * PIXEL_PER_METER  # convert g → pixel/s²
    ay = accel_y_series[i] * PIXEL_PER_METER
    i += 1
    accel = np.array([ax, ay])

    # === Move and Sense ===
    robot = move_with_accel(robot, accel)
    z = sense(robot)

    # === Particle Filter ===
    move_particles(particles, ax, ay)
    update_weights(particles, z)
    particles = resample(particles)
    est = estimate(particles)
    print (f"Estimated Position: {est}")

    # === Draw Map Border ===
    border_rect = pygame.Rect(0, 0, MAP_WIDTH_PX * SCALE, MAP_HEIGHT_PX * SCALE)
    pygame.draw.rect(screen, BLACK, border_rect, 2)

    # === Draw Particles ===
    for p in particles:
        px, py = int(p[0] * SCALE), int(p[1] * SCALE)
        pygame.draw.circle(screen, RED, (px, py), 1)

    # === Draw Measurement (blue) ===
    mx, my = int(z[0] * SCALE), int(z[1] * SCALE)
    pygame.draw.circle(screen, BLUE, (mx, my), 5)

    # === Draw True Robot (red) ===
    rx, ry = int(robot[0] * SCALE), int(robot[1] * SCALE)
    pygame.draw.circle(screen, RED, (rx, ry), 5)

    # === Draw Estimated Position (yellow) ===
    ex, ey = int(est[0] * SCALE), int(est[1] * SCALE)
    pygame.draw.circle(screen, YELLOW, (ex, ey), 5)

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
