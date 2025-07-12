import pygame
import numpy as np
import pandas as pd

# === Load IMU Data and Compute Velocity & Position ===
imu_df = pd.read_csv("position_estimation.csv")
timestamp = imu_df['timestamp'].to_numpy() / 1000.0
accel_x = imu_df['accel_x'].to_numpy()
accel_y = imu_df['accel_y'].to_numpy()

dt = np.diff(timestamp, prepend=timestamp[0])
vel_x = np.zeros_like(accel_x)
vel_y = np.zeros_like(accel_y)
pos_x = np.zeros_like(accel_x)
pos_y = np.zeros_like(accel_y)

vel_x[1:] = vel_x[:-1] + accel_x[1:] * dt[1:]
vel_y[1:] = vel_y[:-1] + accel_y[1:] * dt[1:]
pos_x[1:] = pos_x[:-1] + vel_x[1:] * dt[1:]
pos_y[1:] = pos_y[:-1] + vel_y[1:] * dt[1:]

# === Map Parameters ===
PIXEL_PER_METER = 50
MAP_WIDTH_PX = int(13.2 * PIXEL_PER_METER)
MAP_HEIGHT_PX = int(12.5 * PIXEL_PER_METER)
SCALE = 1

# === Pygame Init ===
pygame.init()
screen = pygame.display.set_mode((MAP_WIDTH_PX, MAP_HEIGHT_PX))
pygame.display.set_caption("2D Particle Filter from Measurement")
clock = pygame.time.Clock()

# === Colors ===
WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

# === Particle Filter Params ===
NUM_PARTICLES = 100
SENSOR_NOISE = 8.0
MOVE_NOISE = 0.3

particles = np.empty((NUM_PARTICLES, 4))  # [x, y, theta, weight]
particles[:, 0] = np.random.uniform(0, MAP_WIDTH_PX, NUM_PARTICLES)
particles[:, 1] = np.random.uniform(0, MAP_HEIGHT_PX, NUM_PARTICLES)
particles[:, 2] = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
particles[:, 3] = 1.0 / NUM_PARTICLES

def move_particles(particles, vx, vy, dt):
    vx_n = vx + np.random.normal(0, MOVE_NOISE, NUM_PARTICLES)
    vy_n = vy + np.random.normal(0, MOVE_NOISE, NUM_PARTICLES)
    particles[:, 0] += vx_n * dt * PIXEL_PER_METER
    particles[:, 1] += vy_n * dt * PIXEL_PER_METER

def update_weights(particles, z):
    dx = particles[:, 0] - z[0]
    dy = particles[:, 1] - z[1]
    dist2 = dx**2 + dy**2
    weights = np.exp(-dist2 / (2 * SENSOR_NOISE**2))
    weights += 1e-300
    weights /= np.sum(weights)
    particles[:, 3] = weights

def resample(particles):
    indices = np.random.choice(NUM_PARTICLES, NUM_PARTICLES, p=particles[:, 3])
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

    if i >= len(timestamp):
        break

    # Measurement (converted to pixels)
    meas_x = pos_x[i] * PIXEL_PER_METER
    meas_y = pos_y[i] * PIXEL_PER_METER
    vx = vel_x[i]
    vy = vel_y[i]
    dt_i = dt[i]
    i += 1

    move_particles(particles, vx, vy, dt_i)
    update_weights(particles, [meas_x, meas_y])
    particles = resample(particles)
    est = estimate(particles)

    # Draw map border
    pygame.draw.rect(screen, BLACK, pygame.Rect(0, 0, MAP_WIDTH_PX, MAP_HEIGHT_PX), 2)

    # Draw particles
    for p in particles:
        pygame.draw.circle(screen, GRAY, (int(p[0]), int(p[1])), 1)

    # Draw measurement
    pygame.draw.circle(screen, BLUE, (int(meas_x), int(meas_y)), 5)

    # Draw estimated position
    pygame.draw.circle(screen, YELLOW, (int(est[0]), int(est[1])), 5)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
