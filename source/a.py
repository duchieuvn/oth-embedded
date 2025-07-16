import pygame
import numpy as np
import pandas as pd

# === Map Parameters ===
PIXEL_PER_METER = 50
MAP_WIDTH_PX = int(13.2 * PIXEL_PER_METER)
MAP_HEIGHT_PX = int(12.5 * PIXEL_PER_METER)

# === Load IMU Data and Compute Velocity & Position ===
imu_df = pd.read_csv("data_record/data.csv")
timestamp = imu_df['timestamp'].to_numpy() / 1000.0
acc_x = imu_df['acc_x'].to_numpy()
acc_y = imu_df['acc_y'].to_numpy()
acc_z = imu_df['acc_z'].to_numpy()

dt = np.diff(timestamp, prepend=timestamp[0])
vel_x = np.zeros_like(acc_x)
vel_z = np.zeros_like(acc_z)
pos_x = np.zeros_like(acc_x)
pos_z = np.zeros_like(acc_z)

pos_x[0] = 6.0
pos_z[0] = 6.0

for i in range(1, len(timestamp)):
    vel_x[i] = vel_x[i - 1] + acc_x[i] * dt[i]
    vel_z[i] = vel_z[i - 1] + acc_z[i] * dt[i]
    pos_x[i] = pos_x[i - 1] + vel_x[i] * dt[i]
    pos_z[i] = pos_z[i - 1] + vel_z[i] * dt[i]

# Save to CSV
output_df = pd.DataFrame({
    'timestamp': timestamp,
    'acc_x': acc_x,
    'acc_y': acc_y,
    'acc_z': acc_z,
    'vel_x': vel_x,
    'vel_z': vel_z,
    'pos_x': pos_x,
    'pos_z': pos_z
})
output_df.to_csv("processed_motion_data.csv", index=False)
print("Saved processed motion data to processed_motion_data.csv")

# === Pygame Init ===
pygame.init()
screen = pygame.display.set_mode((MAP_WIDTH_PX, MAP_HEIGHT_PX))
pygame.display.set_caption("2D Particle Filter on X-Z Plane")
clock = pygame.time.Clock()

WHITE = (255, 255, 255)
GRAY = (100, 100, 100)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

# === Particle Filter Params ===
NUM_PARTICLES = 100
SENSOR_NOISE = 8.0
MOVE_NOISE = 0.5
PARTICLE_STD = 12

particles = np.empty((NUM_PARTICLES, 4))  # [x, z, theta, weight]
particles[:, 0] = np.random.normal(loc=pos_x[0]*PIXEL_PER_METER, scale=PARTICLE_STD, size=NUM_PARTICLES)
particles[:, 1] = np.random.normal(loc=pos_z[0]*PIXEL_PER_METER, scale=PARTICLE_STD, size=NUM_PARTICLES)
particles[:, 2] = np.random.normal(0, 2 * np.pi, NUM_PARTICLES)
particles[:, 3] = 1.0 / NUM_PARTICLES

def move_particles(particles, vx, vz, dt):
    vx_n = vx + np.random.normal(0, MOVE_NOISE, NUM_PARTICLES)
    vz_n = vz + np.random.normal(0, MOVE_NOISE, NUM_PARTICLES)
    particles[:, 0] += vx_n * dt * PIXEL_PER_METER
    particles[:, 1] += vz_n * dt * PIXEL_PER_METER

def update_weights(particles, z):
    dx = particles[:, 0] - z[0]
    dz = particles[:, 1] - z[1]
    dist2 = dx**2 + dz**2
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
    z = np.average(particles[:, 1], weights=particles[:, 3])
    return np.array([x, z])

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

    meas_x = pos_x[i] * PIXEL_PER_METER
    meas_z = pos_z[i] * PIXEL_PER_METER
    vx = vel_x[i]
    vz = vel_z[i]
    dt_i = dt[i]
    i += 1

    move_particles(particles, vx, vz, dt_i)
    update_weights(particles, [meas_x, meas_z])
    particles = resample(particles)
    est = estimate(particles)

    pygame.draw.rect(screen, BLACK, pygame.Rect(0, 0, MAP_WIDTH_PX, MAP_HEIGHT_PX), 2)

    for p in particles:
        pygame.draw.circle(screen, GRAY, (int(p[0]), int(p[1])), 1)

    pygame.draw.circle(screen, BLUE, (int(meas_x), int(meas_z)), 5)
    pygame.draw.circle(screen, YELLOW, (int(est[0]), int(est[1])), 5)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
