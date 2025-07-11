import pygame
import numpy as np

# === Map Parameters ===
MAP_WIDTH_M = 13.2     # meters
MAP_HEIGHT_M = 12.5    # meters
PIXEL_PER_METER = 50   # 10 cm = 1 pixel
MAP_WIDTH_PX = int(MAP_WIDTH_M * PIXEL_PER_METER)     # 132
MAP_HEIGHT_PX = int(MAP_HEIGHT_M * PIXEL_PER_METER)   # 125
SCALE = 1  # visual scaling factor

# === Pygame Init ===
WINDOW_WIDTH = MAP_WIDTH_PX * SCALE
WINDOW_HEIGHT = MAP_HEIGHT_PX * SCALE
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("2D Particle Filter with Map Border")
clock = pygame.time.Clock()

# === Colors ===
WHITE = (255, 255, 255)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
GRAY  = (100, 100, 100)
YELLOW = (255, 255, 0)
BLACK = (0, 0, 0)

# === Filter Parameters ===
NUM_PARTICLES = 500
VELOCITY = 2.0
TURN_RATE = 0.02
SENSOR_NOISE = 20.0
MOVE_NOISE = 1.0
TURN_NOISE = 0.05

# === Robot State: [x, y, theta] in pixels (1 px = 10cm) ===
robot = np.array([MAP_WIDTH_PX / 2, MAP_HEIGHT_PX / 2, 0.0])

# === Initialize Particles: [x, y, theta, weight] ===
particles = np.empty((NUM_PARTICLES, 4))
particles[:, 0] = np.random.uniform(0, MAP_WIDTH_PX, NUM_PARTICLES)
particles[:, 1] = np.random.uniform(0, MAP_HEIGHT_PX, NUM_PARTICLES)
particles[:, 2] = np.random.uniform(0, 2 * np.pi, NUM_PARTICLES)
particles[:, 3] = 1.0 / NUM_PARTICLES

def move(state, v, w, noise=True):
    if noise:
        v += np.random.normal(0, MOVE_NOISE)
        w += np.random.normal(0, TURN_NOISE)
    theta = (state[2] + w) % (2 * np.pi)
    x = state[0] + v * np.cos(theta)
    y = state[1] + v * np.sin(theta)
    return np.array([x, y, theta])

def move_particles(particles, v, w):
    theta = (particles[:, 2] + np.random.normal(w, TURN_NOISE, NUM_PARTICLES)) % (2 * np.pi)
    v_noisy = np.random.normal(v, MOVE_NOISE, NUM_PARTICLES)
    particles[:, 0] += v_noisy * np.cos(theta)
    particles[:, 1] += v_noisy * np.sin(theta)
    particles[:, 2] = theta

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
running = True
while running:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Move robot and simulate sensor
    robot = move(robot, VELOCITY, TURN_RATE, noise=False)
    z = sense(robot)

    # Particle prediction and update
    move_particles(particles, VELOCITY, TURN_RATE)
    update_weights(particles, z)
    particles = resample(particles)

    # Estimate position
    est = estimate(particles)

    # Draw map border
    border_rect = pygame.Rect(0, 0, MAP_WIDTH_PX * SCALE, MAP_HEIGHT_PX * SCALE)
    pygame.draw.rect(screen, BLACK, border_rect, 2)

    # Draw particles (gray)
    for p in particles:
        px, py = int(p[0] * SCALE), int(p[1] * SCALE)
        pygame.draw.circle(screen, GRAY, (px, py), 1)

    # Draw measurement (blue)
    mx, my = int(z[0] * SCALE), int(z[1] * SCALE)
    pygame.draw.circle(screen, BLUE, (mx, my), 5)

    # Draw true robot position (red)
    rx, ry = int(robot[0] * SCALE), int(robot[1] * SCALE)
    pygame.draw.circle(screen, RED, (rx, ry), 5)

    # Draw estimated position (yellow)
    ex, ey = int(est[0] * SCALE), int(est[1] * SCALE)
    pygame.draw.circle(screen, YELLOW, (ex, ey), 5)

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
