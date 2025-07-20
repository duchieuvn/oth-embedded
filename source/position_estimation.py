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


# === Load data ===
FILE_PATH = 'data_record/data_main.csv'
def loadData(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)    
    return data

data = loadData(FILE_PATH)
data[:, [1,2,3]] *= 9.806  # Convert to m/s^2

df = pd.DataFrame(data[:,:-2], columns=['timestamp', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z',])
df = df[df['timestamp'] > 1800]
df['acc_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
df['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

timestamp = df['timestamp'].to_numpy()
dt = np.diff(timestamp, prepend=timestamp[0]) / 1000.0  # ms → s
df['dt'] = dt


# === Rotation matrix around Y ===
def rotation_matrix_y(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([
        [ c, 0,  s],
        [ 0, 1,  0],
        [-s, 0,  c]
    ])

def detect_peaks(df, feature='gyro_y'):
    gyro_y = df[feature].to_numpy()
    dt = df['dt'].to_numpy()

    
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
    return peaks

def acceleration_to_world(df):
    gyro_y = df['gyro_z'].to_numpy()
    acc_x = df['acc_x'].to_numpy()
    acc_y = df['acc_y'].to_numpy()
    acc_z = df['acc_z'].to_numpy()
    timestamp = df['timestamp'].to_numpy()
    dt = np.diff(timestamp, prepend=timestamp[0]) / 1000.0  # ms → s

    # === Tính delta yaw ===
    delta_yaw = gyro_y * dt

    yaw = np.zeros_like(delta_yaw)
    yaw[0] = np.deg2rad(-40)

    for i in range(1, len(yaw)):
        yaw[i] = yaw[i-1]  + delta_yaw[i] 

    acc_world = []

    for i in range(len(yaw)):
        R = rotation_matrix_y(yaw[i])  
        a_body = np.array([acc_x[i], acc_y[i], acc_z[i]])
        a_world = R @ a_body
        acc_world.append(a_world)

    acc_world = np.array(acc_world)  
    return acc_world

peaks = detect_peaks(df, feature='gyro_y')
end_steps = (peaks[:-1] + peaks[1:]) // 2  
end_steps = np.append(end_steps, -1)

acc_world = acceleration_to_world(df)
acc_world_2d = acc_world[:, [0, 2]]

vel_world = np.zeros_like(acc_world)
pos_world = np.zeros_like(acc_world)

pos_world[0] = [CENTER[0], 0]
vel_world[1:] = acc_world_2d[:-1] * dt[1:] 
pos_world[1:] = pos_world[:-1] + vel_world * dt[1:]


start_step = 0
running = True
step_count = 1
for end_step in end_steps:
    screen.fill(WHITE)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            exit()
    
    if step_count % 2 == 0:
        dt = (df['timestamp'][end_step] - df['timestamp'][start_step]) / 1000.0  # ms → s
        avg_acc = acc_world_2d[start_step:end_step].mean(axis=0)
        
        vel_x1 = vel_world_x[start_step] + avg_acc[0] * dt
        vel_y1 = vel_world_y[start_step] + avg_acc[1] * dt 
        print(f"Step {step_count}: vel_x1={vel_x1}, vel_x2={vel_world_x[end_step]}")
        start_step = end_step


    pygame.display.flip()
    clock.tick(2)
    step_count += 1
    

pygame.quit()
