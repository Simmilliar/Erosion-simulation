#%%
import numpy as np
from noise import perlin as pn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.gridspec as gridspec
import time
import math
from typing import NamedTuple

map_size = 256
np.random.seed(int(time.time()))

# Generate terrain
sn = pn.SimplexNoise()
terrain_map = np.empty((map_size, map_size))
for i in range(map_size):
    for j in range(map_size):
        terrain_map[i, j] = (1 + sn.noise2(x=j / 128, y=i / 128)) / 2 + \
                            (1 + sn.noise2(x=j / 64, y=i / 64)) / 6 + \
                            (1 + sn.noise2(x=j / 32, y=i / 32)) / 12 + \
                            (1 + sn.noise2(x=j / 16, y=i / 16)) / 32
terrain_map *= 64
terrain_map = np.transpose(terrain_map)


class HeightAndGradient(NamedTuple):
    height: float
    gradient_x: float
    gradient_y: float


def get_height_and_gradient(x: float, y: float):
    coord_ax = int(x - 0.001)
    coord_ay = int(y - 0.001)
    x_offset = x - coord_ax
    y_offset = y - coord_ay
    height_a = terrain_map[coord_ax, coord_ay]
    height_b = terrain_map[coord_ax + 1, coord_ay]
    height_c = terrain_map[coord_ax + 1, coord_ay + 1]
    height_d = terrain_map[coord_ax, coord_ay + 1]
    height = (1 - x_offset) * (1 - y_offset) * height_a + x_offset * (1 - y_offset) * height_b + \
             (1 - x_offset) * y_offset * height_d + x_offset * y_offset * height_c
    gradient_x = (height_b - height_a) * (1 - y_offset) + (height_c - height_d) * y_offset
    gradient_y = (height_d - height_a) * (1 - x_offset) + (height_c - height_b) * x_offset
    return HeightAndGradient(height, gradient_x, gradient_y)


brush_radius = 4
deposit_speed = .3
erode_speed = .3


def erode_at(x: float, y: float, amount: float):
    range_x = [xx for xx in range(int(x) - brush_radius - 1, int(x) + brush_radius + 2) if 0 <= xx < map_size]
    range_y = [yy for yy in range(int(y) - brush_radius - 1, int(y) + brush_radius + 2) if 0 <= yy < map_size]
    erosion_points = [[point[0], point[1], 1 - math.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) / brush_radius] for point in np.transpose([
        np.tile(range_x, len(range_y)), np.repeat(range_y, len(range_x))
    ]) if math.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) < brush_radius + 0.001]
    weights_sum = np.sum(erosion_points, axis=0)[2]
    delta_sediment = 0
    for point in erosion_points:
        d_sediment = min(terrain_map[point[0], point[1]], amount * (point[2] / weights_sum))
        terrain_map[point[0], point[1]] -= d_sediment
        delta_sediment += d_sediment
    return delta_sediment


def deposit_at(x: float, y: float, amount: float):
    range_x = [xx for xx in range(int(x) - brush_radius - 1, int(x) + brush_radius + 2) if 0 <= xx < map_size]
    range_y = [yy for yy in range(int(y) - brush_radius - 1, int(y) + brush_radius + 2) if 0 <= yy < map_size]
    depositing_points = [[point[0], point[1], 1 - math.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) / brush_radius] for point in np.transpose([
        np.tile(range_x, len(range_y)), np.repeat(range_y, len(range_x))
    ]) if math.sqrt((point[0] - x) ** 2 + (point[1] - y) ** 2) < brush_radius + 0.001]
    weights_sum = np.sum(depositing_points, axis=0)[2]
    for point in depositing_points:
        d_sediment = amount * (point[2] / weights_sum)
        terrain_map[point[0], point[1]] += d_sediment


def display_terrain():
    xx = range(map_size)
    yy = range(map_size)
    xx, yy = np.meshgrid(xx, yy)
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(1, 1)
    gs.update(left=0.025, right=0.975, top=0.975, bottom=0.025, wspace=0.025, hspace=0.025)
    ax_before = fig.add_subplot(gs[0], projection='3d')
    ax_before.set_zlim(0, map_size)
    ls_before = LightSource(270, 45)
    rgb_before = ls_before.shade(terrain_map, cmap=cm.gist_earth, vert_exag=0.1, blend_mode='soft')
    ax_before.plot_surface(yy, xx, terrain_map, rstride=1, cstride=1, facecolors=rgb_before,
                           linewidth=1, antialiased=False, shade=True)
    plt.show()


# Erode terrain
num_iterations = 7000
display_terrain_every = 1000
max_lifetime = 30
inertia = .05
sediment_capacity_factor = 4.
min_sediment_capacity = .01
gravity = -4.
evaporate_speed = .01
for iteration in range(num_iterations):
    if iteration % display_terrain_every == 0:
        display_terrain()
    pos_x = np.random.randint(0, map_size)
    pos_y = np.random.randint(0, map_size)
    dir_x = 0.
    dir_y = 0.
    speed = 1.
    water = 1.
    sediment = 0.
    for _ in range(max_lifetime):
        height_and_gradient = get_height_and_gradient(pos_x, pos_y)
        dir_x = (dir_x * inertia - height_and_gradient.gradient_x * (1 - inertia))
        dir_y = (dir_y * inertia - height_and_gradient.gradient_y * (1 - inertia))
        length = max(0.001, math.sqrt(dir_x ** 2 + dir_y ** 2))
        dir_x /= length
        dir_y /= length
        pos_x += dir_x
        pos_y += dir_y
        if pos_x < 0.001 or pos_x > map_size - 1.001 or pos_y < 0.001 or pos_y > map_size - 1.001:
            deposit_at(pos_x - dir_x, pos_y - dir_y, sediment)
            break
        height_after = get_height_and_gradient(pos_x, pos_y).height
        delta_height = height_after - height_and_gradient.height
        sediment_capacity = max(-delta_height * speed * water * sediment_capacity_factor, min_sediment_capacity)
        if sediment > sediment_capacity or delta_height > 0:
            amount_to_deposit = min(delta_height, sediment) if delta_height > 0 else ((sediment - sediment_capacity) * deposit_speed)
            deposit_at(pos_x - dir_x, pos_y - dir_y, amount_to_deposit)
            sediment -= amount_to_deposit
        else:
            amount_to_erode = min((sediment_capacity - sediment) * erode_speed, -delta_height)
            delta_sediment = erode_at(pos_x - dir_x, pos_y - dir_y, amount_to_erode)
            sediment += delta_sediment
        speed = math.sqrt(max(0., speed ** 2 + delta_height * gravity))
        water = water * (1 - evaporate_speed)
display_terrain()
