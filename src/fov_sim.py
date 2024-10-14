import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from util import pixel2coords
from map import Map
import cv2

origin = [0, 0]
num_zones = 200
crop_image = cv2.imread('data/crop_image.webp')
#crop_bounds = np.array([[528, 128][1338, 977]])

drone_path = np.linspace(0, crop_image.shape[1], 1000)
aspect_ratio = crop_image.shape[1] / crop_image.shape[0]

# map = Map(crop_image.shape[1], crop_image.shape[0], origin)
# zone_centroids, zone_width = map.zones(num_zones)


# fig, ax = plt.subplots()

# plt.imshow(crop_image)
# # plt.scatter(zone_centroids[:,0], zone_centroids[:,1], c = 'orange', s = 20)
# plt.scatter(origin[0], origin[1], s=100, c='r')
# plt.scatter(drone_path, np.zeros(len(drone_path)), s=1, c='b')
# plt.show()


fig, ax = plt.subplots()

def animate(i):
    fov_coords = pixel2coords([0, 0, 639, 639], [0, 639, 0, 639], 20, 30, 2, 0, 40)
    ax.clear()
    ax.imshow(crop_image)
    ax.scatter(drone_path[i], 100, c='b', s=100, marker='x')
    ax.scatter(fov_coords[0], fov_coords[1], c='r', s=50)
    ax.set_aspect('equal')

ani = FuncAnimation(fig, animate, frames=len(drone_path), repeat=False)

# save the animation as an mp4 video
plt.show()

