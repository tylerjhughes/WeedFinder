import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from map import Map
import cv2
def rotate(points, centre_x, centre_y, angle):
        '''
        Rotates the points by the angle (degrees)
        '''

        x = points[0] - centre_x
        y = points[1] - centre_y
        cos = np.cos(angle*np.pi/180)
        sin = np.sin(angle*np.pi/180)

        return np.array([(x * cos - y * sin) + centre_x, (x * sin + y * cos) + centre_y])

def pixel2coords(pixel_idx_x, pixel_idx_y, drone_x_pos, drone_y_pos, drone_height, drone_angle, fov):
    '''
    Calculates the physical positions of pixels in an image from the drone's location, height, heading, and field of view
    '''
    dx = drone_height * np.tan(fov* np.pi / (2 * 180)) 
    coord_grid = np.meshgrid(np.linspace(-dx, dx, 640)+drone_x_pos, np.linspace(-dx, dx, 640)+drone_y_pos)
    # rotate the coord_grid
    if drone_angle != 0:
        coord_grid = rotate(coord_grid, drone_x_pos, drone_y_pos, drone_angle, degrees=False)
    pixel_loc_x = coord_grid[0][pixel_idx_x, pixel_idx_y]
    pixel_loc_y = coord_grid[1][pixel_idx_x, pixel_idx_y]
    return pixel_loc_x, pixel_loc_y

def fov_simulation():
    

    origin = [0, 0]
    num_zones = 200
    crop_image = cv2.imread('data/crop_image.webp')
    #crop_bounds = np.array([[528, 128][1338, 977]])

    drone_x_path = np.linspace(0, 10, 100)
    drone_y = 5
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
        fov_coords = pixel2coords([0, 0, 639, 639], [0, 639, 0, 639], drone_x_path[i], drone_y, 2, 0, 40)
        ax.clear()
        #ax.imshow(crop_image)
        ax.scatter(drone_x_path[i], drone_y, c='b', s=100, marker='x')
        ax.scatter(fov_coords[0], fov_coords[1], c='r', s=10)
        ax.set_aspect('equal')
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    ani = FuncAnimation(fig, animate, frames=len(drone_x_path), repeat=False)

    ani.save("fov_sim.mp4", writer='ffmpeg', fps=30)

if __name__ == '__main__':
    fov_simulation()