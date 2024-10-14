import numpy as np
from ultralytics import YOLO
from drone import Drone
from map import Map
import pandas as pd
import request
import wait

class App(object):

    def __init__(self):
        self.model = YOLO('../data/yolov11n.pt')
        self.drone = Drone()
        self.path = iter(Map().optimise_path())
        self.results = pd.DataFrame(columns=['weed class', 'weed coords', 'model score'])

    def rotate(self, points, centre_x, centre_y, angle, degrees=True):
        '''
        Description
        ---
        Rotates the points by the angle
        '''

        x = points[0] - centre_x
        y = points[1] - centre_y
        cos = np.cos(angle*np.pi/180)
        sin = np.sin(angle*np.pi/180)

        return np.array([(x * cos - y * sin) + centre_x, (x * sin + y * cos) + centre_y])

    def pixel2coords(self, pixel_idx_x, pixel_idx_y):
        '''
        Calculates the physical positions of pixels in an image from the drone's location, height and field of view
        '''
        dx = self.drone.height * np.tan(self.drone.fov * np.pi / (2 * 180)) 
        coord_grid = np.meshgrid(np.linspace(-dx, dx, 640)+self.drone.x, np.linspace(-dx, dx, 640)+self.drone.y)
        # rotate the coord_grid
        if self.drone.angle != 0:
            coord_grid = self.rotate(coord_grid, self.drone.x, self.drone.y, self.drone.angle, degrees=False)
        pixel_loc_x = coord_grid[0][pixel_idx_x, pixel_idx_y]
        pixel_loc_y = coord_grid[1][pixel_idx_x, pixel_idx_y]
        return pixel_loc_x, pixel_loc_y

    def send_results(self, weed_coords, drone_coords):
        ''' 
        Sends the drones current position and results from the YOLO model to server
        '''
        
        url = 'webapp_url'

        data = {
            'drone_coords':  drone_coords,
            'weed_coords': weed_coords
        }
        
        try: 
            request.post(url, data)
        except Exception as e:
            wait(5)
            request.post(url, data)

    def run(self):
        
        for next_position in self.path:
            self.drone.go_to(next_position)
            image_batch, drone_coords = self.drone.get_image_batch(self.model)
            image_batch_processed = self.preprocess(image_batch)
            results = self.model.predict(source=image_batch_processed, show=True)

            weed_coords = self.pixel2coords(results[0].boxes.xyxy[:, 0], 
                                       results[0].boxes.xyxy[:, 1], 
                                       self.drone.x, 
                                       self.drone.y, 
                                       self.drone.height, 
                                       self.drone.angle, 
                                       fov=40)
            
            self.results = self.results.append({'weed class': results[0].boxes.cls, 'weed coords': weed_coords, 'model score': results[0].boxes.conf}, ignore_index=True)
            self.send_results(weed_coords, drone_coords)
        else: 
                self.drone.return_home()

                
if __name__ == "__main__":
    app = App()
    app.run()
