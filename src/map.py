import numpy as np

class Map(object):

    def __init__(self, width, length, origin):
        self.width = width
        self.length = length
        self.origin = origin

    def zones(self, num_zones):
        '''
        Devides the field into suqazones and returns their width
        '''

        field_area = self.width * self.length
        zone_width = np.sqrt(field_area / num_zones)

        # calculate the centroids of the zones
        num_zones_width = int(self.width / zone_width)
        num_zones_length = int(self.length / zone_width)     

        zone_centroids = np.zeros((num_zones, 2))
        for i in range(num_zones_width):
            for j in range(num_zones_length):
                zone_centroids[i * num_zones_length + j, 0] = (i + 0.5) * zone_width + self.origin[0]
                zone_centroids[i * num_zones_length + j, 1] = (j + 0.5) * zone_width + self.origin[1]

        return zone_centroids, zone_width

    def optimise_path(self):
        pass