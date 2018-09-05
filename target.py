import numpy as np
from numpy.random import uniform,normal

def render_target(radius=4):
    """
    generates and returns target marker
    values around target position are calculated as function of distance to center.
    center region is set to 0 to have more distinct features (edges) present
    """
    t = np.zeros([radius*2,radius*2])
    center = np.array([radius-.5,radius-.5])
    for i in range(radius*2):
        for j in range(radius*2):
            distance = np.abs(np.linalg.norm(center-np.array([i,j])))
            t[i,j] = np.clip((radius-distance)/radius,0,1)
    t[radius,radius] = 0
    t[radius-1,radius] = 0
    t[radius,radius-1] = 0
    t[radius-1,radius-1] = 0
    return t

class Target(object):

    def __init__(self, shape, static=True, speed_range=[.5,.75]):
        """
        Sets up target with marker (shape), radius, whether its static or moving and its speed range.
        """
        self.static = static
        if self.static:
            self.ID = 'static'
        else:
            self.ID = 'moving'
            self.speed_range = speed_range

        self.shape = shape
        self.radius = int(np.shape(shape)[0]/2)

        self.reset() # initialize to start of episode

    def reset(self):
        """ Resets direction, movement path and speed """
        self.direction = np.array([uniform(-1,1),uniform(-1,1)])
        self.path = []
        if self.static:
            self.speed = 0
        else:
            self.speed = uniform(self.speed_range[0],self.speed_range[1])
        # normalize direction vector to speed
        self.direction = (self.direction / np.linalg.norm(self.direction)) * self.speed

    def set_location(self, location):
        """ Sets location of target by appending location to path """
        self.path.append(location)

    def get_location(self, step=-1):
        """ Returns location for a given simulation step """
        if step != -1 and len(self.path) == 1:
            return self.path[-1]
        return self.path[step]

    def get_path(self):
        """ returns complete movement path of target """
        return self.path

    def update_direction(self):
        """ Updates direction by adding Gaussian noise and normalizes it to speed """
        self.direction += normal(0,.05,2)
        self.direction = (self.direction / np.linalg.norm(self.direction)) * self.speed

    def step(self):
        """ Performs one simulation step for target by appending its current
        location + movement direction to path, updates its direction """
        tmp = self.path[-1].copy()
        tmp += self.direction
        self.path.append(tmp)
        self.update_direction()
