import numpy as np
from dolfin import *

class obs_point(object):
    """Observation point object for monitoring displacement.
    """

    def __init__(self, name, point, value_names):

        self.name = name
        self.point = point
        self.value_names = value_names
        self.values = [0]*len(self.value_names)
        for i,_ in enumerate(self.values):
            self.values[i] = []

    def append(self, index, value):
        self.values[index].append(value)

    def save(self, folder):
        for i, value_name in enumerate(self.value_names):
            np.save(folder + "/{}_{}.npy".format(self.name, value_name), self.values[i])

    def move_point(self, d):
        coord = []
        for i in range(len(d)):
            coord.append(self.point[i]+d[i])
        coord = np.array(coord)
        self.point = Point(coord)

    def __str__(self):
        return "{}: ({})".format(self.name, self.point)


class Setup_base():

    def __init__(self):

        # Quantities assumed to be zero if not specified
        self.p_exp = []
        self.pre_press_val = []
        self.prestress = []
        self.body_force = []
        self.u0 = []
        self.v0 = []
        self.extension = ""
        self.obs_points = []

    def reshape_parent_mesh(self, **namespace):
        """
        """
        pass

    def post_process(self, **namespace):
        """
        """
        pass
