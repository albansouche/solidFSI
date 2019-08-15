import numpy as np


class obs_point(object):
    """Observation point object for monitoring displacement.
    """

    def __init__(self, name, point):

        self.name = name
        self.point = point
        self.values = []

    def append(self, value):

        self.values.append(value)

    def save(self, folder):
        np.save(folder + "/{}.npy".format(self.name), self.values)


class Setup_base():

    def __init__(self):

        # Quantities assumed to be zero if not specified
        self.p_exp = []
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
