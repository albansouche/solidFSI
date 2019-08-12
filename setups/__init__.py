

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
